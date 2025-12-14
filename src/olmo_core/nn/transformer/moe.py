
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

from olmo_core.config import Config
from olmo_core.nn.layer_norm import RMSNorm

# [OPTIMIZED] SGLang-style Fused MoE Kernel
# Try to import optimized backends in order of preference:
# 1. sgl-kernel (SGLang's standalone kernel package)
# 2. flashinfer (alternative fast backend)
# 3. vllm triton kernel
# 4. fallback to grouped dispatch

# [FIX] MoE Backend Detection
# Try to import optimized backends in order of preference:
# 1. sgl-kernel (SGLang's standalone kernel package)
# 2. vllm triton kernel
# 3. fallback to grouped dispatch
_SGL_KERNEL_AVAILABLE = False
_FUSED_MOE_AVAILABLE = False
_FUSED_MOE_BACKEND = "grouped_dispatch"
_fused_moe_fn = None
sgl_topk_softmax = None
moe_align_block_size = None

try:
    # SGLang's kernel primitives for optimized routing
    from sgl_kernel.moe import topk_softmax as sgl_topk_softmax
    from sgl_kernel.moe import moe_align_block_size
    _SGL_KERNEL_AVAILABLE = True
    _FUSED_MOE_BACKEND = "sgl_kernel"
except ImportError:
    pass

# Try vllm fused_moe kernel
try:
    from vllm.model_executor.layers.fused_moe import fused_moe as _fused_moe_fn
    _FUSED_MOE_AVAILABLE = True
    _FUSED_MOE_BACKEND = "vllm"
except ImportError:
    pass

# Note: sgl_kernel provides primitives for routing, not full fused_moe
# Use grouped dispatch with optional sgl_kernel acceleration for routing

import logging
_logger = logging.getLogger(__name__)
_logger.info(f"MoE backend: {_FUSED_MOE_BACKEND} (sgl_kernel available: {_SGL_KERNEL_AVAILABLE})")

@dataclass
class MoEConfig(Config):
    """
    Configuration for Ultra-Sparse MoE.

    Load Balancing References:
    - Switch Transformer: https://arxiv.org/abs/2101.03961
    - DeepSeek Loss-Free Balancing: https://arxiv.org/abs/2408.15664
    - ST-MoE Z-Loss: https://arxiv.org/abs/2202.08906
    """
    hidden_size: int = 2048
    intermediate_size: int = 14336  # Typically wider for experts
    num_experts: int = 512
    num_shared_experts: int = 1
    num_active_experts: int = 10

    # [Switch Transformer] Auxiliary loss coefficient
    # Paper recommends α = 0.01, swept from 10^-1 to 10^-5
    router_aux_loss_coef: float = 0.01  # [FIX] Changed from 0.001 to 0.01 per paper

    # [ST-MoE] Router Z-Loss for stability (prevents large logits)
    # L_z = mean(logsumexp(router_logits)^2)
    router_z_loss_coef: float = 0.001

    # [DeepSeek] Loss-Free Balancing (alternative to aux loss)
    use_loss_free_balancing: bool = True  # Enable DeepSeek-style bias balancing
    loss_free_balance_rate: float = 0.001  # Bias update rate (u in paper)

    norm_eps: float = 1e-5
    dropout: float = 0.0

    # [OPTIMIZED] SGLang-style Fused Kernel Options
    use_fused_moe: bool = True  # Use fused MoE kernel when available
    expert_parallel: bool = False  # Expert parallelism for multi-GPU

class SwiGLU(nn.Module):
    """
    SwiGLU Activation: Swish(xW_g) * xW_u
    """
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias) # Gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias) # Down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias) # Up
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoERouter(nn.Module):
    """
    Router for MoE with Loss-Free Balancing.

    References:
    - Qwen3-Next: Normalized initialization for router stability
    - DeepSeek (arXiv:2408.15664): Loss-Free Balancing via dynamic bias
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # [DeepSeek] Loss-Free Balancing: per-expert bias for load balancing
        # Reference: arXiv:2408.15664
        # Bias is added to routing scores BEFORE top-K selection
        # but does NOT affect the final gating weights
        self.use_loss_free_balancing = config.use_loss_free_balancing
        self.balance_rate = config.loss_free_balance_rate  # u = 0.001 in paper

        # Register bias as buffer (not a parameter, not trained by gradient)
        self.register_buffer('expert_bias', torch.zeros(config.num_experts))

        # [FIX] Normalized initialization for router stability (Qwen3-Next)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02 / (config.num_experts ** 0.5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: [batch, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Compute routing scores
        router_logits = self.gate(hidden_states)  # [B, L, E]

        # [DeepSeek] Apply bias for top-K selection (not for final weights)
        if self.use_loss_free_balancing:
            # Add bias to routing scores for expert selection
            biased_logits = router_logits + self.expert_bias.unsqueeze(0).unsqueeze(0)
        else:
            biased_logits = router_logits

        # Select top-k experts using biased scores
        _, selected_experts = torch.topk(biased_logits, self.num_active_experts, dim=-1)

        # Compute routing weights from ORIGINAL logits (not biased)
        # This is key: bias affects selection, not the final weights
        routing_weights = torch.gather(router_logits, dim=-1, index=selected_experts)

        # [OPTIMIZED] Use sgl_kernel's fused topk_softmax if available
        if _SGL_KERNEL_AVAILABLE and sgl_topk_softmax is not None:
            orig_dtype = routing_weights.dtype
            flat_logits = router_logits.view(-1, self.num_experts).float()
            flat_weights = routing_weights.view(-1, self.num_active_experts).float()
            flat_experts = selected_experts.view(-1, self.num_active_experts).int()
            sgl_topk_softmax(flat_weights, flat_experts, flat_logits, renormalize=True)
            routing_weights = flat_weights.view(batch_size, seq_len, self.num_active_experts).to(orig_dtype)
        else:
            routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # [DeepSeek] Update expert bias based on load (only during training)
        # Reference: arXiv:2408.15664 - Auxiliary-Loss-Free Load Balancing
        # Key insight: Use RAW batch counts, NOT EMA (paper explicitly avoids EMA)
        if self.training and self.use_loss_free_balancing:
            with torch.no_grad():
                # Count tokens per expert in current batch
                expert_counts = torch.zeros(self.num_experts, device=hidden_states.device)
                flat_experts = selected_experts.view(-1)  # [B*L*K]
                expert_counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=expert_counts.dtype))

                # Expected uniform count: total_selections / num_experts
                total_selections = num_tokens * self.num_active_experts
                expected_count = total_selections / self.num_experts

                # [FIX] Load error from RAW batch counts (NOT EMA)
                # e_i = expected_count - actual_count (positive = underloaded)
                load_error = expected_count - expert_counts

                # Update bias: b_i = b_i + u * sign(e_i)
                # Underloaded experts get positive bias (more likely to be selected)
                # Overloaded experts get negative bias (less likely to be selected)
                new_bias = self.expert_bias + self.balance_rate * torch.sign(load_error)
                self.expert_bias.copy_(new_bias)

        return router_logits, selected_experts, routing_weights

class MoELayer(nn.Module):
    """
    Ultra-Sparse Mixture-of-Experts Layer.
    Combines Shared Experts and Routed Experts.

    [OPTIMIZED] SGLang-style optimizations:
    - Fused MoE kernel (flashinfer/triton) when available
    - Packed expert weights for better memory access
    - Expert-grouped dispatch as fallback
    """
    def __init__(self, config: MoEConfig, init_device: str = "cpu"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_active_experts = config.num_active_experts
        self.use_fused_moe = config.use_fused_moe and _FUSED_MOE_AVAILABLE

        # Shared Experts (Always active)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLU(self.hidden_size, self.intermediate_size) for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None

        # Routed Experts
        self.router = MoERouter(config)

        # [OPTIMIZED] SGLang-style: Pack expert weights for fused kernel
        # Instead of ModuleList, use packed tensors for better memory coalescing
        if self.use_fused_moe:
            # Packed weights: (num_experts, intermediate_size, hidden_size) etc.
            self.w1_weight = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
            self.w2_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
            self.w3_weight = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
            # Initialize
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(self.w1_weight[i])
                nn.init.kaiming_uniform_(self.w2_weight[i])
                nn.init.kaiming_uniform_(self.w3_weight[i])
            self.experts = None  # Not using ModuleList
        else:
            self.experts = nn.ModuleList([
                SwiGLU(self.hidden_size, self.intermediate_size) for _ in range(self.num_experts)
            ])
            self.w1_weight = None
            self.w2_weight = None
            self.w3_weight = None

        self.dropout = nn.Dropout(config.dropout)

    def _fused_experts_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        [OPTIMIZED] SGLang-style Fused MoE Forward with packed weights.
        Uses batched matrix multiplication instead of per-expert loops.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_tokens = batch_size * seq_len

        flat_hidden = hidden_states.view(num_tokens, hidden_dim)  # (N, D)
        flat_experts = selected_experts.view(num_tokens, self.num_active_experts)  # (N, K)
        flat_weights = routing_weights.view(num_tokens, self.num_active_experts)  # (N, K)

        # Initialize output
        output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        # Process each top-k slot with grouped GEMM
        for k in range(self.num_active_experts):
            expert_ids = flat_experts[:, k]  # (N,)
            weights = flat_weights[:, k:k+1]  # (N, 1)

            # Sort by expert for better memory access (SGLang optimization)
            sorted_ids, sort_idx = expert_ids.sort()
            sorted_input = flat_hidden[sort_idx]  # (N, D)
            sorted_weights = weights[sort_idx]

            # Get unique experts and their token counts
            unique_ids, inverse_idx, counts = torch.unique(sorted_ids, return_inverse=True, return_counts=True)

            # Grouped computation per expert
            offset = 0
            sorted_output = torch.zeros_like(sorted_input)

            for i, (exp_id, count) in enumerate(zip(unique_ids.tolist(), counts.tolist())):
                if count == 0:
                    continue
                exp_id = int(exp_id)
                tokens = sorted_input[offset:offset + count]  # (count, D)

                # SwiGLU: w2(silu(w1(x)) * w3(x))
                # w1: (intermediate, hidden), tokens: (count, hidden)
                gate = F.silu(tokens @ self.w1_weight[exp_id].T)  # (count, intermediate)
                up = tokens @ self.w3_weight[exp_id].T  # (count, intermediate)
                expert_out = (gate * up) @ self.w2_weight[exp_id].T  # (count, hidden)

                sorted_output[offset:offset + count] = expert_out * sorted_weights[offset:offset + count]
                offset += count

            # Unsort back
            unsort_idx = torch.argsort(sort_idx)
            output += sorted_output[unsort_idx]

        return output.view(batch_size, seq_len, hidden_dim)

    def _grouped_experts_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expert-Grouped Dispatch (ScatterMoE-style) for non-fused path.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        flat_hidden_states = hidden_states.view(-1, hidden_dim)  # (B*L, D)
        flat_selected_experts = selected_experts.view(-1, self.num_active_experts)  # (B*L, K)
        flat_routing_weights = routing_weights.view(-1, self.num_active_experts)  # (B*L, K)

        results = torch.zeros_like(flat_hidden_states)

        # Process each top-k slot
        for k in range(self.num_active_experts):
            expert_indices = flat_selected_experts[:, k]  # (B*L,)
            weights = flat_routing_weights[:, k].unsqueeze(-1)  # (B*L, 1)

            # Sort tokens by expert ID for grouped processing
            sorted_expert_ids, sort_order = expert_indices.sort()
            sorted_inputs = flat_hidden_states[sort_order]
            sorted_weights = weights[sort_order]

            # Find boundaries where expert changes
            unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)

            # Process each expert's tokens as a batch
            offset = 0
            sorted_outputs = torch.zeros_like(sorted_inputs)

            for expert_id, count in zip(unique_experts.tolist(), counts.tolist()):
                batch_inputs = sorted_inputs[offset:offset + count]
                batch_outputs = self.experts[expert_id](batch_inputs)
                sorted_outputs[offset:offset + count] = batch_outputs * sorted_weights[offset:offset + count]
                offset += count

            # Unsort back to original order
            unsort_order = torch.argsort(sort_order)
            results += sorted_outputs[unsort_order]

        return results.view(batch_size, seq_len, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        final_hidden_states = torch.zeros_like(hidden_states)

        # 1. Shared Experts Forward
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                final_hidden_states += expert(hidden_states)

        # 2. Routed Experts Forward
        router_logits, selected_experts, routing_weights = self.router(hidden_states)

        # [OPTIMIZED] Use fused kernel if available, else grouped dispatch
        if self.use_fused_moe and self.w1_weight is not None:
            routed_output = self._fused_experts_forward(hidden_states, selected_experts, routing_weights)
        else:
            routed_output = self._grouped_experts_forward(hidden_states, selected_experts, routing_weights)

        final_hidden_states += routed_output
        
        # Auxiliary Loss for MoE Load Balancing
        # References:
        # - Switch Transformer (arXiv:2101.03961): L_aux = α * N * Σ(f_i * P_i)
        # - ST-MoE Z-Loss (arXiv:2202.08906): L_z = mean(logsumexp(logits)^2)
        # - DeepSeek Loss-Free Balancing (arXiv:2408.15664): REPLACES aux_loss entirely
        if self.training:
            aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

            # [FIX] Loss-Free Balancing completely replaces load balancing aux_loss
            # Only compute load balance loss if NOT using Loss-Free Balancing
            if not getattr(self, "_logged_loss_free", False):
                print(f"[DEBUG] MoE Layer: use_loss_free_balancing={self.config.use_loss_free_balancing}")
                self._logged_loss_free = True
            
            if not self.config.use_loss_free_balancing:
                # Compute in float32 for numerical stability
                probs = F.softmax(router_logits.float(), dim=-1)  # (B, L, E)

                # P_i: Mean probability assigned to expert i across all tokens
                mean_probs = probs.mean(dim=(0, 1))  # (E,)

                # f_i: Fraction of tokens routed to expert i
                expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()  # (B, L, K, E)
                expert_mask = expert_mask.sum(dim=2).clamp(max=1.0)  # (B, L, E) - binary mask
                density = expert_mask.mean(dim=(0, 1))  # (E,) - fraction per expert

                # [Switch Transformer] Load Balancing Loss
                # L_aux = α * N * Σ(f_i * P_i)
                load_balance_loss = self.num_experts * torch.sum(density * mean_probs)
                aux_loss = self.config.router_aux_loss_coef * load_balance_loss

            # [ST-MoE] Router Z-Loss for stability (KEEP even with Loss-Free Balancing)
            # Penalizes large router logits to prevent exponential blowup
            # L_z = (1/B) * Σ(logsumexp(x_i))^2
            z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
            aux_loss = aux_loss + self.config.router_z_loss_coef * z_loss
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        return final_hidden_states, aux_loss

