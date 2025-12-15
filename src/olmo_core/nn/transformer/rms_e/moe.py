"""
DeepSeekMoE - Official Implementation

Reference:
- DeepSeek-MoE: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AddAuxiliaryLoss


class MoEGate(nn.Module):
    """
    MoE Gate with Loss-Free Balancing (DeepSeek-V3 style)

    Reference: DeepSeek-V3 official implementation
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

    Key pattern from official code (Gate class, line 566-598):
    1. scores = linear(x, weight).softmax()  # or sigmoid
    2. original_scores = scores  # save for weight computation
    3. if bias: scores = scores + bias  # add bias for routing only
    4. indices = topk(scores)  # route based on biased scores
    5. weights = original_scores.gather(indices)  # weight from original scores
    6. weights *= route_scale  # optional scaling
    """
    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int = 2,
        aux_loss_alpha: float = 0.001,  # For monitoring only
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        # Loss-Free Balancing parameters (DeepSeek-V3 official)
        use_loss_free_balancing: bool = True,
        bias_update_rate: float = 0.001,  # u in the paper
        route_scale: float = 1.0,  # From DeepSeek-V3 official
        score_func: str = "softmax",  # "softmax" or "sigmoid"
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.gating_dim = hidden_size
        self.route_scale = route_scale
        self.score_func = score_func

        # Loss-Free Balancing (DeepSeek-V3)
        self.use_loss_free_balancing = use_loss_free_balancing
        self.bias_update_rate = bias_update_rate

        # Gating weight
        self.weight = nn.Parameter(torch.empty((n_routed_experts, hidden_size)))

        # Expert bias for Loss-Free Balancing
        # DeepSeek-V3 official uses nn.Parameter with float32 dtype
        # We use buffer since it shouldn't receive gradient (updated via custom rule)
        if use_loss_free_balancing:
            # Initialize with zeros, will be updated during training
            self.register_buffer('expert_bias', torch.zeros(n_routed_experts, dtype=torch.float32))
        else:
            self.register_buffer('expert_bias', None)

        # Track expert load for bias update
        self.register_buffer('expert_load', torch.zeros(n_routed_experts))
        self.register_buffer('total_tokens', torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def update_expert_bias(self):
        """
        Update expert bias based on load imbalance (Loss-Free Balancing).

        Algorithm from arXiv:2408.15664:
        1. e_i = avg_load - load_i  (load violation error)
        2. b_i = b_i + u * sign(e_i)  (bias update)
        """
        if not self.use_loss_free_balancing or self.total_tokens.item() == 0:
            return

        # Calculate average load per expert
        avg_load = self.total_tokens.item() / self.n_routed_experts

        # Calculate load violation error: e_i = avg - actual
        load_error = avg_load - self.expert_load

        # Update bias: b_i = b_i + u * sign(e_i)
        # Heavy-load experts (negative error) get decreased bias
        # Light-load experts (positive error) get increased bias
        self.expert_bias.add_(self.bias_update_rate * torch.sign(load_error))

        # Reset load counters
        self.expert_load.zero_()
        self.total_tokens.zero_()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass following DeepSeek-V3 official pattern.

        Official code reference (Gate.forward, line 566-598):
        1. scores = linear(x, weight)
        2. scores = scores.softmax() or scores.sigmoid()
        3. original_scores = scores
        4. if bias: scores = scores + bias
        5. indices = topk(scores)
        6. weights = original_scores.gather(indices)
        7. if sigmoid: weights /= weights.sum()
        8. weights *= route_scale
        """
        bsz, seq_len, h = hidden_states.shape
        num_tokens = bsz * seq_len

        # Step 1: Compute gating logits
        hidden_states_flat = hidden_states.view(-1, h)
        scores = F.linear(hidden_states_flat, self.weight, None)

        # Step 2: Apply score function (DeepSeek-V3 uses float32 for softmax)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:  # sigmoid
            scores = scores.sigmoid()

        # Step 3: Save original scores for weight computation
        original_scores = scores

        # Step 4: Loss-Free Balancing - add bias for routing decision only
        if self.expert_bias is not None:
            scores = scores + self.expert_bias.unsqueeze(0)

        # Step 5: Top-K selection on (possibly biased) scores
        topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)[1]

        # Step 6: Get weights from ORIGINAL scores (without bias)
        topk_weight = original_scores.gather(dim=-1, index=topk_idx)

        # Step 7: Normalize weights
        if self.score_func == "sigmoid":
            # DeepSeek-V3: sigmoid uses sum normalization
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        elif self.top_k > 1 and self.norm_topk_prob:
            # Softmax: optional normalization
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # Step 8: Apply route scale (DeepSeek-V3 official)
        topk_weight = topk_weight * self.route_scale

        # Track expert load for bias update (Loss-Free Balancing)
        if self.training and self.use_loss_free_balancing:
            expert_counts = torch.zeros(self.n_routed_experts, device=hidden_states.device)
            expert_counts.scatter_add_(
                0, topk_idx.view(-1),
                torch.ones(num_tokens * self.top_k, device=hidden_states.device)
            )
            self.expert_load.add_(expert_counts)
            self.total_tokens.add_(num_tokens * self.top_k)

        # Compute aux_loss for monitoring (gradient detached with Loss-Free Balancing)
        aux_loss = None
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = original_scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device)
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = original_scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

            # With Loss-Free Balancing, aux_loss is for monitoring only (no gradient)
            if self.use_loss_free_balancing:
                aux_loss = aux_loss.detach()

        # Cast weight back to input dtype
        topk_weight = topk_weight.type_as(hidden_states)

        return topk_idx, topk_weight, aux_loss


class DeepSeekMLP(nn.Module):
    """SwiGLU MLP - DeepSeek style"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with Loss-Free Balancing (DeepSeek-V3 style)

    Reference: DeepSeek-V3 official implementation
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

    From official MoE class (line 636-693):
    - Gate for routing
    - Routed experts (local in distributed setting)
    - Shared experts (applied to all tokens)
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_routed_experts: int,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 2,
        aux_loss_alpha: float = 0.001,
        # Loss-Free Balancing (DeepSeek-V3 official)
        use_loss_free_balancing: bool = True,
        bias_update_rate: float = 0.001,
        route_scale: float = 1.0,
        score_func: str = "softmax",
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.use_loss_free_balancing = use_loss_free_balancing

        # Routed experts
        self.experts = nn.ModuleList([
            DeepSeekMLP(hidden_size, intermediate_size)
            for _ in range(n_routed_experts)
        ])

        # Gate with Loss-Free Balancing (DeepSeek-V3 official pattern)
        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            aux_loss_alpha=aux_loss_alpha,
            use_loss_free_balancing=use_loss_free_balancing,
            bias_update_rate=bias_update_rate,
            route_scale=route_scale,
            score_func=score_func,
        )

        # Shared experts (always activated) - DeepSeek-V3 official pattern
        if n_shared_experts > 0:
            shared_intermediate = intermediate_size * n_shared_experts
            self.shared_experts = DeepSeekMLP(hidden_size, shared_intermediate)
        else:
            self.shared_experts = None

    def update_expert_bias(self):
        """Update expert bias after each training batch (Loss-Free Balancing)."""
        self.gate.update_expert_bias()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # Training: process all expert assignments
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    expert_out = expert(hidden_states[mask])
                    # Under autocast, experts may produce bf16 outputs even if inputs are fp32.
                    # Advanced indexing assignment requires exact dtype match.
                    if expert_out.dtype != y.dtype:
                        expert_out = expert_out.to(y.dtype)
                    y[mask] = expert_out
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # Cast back to input dtype (sum() may promote bf16 to float32 for numerical stability)
            y = y.to(identity.dtype)
            y = y.view(*orig_shape)
            if aux_loss is not None:
                y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            # Inference: optimized computation
            y = self._moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1))
            y = y.view(*orig_shape)

        # Add shared expert contributions
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y, aux_loss

    @torch.no_grad()
    def _moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor) -> torch.Tensor:
        """Optimized inference computation - DeepSeek official"""
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            # Keep dtypes aligned for in-place ops and scatter_reduce.
            if expert_out.dtype != expert_cache.dtype:
                expert_out = expert_out.to(expert_cache.dtype)
            weights = flat_expert_weights[idxs[start_idx:end_idx]]
            if weights.dtype != expert_out.dtype:
                weights = weights.to(expert_out.dtype)
            expert_out.mul_(weights)
            expert_cache.scatter_reduce_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out, reduce='sum'
            )
        return expert_cache


class DenseFFN(nn.Module):
    """
    Dense FFN (SwiGLU) - Alternative to MoE for smaller models.

    Same interface as DeepSeekMoE for easy swapping.
    Uses standard SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        **kwargs,  # Accept but ignore MoE-specific args
    ):
        super().__init__()
        self.mlp = DeepSeekMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching MoE interface.

        Returns:
            output: [batch, seq, hidden]
            aux_loss: None (no auxiliary loss for Dense)
        """
        output = self.mlp(hidden_states)
        return output, None


__all__ = [
    "MoEGate",
    "DeepSeekMLP",
    "DeepSeekMoE",
    "DenseFFN",
]
