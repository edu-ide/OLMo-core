import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Union
import torch.utils.checkpoint

from olmo_core.config import Config
from olmo_core.nn.layer_norm import RMSNorm, LayerNormConfig
from olmo_core.nn.transformer.moe import MoEConfig, MoELayer
from olmo_core.nn.transformer.mtp import MTPConfig, MTPHead, MTPLoss

# [OPTIMIZED] FSDP2 support for multi-GPU training
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False

# [OPTIMIZED] FP8 Training with TransformerEngine (GB10/Blackwell native)
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    te = None

# Reuse ATLAS from existing file
try:
    from olmo_core.nn.transformer.atlas_memory import ATLASConfig, ATLASMemory
except ImportError:
    # Fallback if file doesn't exist yet (for testing isolation)
    ATLASConfig = Any
    ATLASMemory = None

@dataclass
class RMS_E_Config(Config):
    """
    Configuration for RMS-E (Recursive Memory-Sparse Experts) Architecture.
    """
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: Optional[int] = None # Support explicit head dim (e.g. for Qwen)
    # [VERIFIED] RoPE Percentage:
    # Qwen uses 100% RoPE (rotary_dim == head_dim).
    # Llama uses 50% or 100% depending on impl.
    # Default here MUST be 1.0 for Qwen compatibility, or position info is corrupted.
    rope_percentage: float = 1.0 
    rope_theta: float = 1000000.0 # Qwen 2.5/3 uses 1M theta.
    
    # LoopLM / Ouro Settings
    loop_max_steps: int = 40  # Maximum recursion depth (T_max)
    entropy_beta: float = 0.01 # Beta for entropy regularization
    
    # MoE Settings
    num_shared_experts: int = 1
    num_routed_experts: int = 512
    num_active_experts: int = 10
    router_aux_loss_coef: float = 0.001 # [VERIFIED] Qwen3/DeepSeekMoE default (1e-3). 0.01 was too high.
    router_z_loss_coef: float = 0.001 # [VERIFIED] Z-Loss for logit stability.
    
    # Memory Settings
    # NOTE: Titans/ATLAS DISABLED by default until official implementations are released.
    # Current implementations are based on papers but not verified against official code.
    # - Titans (arXiv:2501.00663): No official Google release yet
    # - ATLAS (arXiv:2505.23735): No official Google release yet
    # Enable at your own risk for experimental purposes.
    use_titans: bool = False  # DISABLED: No official implementation
    titans_memory_size: int = 4096  # Titans internal memory dim (D_mem)
    # [NEW] Titans Paper Implementation (arXiv:2501.00663)
    titans_num_persistent: int = 8     # N_p: Persistent memory slots (task knowledge)
    titans_memory_layers: int = 2      # L_M: Deep memory MLP layers (≥2 recommended)
    titans_conv_kernel: int = 4        # Convolution kernel size for Q/K/V

    # ATLAS Neural Memory with Omega Rule (arXiv:2505.23735).
    # Uses closed-form linear memory update (Eq. 11) or isolated gradient optimization.
    use_atlas: bool = False  # DISABLED: No official implementation
    atlas_config: Optional[ATLASConfig] = None

    # NOTE: SBT (Self-Braking Tuning) removed.
    # arXiv:2505.14604 describes SBT as a training technique, not inference module.
    # Early exit is handled by OuroExitGate (learned exit probability).

    # Qwen3-Next Hybrid Pattern
    # Reference: Qwen3-Next uses layer-based 3:1 ratio (75% DeltaNet, 25% Attention)
    # In recursive architecture, "layer" translates to "step"
    # Pattern: D, D, D, A, D, D, D, A, ... (every 4th step uses Attention)
    use_hybrid_loop: bool = True
    hybrid_attention_ratio: int = 4 # Every N-th step uses Attention (1:3 ratio = 25% attention)

    # NOTE: Mamba-3 integration deferred until official implementation is released
    # Paper: https://openreview.net/forum?id=HwCvaJOiCj (Oct 2025, under review)
    # GitHub Issue: https://github.com/state-spaces/mamba/issues/809

    # MTP (Multi-Token Prediction) Settings
    use_mtp: bool = True
    mtp_num_predict_tokens: int = 4  # Number of future tokens to predict
    mtp_head_type: str = "medusa"    # 'medusa', 'transformer', 'mlp'
    mtp_loss_weight: float = 0.3     # Weight for MTP auxiliary loss

    # Gradient Checkpointing
    # WARNING: MoE dynamic routing is INCOMPATIBLE with gradient checkpointing.
    # During backward, checkpoint recomputes forward, but MoE routes different
    # token counts to experts, causing shape mismatch.
    # Keep False when using MoE (num_routed_experts > 0).
    use_gradient_checkpointing: bool = False

    # [OPTIMIZED] FSDP2 Settings for multi-GPU training
    use_fsdp: bool = False  # Enable FSDP2 sharding
    fsdp_mixed_precision: str = "bf16"  # "fp32", "fp16", "bf16"
    fsdp_cpu_offload: bool = False  # Offload parameters to CPU

    # [OPTIMIZED] FP8 Training Settings (TransformerEngine)
    # GB10/Blackwell: Native FP8 support, 1.5-2x speedup
    use_fp8: bool = False  # Enable FP8 training
    fp8_format: str = "hybrid"  # "e4m3" (forward), "e5m2" (backward), "hybrid" (both)
    fp8_amax_history_len: int = 1024  # History length for dynamic scaling
    fp8_amax_compute_algo: str = "max"  # "max" or "most_recent"

    # [OPTIMIZED] MXFP8 Training Settings (Blackwell-specific)
    # MXFP8: Block-wise scaling (32 elements per block) for better precision
    # Reference: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
    use_mxfp8: bool = False  # Enable MXFP8 (Blackwell only, overrides use_fp8)
    mxfp8_margin: int = 0    # Margin for MXFP8 scaling

    vocab_size: int = 50257
    max_sequence_length: int = 2048
    norm_eps: float = 1e-6 # [VERIFIED] Qwen uses 1e-6.
    
    def __post_init__(self):
        if self.atlas_config is None and self.use_atlas and ATLASMemory is not None:
            self.atlas_config = ATLASConfig(hidden_size=self.hidden_size)


class TitansMemory(nn.Module):
    """
    Titans: Neural Long-term Memory (Full Implementation).
    Reference: arXiv:2501.00663 - 'Titans: Learning to Memorize at Test Time' (Google Research)

    Paper Components Implemented:
    1. Deep Memory MLP (L_M ≥ 2 layers) - "deep memory modules are more effective"
    2. 1D Depthwise-Separable Convolution - after Q, K, V projections
    3. L2 Normalization - on queries and keys
    4. Output Gating - linear gate before output projection
    5. Persistent Memory - learnable input-independent parameters
    6. Data-dependent η, θ, α - computed from input at chunk level

    Paper Formulas:
    - Surprise: St = ηt·St-1 - θt·∇ℓ(Mt-1; xt)
    - Memory Update: Mt = (1 - αt)·Mt-1 + St
    - Retrieval: yt = M*(qt) where qt = xt·WQ (inference without weight updates)
    - Loss: ℓ(Mt-1; xt) = ||Mt-1(kt) - vt||²₂

    IMPORTANT (Paper Section 3.2):
    "ηt, θt, αt are DATA-DEPENDENT functions of input xt"
    """
    def __init__(self, config: RMS_E_Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.memory_size = config.titans_memory_size
        self.num_persistent = getattr(config, 'titans_num_persistent', 8)  # N_p persistent memory slots
        self.memory_layers = getattr(config, 'titans_memory_layers', 2)  # L_M ≥ 2 recommended
        self.conv_kernel_size = getattr(config, 'titans_conv_kernel', 4)  # Convolution kernel size

        # ============================================================
        # 1. Projections with 1D Depthwise-Separable Convolution
        # Paper: "we incorporate a 1D depthwise-separable convolution layer
        #         after each of the query, key, and value projections"
        # ============================================================
        self.query_proj = nn.Linear(self.hidden_size, self.memory_size, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.memory_size, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, self.memory_size, bias=False)

        # Depthwise-separable convolution: depthwise (groups=channels) + pointwise (1x1)
        # Applied after projection to capture local patterns
        self.query_conv = nn.Sequential(
            nn.Conv1d(self.memory_size, self.memory_size, kernel_size=self.conv_kernel_size,
                     padding=self.conv_kernel_size - 1, groups=self.memory_size, bias=False),
            nn.Conv1d(self.memory_size, self.memory_size, kernel_size=1, bias=False),
            nn.SiLU(),
        )
        self.key_conv = nn.Sequential(
            nn.Conv1d(self.memory_size, self.memory_size, kernel_size=self.conv_kernel_size,
                     padding=self.conv_kernel_size - 1, groups=self.memory_size, bias=False),
            nn.Conv1d(self.memory_size, self.memory_size, kernel_size=1, bias=False),
            nn.SiLU(),
        )
        self.value_conv = nn.Sequential(
            nn.Conv1d(self.memory_size, self.memory_size, kernel_size=self.conv_kernel_size,
                     padding=self.conv_kernel_size - 1, groups=self.memory_size, bias=False),
            nn.Conv1d(self.memory_size, self.memory_size, kernel_size=1, bias=False),
            nn.SiLU(),
        )

        # ============================================================
        # 2. Persistent Memory (Paper Section 3.4)
        # "learnable but input-independent parameters" P = [p₁, p₂, ..., p_Np]
        # Stores "task knowledge" separate from contextual information
        # ============================================================
        self.persistent_keys = nn.Parameter(torch.randn(self.num_persistent, self.memory_size) * 0.02)
        self.persistent_values = nn.Parameter(torch.randn(self.num_persistent, self.memory_size) * 0.02)

        # ============================================================
        # 3. Deep Memory MLP (Paper Section 3.1)
        # "deep neural long-term memory as simple MLPs with L_M ≥ 1 layers"
        # "deep memory modules (i.e., L_M ≥ 2) are more effective in practice"
        # ============================================================
        memory_mlp_layers = []
        for i in range(self.memory_layers):
            if i == 0:
                memory_mlp_layers.append(nn.Linear(self.memory_size, self.memory_size, bias=False))
            else:
                memory_mlp_layers.append(nn.Linear(self.memory_size, self.memory_size, bias=False))
            if i < self.memory_layers - 1:  # No activation on last layer
                memory_mlp_layers.append(nn.SiLU())
        self.memory_mlp = nn.Sequential(*memory_mlp_layers)

        # Initialize memory MLP close to identity for stability
        for layer in self.memory_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                layer.weight.data *= 0.1  # Scale down for gradual learning

        # ============================================================
        # 4. Data-dependent Parameters (Paper Section 3.2)
        # "ηt, θt, αt are data-dependent functions of input xt"
        # Can be "functions of tokens" or "functions of chunks" for efficiency
        # ============================================================
        self.eta_proj = nn.Linear(self.hidden_size, 1, bias=True)    # η_t: surprise decay
        self.theta_proj = nn.Linear(self.hidden_size, 1, bias=True)  # θ_t: momentary surprise scale
        self.alpha_proj = nn.Linear(self.hidden_size, 1, bias=True)  # α_t: forgetting factor

        # Initialize biases for reasonable default values after sigmoid
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.constant_(self.eta_proj.bias, 2.0)    # Default η ≈ 0.88 (high decay retention)
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.constant_(self.theta_proj.bias, -1.0)  # Default θ ≈ 0.27 (moderate surprise)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -2.0)  # Default α ≈ 0.12 (slow forgetting)

        # ============================================================
        # 5. Output Gating (Paper mentions "gating with a linear layer")
        # ============================================================
        self.output_gate = nn.Linear(self.memory_size, self.memory_size, bias=True)
        self.out_proj = nn.Linear(self.memory_size, self.hidden_size, bias=False)
        self.layer_norm = RMSNorm(size=config.hidden_size)

        # ============================================================
        # 6. Memory State Buffers
        # For deep memory, we store the MLP parameters as the "memory state"
        # We use a simpler approach: store key-value associations
        # ============================================================
        self.register_buffer('memory_state', torch.zeros(self.memory_size, self.memory_size))
        self.register_buffer('surprise_state', torch.zeros(self.memory_size, self.memory_size))
        self.register_buffer('_memory_snapshot', torch.zeros(self.memory_size, self.memory_size))

    def _apply_conv_causal(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution with causal masking (only look at past)."""
        # x: (B, L, D) -> (B, D, L) for Conv1d
        x_t = x.transpose(1, 2)
        out = conv(x_t)
        # Causal: trim future padding
        out = out[..., :x.shape[1]]
        return out.transpose(1, 2)

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalize along specified dimension (Paper: "we normalize queries and keys")."""
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, D)
        Returns:
            output: Memory-augmented output (B, L, D)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # ============================================================
        # Step 1: Project and apply convolution
        # ============================================================
        q = self.query_proj(x)   # (B, L, D_mem)
        k = self.key_proj(x)     # (B, L, D_mem)
        v = self.value_proj(x)   # (B, L, D_mem)

        # Apply depthwise-separable convolution (causal)
        q = self._apply_conv_causal(self.query_conv, q)
        k = self._apply_conv_causal(self.key_conv, k)
        v = self._apply_conv_causal(self.value_conv, v)

        # ============================================================
        # Step 2: L2 Normalize queries and keys
        # Paper: "we normalize queries and keys using ℓ₂-norm"
        # ============================================================
        q = self._l2_normalize(q, dim=-1)
        k = self._l2_normalize(k, dim=-1)

        # ============================================================
        # Step 3: Compute data-dependent parameters (chunk-level)
        # Paper: "parameters can be made functions of their chunk"
        # ============================================================
        x_pooled = x.mean(dim=1)  # (B, D)
        eta = torch.sigmoid(self.eta_proj(x_pooled))      # (B, 1) - surprise decay
        theta = torch.sigmoid(self.theta_proj(x_pooled))  # (B, 1) - learning rate
        alpha = torch.sigmoid(self.alpha_proj(x_pooled))  # (B, 1) - forget gate

        # Average across batch for global memory update
        eta_scalar = eta.mean()
        theta_scalar = theta.mean()
        alpha_scalar = alpha.mean()

        # ============================================================
        # Step 4: Prepend Persistent Memory
        # Paper: "appended to the start of the sequence"
        # Persistent memory stores task-specific knowledge
        # ============================================================
        # Expand persistent memory to batch size
        persistent_k = self.persistent_keys.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N_p, D_mem)
        persistent_v = self.persistent_values.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N_p, D_mem)

        # Normalize persistent keys
        persistent_k = self._l2_normalize(persistent_k, dim=-1)

        # Concatenate persistent with sequence
        k_with_persistent = torch.cat([persistent_k, k], dim=1)  # (B, N_p + L, D_mem)
        v_with_persistent = torch.cat([persistent_v, v], dim=1)  # (B, N_p + L, D_mem)

        # ============================================================
        # Step 5: Update Memory State (Surprise-based)
        # Paper Eq: St = ηt·St-1 - θt·∇ℓ(Mt-1; xt)
        #           Mt = (1 - αt)·Mt-1 + St
        # ============================================================
        with torch.no_grad():
            k_f = k.float()  # Original k (not persistent) for memory update
            v_f = v.float()
            mem_f = self.memory_state.float()

            # Memory prediction through the MLP (for gradient computation)
            # Paper: ℓ(M; x) = ||M(k) - v||²
            memory_pred = torch.matmul(k_f, mem_f)  # (B, L, D_mem)

            # Momentary surprise: gradient of loss w.r.t. memory
            # ∇ℓ = 2 * k^T @ (M(k) - v) ≈ k^T @ error
            error = v_f - memory_pred  # (B, L, D_mem)
            momentary_surprise = torch.einsum('bld,blh->dh', k_f, error) / (batch_size * seq_len)

            # Update surprise with momentum: St = η·St-1 + θ·∇ℓ
            prev_surprise_f = self.surprise_state.float()
            new_surprise_f = eta_scalar.float() * prev_surprise_f + theta_scalar.float() * momentary_surprise
            self.surprise_state.copy_(new_surprise_f.to(self.surprise_state.dtype))

            # Update memory with forgetting: Mt = (1 - α)·Mt-1 + St
            new_memory_f = (1 - alpha_scalar.float()) * mem_f + new_surprise_f
            self.memory_state.copy_(new_memory_f.to(self.memory_state.dtype))
            self._memory_snapshot.copy_(self.memory_state)

        # ============================================================
        # Step 6: Memory Retrieval
        # Paper: y_t = M*(q_t) - inference without weight updates
        # Use both linear memory and deep MLP memory
        # ============================================================
        # Linear memory retrieval
        linear_retrieved = torch.matmul(q, self._memory_snapshot)  # (B, L, D_mem)

        # Deep MLP memory retrieval (the MLP acts as a learned memory function)
        mlp_retrieved = self.memory_mlp(q)  # (B, L, D_mem)

        # Combine linear and MLP memory
        retrieved = linear_retrieved + mlp_retrieved

        # Also attend to persistent + sequence key-values (soft attention)
        # This provides explicit access to stored information
        attn_scores = torch.matmul(q, k_with_persistent.transpose(-2, -1))  # (B, L, N_p+L)
        attn_scores = attn_scores / (self.memory_size ** 0.5)

        # Causal mask for sequence part (allow full access to persistent)
        causal_mask = torch.ones(seq_len, self.num_persistent + seq_len, device=x.device)
        causal_mask[:, self.num_persistent:] = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0) == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(x.dtype)
        context = torch.matmul(attn_weights, v_with_persistent)  # (B, L, D_mem)

        # Combine all retrievals
        retrieved = retrieved + context

        # ============================================================
        # Step 7: Output Gating and Projection
        # Paper: "gating with a linear layer before the final output projection"
        # ============================================================
        gate = torch.sigmoid(self.output_gate(retrieved))
        gated_output = gate * retrieved

        output = x + self.out_proj(gated_output)
        output = self.layer_norm(output)

        return output

    def reset_memory(self):
        """Reset memory state (call at start of new sequence)."""
        self.memory_state.zero_()
        self.surprise_state.zero_()
        self._memory_snapshot.zero_()


class OuroExitGate(nn.Module):
    """
    Ouro Exit Gate for Learned Depth Allocation.
    Computes P(exit) at each step.
    """
    def __init__(self, config: RMS_E_Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden state (B, L, D) or (B, D) depending on pooling
        Returns:
            p_exit: (B, L) Probability of exiting
        """
        return torch.sigmoid(self.gate_proj(h)).squeeze(-1)


# NOTE: SBTGate removed.
# Reference: arXiv:2505.14604 - 'Let LRMs Break Free from Overthinking via Self-Braking Tuning'
#
# The paper describes SBT as a TRAINING methodology, not an inference-time module:
# 1. Training data constructed by truncating reasoning at optimal points
# 2. Models learn to produce natural language "braking prompts" (e.g., "I've verified my answer")
# 3. Uses Reasoning Efficiency Ratio and Overthinking Marker Ratio metrics
#
# The previous implementation using cosine similarity and delta norm was NOT from the paper.
# Early exit during inference is handled by OuroExitGate (learned exit probability).


# Imports for LoopLMBlock
from olmo_core.nn.transformer.quad_hybrid import (
    GatedDeltaNetConfig, GatedDeltaNetBlock,
    GatedAttentionConfig, GatedAttentionBlock,
)


class LoopLMBlock(nn.Module):
    def __init__(self, config: RMS_E_Config):
        super().__init__()
        self.config = config
        
        # 1. Gated DeltaNet (O(N) - Majority)
        # We need to construct the config from RMS_E_Config
        # Note: GatedDeltaNet might handle head_dim implicitly or explicit?
        # Assuming we can pass kwargs or it has the field.
        dn_config = GatedDeltaNetConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads, # CORRECTED MAPPING
            num_v_heads=config.num_key_value_heads,
            head_dim=config.head_dim, 
            expand_v=1.0, # RMS-E/Qwen Hybrid: Match V dim to Head dim (4096 vs 8192)
            # ... other params ...
        )
             
        self.deltanet_block = GatedDeltaNetBlock(dn_config, layer_idx=0, n_layers=config.loop_max_steps, layer_norm=LayerNormConfig(name="rms"))
        
        # 2. Gated Attention (O(N^2) - Minority)
        ga_config = GatedAttentionConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,  # GQA support
            rope_percentage=config.rope_percentage,
            rope_theta=config.rope_theta,
            norm_eps=config.norm_eps,
            # ...
        )
        if config.head_dim is not None:
             ga_config.head_dim = config.head_dim
        
        self.attn_block = GatedAttentionBlock(ga_config, layer_idx=0, n_layers=config.loop_max_steps, layer_norm=LayerNormConfig(name="rms"))
        
        # MoE Injection (If blocks don't natively support MoE via simple init, we might monkey-patch or use specific MoE blocks)
        # For this refactor, let's assume valid MoE injection or simple MLP for now, 
        # OR better: use the `MoELayer` I already verified and SWAP the mlp of the blocks!
        
        # Override MLPs with Shared Loop MoE
        # [DeepSeekMoE paper] Shared experts are always activated to capture common knowledge
        self.moe = MoELayer(
            MoEConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_routed_experts,
                num_shared_experts=config.num_shared_experts,  # DeepSeekMoE: always activated
                num_active_experts=config.num_active_experts,
                router_aux_loss_coef=config.router_aux_loss_coef,
                router_z_loss_coef=config.router_z_loss_coef,
                use_loss_free_balancing=True,  # [DeepSeek] Use bias-based balancing, no aux_loss
            )
        )
        # Binding MoE to blocks (Pythonic Dynamic Patching for shared weights)
        self.deltanet_block.mlp = self.moe
        self.attn_block.mlp = self.moe
        
        # Step Embedding
        self.step_embed = nn.Embedding(config.loop_max_steps + 1, config.hidden_size)

    def forward(self, h: torch.Tensor, step_k: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Step Embedding
        # Handle both int (inference) and tensor (gradient checkpointing)
        if isinstance(step_k, torch.Tensor):
            step_idx = torch.clamp(step_k, 0, self.step_embed.num_embeddings - 1)
        else:
            step_idx = torch.clamp(torch.tensor(step_k, device=h.device), 0, self.step_embed.num_embeddings - 1)

        # [FIX] Ensure proper shape for broadcasting
        # step_embed(step_idx) returns (D,) for scalar input, need to broadcast to (B, L, D)
        step_emb = self.step_embed(step_idx)  # (D,) or (1, D) depending on input shape
        if step_emb.dim() == 1:
            step_emb = step_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, D) for proper broadcast
        h = h + step_emb

        # Hybrid Switching (Qwen3-Next style)
        # Reference: Qwen3-Next uses 3:1 ratio by layer (75% DeltaNet, 25% Attention)
        # Pattern: D, D, D, A, D, D, D, A, ...
        # Steps 0,1,2 -> DeltaNet, Step 3 -> Attention, Steps 4,5,6 -> DeltaNet, etc.
        # Formula: use Attention when (step + 1) % ratio == 0
        step_k_int = step_k.item() if isinstance(step_k, torch.Tensor) else step_k
        use_attention = self.config.use_hybrid_loop and ((step_k_int + 1) % self.config.hybrid_attention_ratio == 0)

        if use_attention:
             # Use Attention (every 4th step: 3, 7, 11, 15, ...)
             h, aux_loss = self.attn_block(h)
        else:
             # Use DeltaNet (majority: 0,1,2, 4,5,6, 8,9,10, ...)
             h, aux_loss = self.deltanet_block(h)
             
        return h, aux_loss


class RMS_E_Model(nn.Module):
    """
    RMS-E Main Model.
    Integrates Ouro Loop, Titans, and Atlas.
    """
    def __init__(self, config: RMS_E_Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Atlas
        if config.use_atlas and config.atlas_config is not None:
            # Use build() method which handles proper argument unpacking
            self.atlas = config.atlas_config.build(layer_idx=0)
        else:
            self.atlas = None
            
        # Titans
        if config.use_titans:
            self.titans = TitansMemory(config)
        else:
            self.titans = None
            
        # Recursive Core
        self.loop_block = LoopLMBlock(config)
        
        # Exit Gate (Ouro-style learned exit probability)
        self.ouro_gate = OuroExitGate(config)
            
        # Output Head
        self.norm_final = RMSNorm(size=config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP (Multi-Token Prediction) Head
        if config.use_mtp:
            mtp_config = MTPConfig(
                num_predict_tokens=config.mtp_num_predict_tokens,
                mtp_head_type=config.mtp_head_type,
                mtp_loss_weight=config.mtp_loss_weight,
            )
            self.mtp_head = MTPHead(
                d_model=config.hidden_size,
                vocab_size=config.vocab_size,
                config=mtp_config,
            )
            self.mtp_loss_fn = MTPLoss(mtp_config)
        else:
            self.mtp_head = None
            self.mtp_loss_fn = None

    def reset_memory(self):
        """Reset stateful memory modules (Titans / ATLAS)."""
        if self.atlas is not None and hasattr(self.atlas, "reset_memory"):
            self.atlas.reset_memory()
        if self.titans is not None and hasattr(self.titans, "reset_memory"):
            self.titans.reset_memory()

    def apply_fsdp(self, dp_mesh=None, **fsdp_kwargs):
        """
        [OPTIMIZED] Apply FSDP2 sharding for multi-GPU training (SGLang-style).

        SGLang optimizations:
        - Per-module sharding for better memory efficiency
        - Mixed precision with param_dtype for reduced memory
        - Selective sharding of large modules (MoE, embeddings)

        Args:
            dp_mesh: DeviceMesh for FSDP
            **fsdp_kwargs: Additional FSDP arguments
        """
        if not FSDP2_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning("FSDP2 not available. Skipping sharding.")
            return

        # [OPTIMIZED] Configure mixed precision policy based on config
        mp_policy = None
        if self.config.fsdp_mixed_precision == "bf16":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        elif self.config.fsdp_mixed_precision == "fp16":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
            )

        # [OPTIMIZED] SGLang-style: Shard large modules individually
        # 1. Shard embedding layer (large memory footprint)
        if hasattr(self, 'embed_tokens'):
            fully_shard(
                self.embed_tokens,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                **fsdp_kwargs
            )

        # 2. Shard loop block (contains MoE with 512 experts)
        if hasattr(self, 'loop_block') and hasattr(self.loop_block, 'moe'):
            fully_shard(
                self.loop_block.moe,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                **fsdp_kwargs
            )

        # 3. Shard LM head
        if hasattr(self, 'lm_head'):
            fully_shard(
                self.lm_head,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                **fsdp_kwargs
            )

        # 4. Shard MTP head if present
        if self.mtp_head is not None:
            fully_shard(
                self.mtp_head,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                **fsdp_kwargs
            )

        # 5. Finally shard the whole model
        fully_shard(
            self,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            **fsdp_kwargs
        )

    def apply_fp8(self, skip_first_last=True):
        """
        [OPTIMIZED] Apply FP8 Training with TransformerEngine.

        GB10/Blackwell: Native FP8 support for 1.5-2x training speedup.
        Converts nn.Linear layers to TransformerEngine's FP8-enabled layers.

        [STABILITY] Best Practice from NVIDIA/iGenius:
        - Keep first layer (embedding) in BF16 for numerical stability
        - Keep last layer (lm_head) in BF16 for output precision
        - Only convert internal layers to FP8

        Reference: https://github.com/NVIDIA/TransformerEngine
        Reference: https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/
        """
        if not TE_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning(
                "TransformerEngine not available. Install with: pip install transformer-engine"
            )
            return

        import logging
        logger = logging.getLogger(__name__)

        converted_count = 0

        def replace_linear_with_te(module, name=""):
            """Recursively replace nn.Linear with te.Linear"""
            nonlocal converted_count
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child, nn.Linear):
                    # Create TE Linear with same parameters
                    te_linear = te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                    )
                    # Copy weights
                    te_linear.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        te_linear.bias.data.copy_(child.bias.data)

                    # Replace module
                    setattr(module, child_name, te_linear)
                    converted_count += 1
                    logger.debug(f"Converted {full_name} to TE Linear (FP8)")
                else:
                    # Recurse into child modules
                    replace_linear_with_te(child, full_name)

        # [STABILITY] Skip first/last layers for numerical stability
        if skip_first_last:
            logger.info("FP8 Stability Mode: Keeping embed_tokens and lm_head in BF16")
            # 1. Embedding layer - KEEP IN BF16 (first layer)
            # embed_tokens is nn.Embedding, not nn.Linear, so it stays in original precision

            # 2. LM Head - KEEP IN BF16 (last layer)
            # Do NOT convert lm_head to FP8 for output precision

        else:
            # Convert LM Head to FP8 (not recommended for stability)
            if hasattr(self, 'lm_head') and isinstance(self.lm_head, nn.Linear):
                self.lm_head = te.Linear(
                    self.lm_head.in_features,
                    self.lm_head.out_features,
                    bias=self.lm_head.bias is not None,
                )
                self.lm_head.weight.data.copy_(self.embed_tokens.weight.data.T if hasattr(self, 'embed_tokens') else torch.zeros_like(self.lm_head.weight))
                converted_count += 1

        # 3. Loop Block (contains attention and MoE) - CONVERT TO FP8
        # This is where most computation happens
        if hasattr(self, 'loop_block'):
            replace_linear_with_te(self.loop_block)

        # 4. MTP Head - CONVERT TO FP8 (auxiliary head)
        if self.mtp_head is not None:
            replace_linear_with_te(self.mtp_head)

        # 5. Exit Gate - CONVERT TO FP8 (small layer)
        if hasattr(self, 'ouro_gate'):
            replace_linear_with_te(self.ouro_gate)

        logger.info(f"FP8 Training enabled: {converted_count} layers converted")
        if skip_first_last:
            logger.info("  - embed_tokens: BF16 (stability)")
            logger.info("  - lm_head: BF16 (stability)")
            logger.info("  - loop_block: FP8 (speedup)")
            logger.info("  - mtp_head: FP8 (speedup)")

    def get_fp8_recipe(self):
        """Get FP8 recipe for training context manager."""
        if not TE_AVAILABLE:
            return None

        import logging
        logger = logging.getLogger(__name__)

        # [OPTIMIZED] Block-wise FP8 scaling for Blackwell GPUs
        if self.config.use_mxfp8:
            try:
                from transformer_engine.pytorch import is_fp8_block_scaling_available

                if is_fp8_block_scaling_available():
                    # Try MXFP8BlockScaling first (best for Blackwell)
                    try:
                        from transformer_engine.common.recipe import MXFP8BlockScaling
                        recipe = MXFP8BlockScaling(
                            margin=self.config.mxfp8_margin,
                            fp8_format=Format.E4M3,
                        )
                        logger.info("Using MXFP8BlockScaling recipe (Blackwell optimized)")
                        logger.info("  - Block size: 32 elements")
                        logger.info("  - Format: E4M3 everywhere")
                        logger.info("  - Scaling: Hardware-accelerated E8M0")
                        return recipe
                    except (ImportError, AssertionError) as e:
                        # MXFP8 not supported, try Float8BlockScaling
                        logger.warning(f"MXFP8 not available: {e}")
                        logger.info("Falling back to Float8BlockScaling...")
                        from transformer_engine.common.recipe import Float8BlockScaling
                        recipe = Float8BlockScaling(fp8_format=Format.E4M3)
                        logger.info("Using Float8BlockScaling recipe")
                        logger.info("  - Format: E4M3")
                        logger.info("  - Block-wise scaling enabled")
                        return recipe
                else:
                    logger.warning("FP8 block scaling not available. Falling back to DelayedScaling.")
            except ImportError as e:
                logger.warning(f"Block scaling not found: {e}. Falling back to DelayedScaling.")

        # Standard FP8 with DelayedScaling (Hopper/Ada compatible)
        if self.config.fp8_format == "e4m3":
            fp8_format = Format.E4M3
        elif self.config.fp8_format == "e5m2":
            fp8_format = Format.E5M2
        else:  # hybrid
            fp8_format = Format.HYBRID

        return DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=self.config.fp8_amax_history_len,
            amax_compute_algo=self.config.fp8_amax_compute_algo,
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, training: bool = None) -> Union[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Forward pass.
        If training=True, returns lists of (logits, exit_probs) for all steps.
        If training=False, returns final logits (inference mode with early exit).
        """
        training = training if training is not None else self.training
        
        x = self.embed_tokens(input_ids)
        
        # Pre-Loop: Memory & Retrieval
        # NOTE: ATLAS/Titans modules already apply their own residual connections.
        # So we should *replace* x with their outputs, not add again (avoids double residual).
        if self.atlas:
            x, _ = self.atlas(x)
            
        if self.titans:
            x = self.titans(x)
            
        h_t = x
        
        all_hidden = []  # [OPTIMIZED] Store hidden states, compute logits lazily
        all_exit_probs = []
        
        # Inference State
        final_logits = None
        
        # Loop
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        for k in range(self.config.loop_max_steps):
            h_prev = h_t
            # Gradient Checkpointing for Memory Efficiency (Recursive Unroll)
            # WARNING: MoE dynamic routing is INCOMPATIBLE with checkpointing.
            # Recomputation routes different token counts → shape mismatch.
            # Only enable if not using MoE (num_routed_experts == 0).
            use_ckpt = (
                self.training and
                self.config.use_gradient_checkpointing and
                self.config.num_routed_experts == 0  # Safety: disable for MoE
            )

            if use_ckpt:
                 # [FIX] Checkpoint requires tensors for autograd. Wrap k.
                 k_tensor = torch.tensor(k, device=h_t.device, requires_grad=False)
                 h_t, aux_loss = torch.utils.checkpoint.checkpoint(self.loop_block, h_t, k_tensor, use_reentrant=False)
            else:
                 h_t, aux_loss = self.loop_block(h_t, step_k=k)
            
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
            
            # Compute Exit Prob
            p_exit = self.ouro_gate(h_t) # (B, L)
            
            if training:
                # [OPTIMIZED] Store hidden states instead of logits
                # Compute logits lazily at the end to reduce VRAM
                # Reference: Fused Projection-Prediction paper - avoid materializing full logits
                all_hidden.append(h_t)
                all_exit_probs.append(p_exit)
            else:
                # Inference: Early exit based on Ouro exit gate
                # Reference: arXiv:2510.25741 - learned exit probability
                should_stop = p_exit > 0.5

                # If all sequences/tokens in batch want to stop, break early
                if should_stop.all():
                    final_logits = self.lm_head(self.norm_final(h_t))
                    break
                    
        if training:
            # [FIX] Guard against empty all_hidden (edge case: loop_max_steps=0)
            if len(all_hidden) == 0:
                # Fallback: use input embeddings directly
                all_hidden.append(h_t)
                all_exit_probs.append(torch.zeros(h_t.shape[0], h_t.shape[1], device=h_t.device))

            # [OPTIMIZED] Compute logits lazily from stored hidden states
            # This reduces peak VRAM by not materializing all logits simultaneously
            all_logits = [self.lm_head(self.norm_final(h)) for h in all_hidden]

            # MTP Loss (if enabled and labels provided)
            mtp_loss = torch.tensor(0.0, device=input_ids.device)
            if self.mtp_head is not None and labels is not None:
                # Use the final hidden state for MTP prediction
                final_hidden = all_hidden[-1]
                mtp_logits = self.mtp_head(final_hidden, input_ids, self.embed_tokens)
                mtp_loss, _ = self.mtp_loss_fn(mtp_logits, labels)

            return all_logits, all_exit_probs, total_aux_loss, mtp_loss
        else:
            if final_logits is None:
                final_logits = self.lm_head(self.norm_final(h_t))
            return final_logits


def rms_e_loss(
    logits_per_step: List[torch.Tensor],
    exit_probs_per_step: List[torch.Tensor],
    labels: torch.Tensor,
    beta: float = 0.01
) -> torch.Tensor:
    """
    Ouro/LoopLM Entropy-Regularized Loss (arXiv:2510.25741)

    L = Σ(t=1 to T) q(t) · CE(y, ŷ_t) - β · H(q)

    where:
    - q(t) = λ(t) · Π_{j<t}(1 - λ(j))  [exit probability distribution]
    - H(q) = -Σ q(t) · log(q(t))        [entropy over depth distribution]
    - β encourages exploration of different computation depths

    Reference: https://arxiv.org/abs/2510.25741
    """
    batch_size, seq_len = labels.shape
    device = labels.device
    T = len(exit_probs_per_step)

    # Compute q(t) distribution
    # q(t) = p_exit(t) * prod_{j<t}(1 - p_exit(j))
    q_probs = []
    accum_continue_prob = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)

    for i, p_exit in enumerate(exit_probs_per_step):
        p_exit_f = p_exit.float()

        if i == T - 1:
            # Last step: take all remaining probability (ensures q sums to 1)
            q_t = accum_continue_prob
        else:
            # q(t) = p_exit(t) * accum_continue
            q_t = accum_continue_prob * p_exit_f

        q_probs.append(q_t)

        # Update: accum_continue *= (1 - p_exit)
        accum_continue_prob = accum_continue_prob * (1.0 - p_exit_f)

    # Weighted Cross-Entropy Loss: Σ q(t) · CE(y, ŷ_t)
    total_ce_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    for logits, q_t in zip(logits_per_step, q_probs):
        # Cross-entropy per token
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1),
            reduction='none'
        )
        ce = ce.view(batch_size, seq_len)

        # Weighted by exit probability q(t)
        # Note: q_t should NOT be detached - gradients flow through both CE and q_t
        weighted_ce = (q_t * ce).mean()
        total_ce_loss = total_ce_loss + weighted_ce

    # Entropy Regularization: H(q) = -Σ q(t) · log(q(t))
    # Encourages exploration of different computation depths
    q_stacked = torch.stack(q_probs, dim=0)  # (T, B, L)
    q_safe = q_stacked.clamp(min=1e-8)  # Avoid log(0)

    # H(q) per position, then average
    entropy = -(q_safe * torch.log(q_safe)).sum(dim=0).mean()

    # Final loss: L = CE_weighted - β · H(q)
    # Subtracting entropy encourages higher entropy (more uniform depth distribution)
    final_loss = total_ce_loss - beta * entropy

    # NaN safety (should not happen with proper clamping)
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        return total_ce_loss

    return final_loss
