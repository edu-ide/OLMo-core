"""
RMS-E Configuration

RMS-E (Recursive Memory-Sparse Experts) configuration using official implementations only.
"""

from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config


@dataclass
class RMSEConfig(Config):
    """
    RMS-E Configuration using TRM-style architecture.

    Components:
    1. TRM: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
    2. DeepSeekMoE: https://github.com/deepseek-ai/DeepSeek-MoE (Apache 2.0)
    3. GatedDeltaNet: FLA library - https://github.com/NVlabs/GatedDeltaNet (ICLR 2025)
    4. Gated Attention: https://github.com/qiuzh20/gated_attention (NeurIPS 2025 Oral)
    5. YaRN RoPE: https://github.com/jquesnelle/yarn (ICLR 2024)
    6. HSA-UltraLong: arxiv 2511.23319
    """
    # Model dimensions
    hidden_size: int = 2048
    intermediate_size: int = 5632
    vocab_size: int = 151936  # Qwen3 vocab size

    # Attention (for hybrid mode)
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: Optional[int] = None

    # TRM-style nested loops
    # Effective iterations = H_cycles × L_cycles for z_L, H_cycles for z_H
    H_cycles: int = 3  # Outer loop: z_H updates (high-level reasoning)
    L_cycles: int = 6  # Inner loop: z_L updates per H_cycle (detailed reasoning)
    L_layers: int = 2  # Number of blocks in L_level module (TRM default: 2)

    # ACT (Adaptive Computation Time) - TRM style halting
    use_act: bool = True  # Enable ACT for early exit
    act_max_steps: int = 3  # Max H_cycles (same as H_cycles by default)
    act_exploration_prob: float = 0.1  # Random exploration during training

    # FFN type: Dense or MoE
    use_moe: bool = True  # True: MoE, False: Dense FFN

    # MoE (DeepSeek-style) - only used when use_moe=True
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    aux_loss_alpha: float = 0.001
    moe_intermediate_size: int = 1408  # Per-expert intermediate size

    # Loss-Free Balancing (DeepSeek-V3 official, arXiv:2408.15664)
    use_loss_free_balancing: bool = True  # Use dynamic bias instead of aux loss
    bias_update_rate: float = 0.001  # u in the paper (optimal for 1B models)
    route_scale: float = 1.0  # From DeepSeek-V3 official (scales gating weights)
    score_func: str = "softmax"  # "softmax" (default) or "sigmoid"

    # GatedDeltaNet (FLA)
    deltanet_num_heads: int = 4
    deltanet_head_dim: int = 64  # Reduced for GPU memory compatibility
    deltanet_expand_v: float = 1.0

    # Hybrid ratio (DeltaNet:Attention)
    hybrid_attention_ratio: int = 4  # Every Nth layer adds SWA + HSA

    # Sliding Window Attention (Gated-SWA)
    sliding_window: int = 512  # Window size for local attention
    use_swa: bool = True  # Enable Gated-SWA for local context

    # Gated Attention (NeurIPS 2025 official)
    headwise_gate: bool = True  # Use headwise gating (efficient)
    elementwise_gate: bool = False  # Use elementwise gating (more expressive)
    use_qk_norm: bool = True  # QK normalization

    # MLA (Multi-Head Latent Attention) Settings (DeepSeek-V3)
    use_mla: bool = False  # Enable MLA to compress KV cache
    q_lora_rank: int = 0  # 0 means no low-rank for Q (DeepSeek-V3 uses 1536 for huge models)
    kv_lora_rank: int = 512  # Low-rank dimension for KV compression
    qk_nope_head_dim: int = 128  # Non-positional Q/K dimension
    qk_rope_head_dim: int = 64  # Positional Q/K dimension (with RoPE)
    v_head_dim: int = 128  # Value head dimension

    # Gated-HSA (Hierarchical Sparse Attention + Gating)
    # HSA: arxiv 2511.23319 + Gating: arxiv 2505.06708
    # Train 4K → Generalize 16M+
    use_hsa: bool = True  # Enable Gated-HSA for long context (default: ON)
    hsa_chunk_size: int = 64  # Tokens per chunk
    hsa_chunk_topk: int = 8  # Top-k chunks to retrieve
    hsa_retrieval_dim: int = 256  # Landmark embedding dimension

    # Normalization
    rms_norm_eps: float = 1e-6

    # RoPE with YaRN support (DeepSeek-V3 / ICLR 2024)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    # YaRN parameters for extended context
    use_yarn: bool = True  # Enable YaRN scaling
    original_max_position_embeddings: int = 4096  # Original training length
    yarn_beta_fast: int = 32  # Fast beta for correction range
    yarn_beta_slow: int = 1  # Slow beta for correction range
    yarn_mscale: float = 1.0  # Magnitude scaling factor
    yarn_rope_factor: float = 8.0  # RoPE scaling factor (8x = 32K from 4K)

    # MTP (Multi-Token Prediction) - DeepSeek-V3
    use_mtp: bool = True
    num_mtp_tokens: int = 1
    mtp_loss_weight: float = 0.3
