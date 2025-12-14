"""
RMS-E (Recursive Memory-Sparse Experts) - Official Implementation Based

This file re-exports from the rms_e package for backwards compatibility.
For new code, import directly from olmo_core.nn.transformer.rms_e

This implementation uses ONLY official/verified code:
1. DeepSeekMoE: https://github.com/deepseek-ai/DeepSeek-MoE (Apache 2.0)
2. Ouro/LoopLM: https://huggingface.co/ByteDance/Ouro-2.6B (Apache 2.0)
3. GatedDeltaNet: FLA library - https://github.com/NVlabs/GatedDeltaNet (ICLR 2025)
4. Gated Attention: https://github.com/qiuzh20/gated_attention (NeurIPS 2025 Oral, arXiv 2505.06708)
5. YaRN RoPE: https://github.com/jquesnelle/yarn (ICLR 2024)
6. MLA: https://github.com/deepseek-ai/DeepSeek-V3 (DeepSeek-V2/V3)

NO custom implementations of core components.
"""

# Re-export everything from the rms_e package
from .rms_e import (
    # Config
    RMSEConfig,
    # Utils
    RMSNorm,
    AddAuxiliaryLoss,
    UniversalTransformerCache,
    # RoPE
    YaRNRotaryEmbedding,
    yarn_find_correction_dim,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
    yarn_get_mscale,
    rotate_half,
    apply_rotary_pos_emb,
    apply_yarn_rotary_pos_emb,
    # MoE
    MoEGate,
    DeepSeekMLP,
    DeepSeekMoE,
    # Attention
    GatedAttention,
    MLA,
    repeat_kv,
    # Layer
    RMSEDecoderLayer,
    # Model
    RMSEModel,
    RMSEForCausalLM,
    create_rmse_model,
)

__all__ = [
    "RMSEConfig",
    "RMSEModel",
    "RMSEForCausalLM",
    "DeepSeekMoE",
    "MoEGate",
    "GatedAttention",
    "MLA",
    "UniversalTransformerCache",
    "create_rmse_model",
    # Additional exports
    "RMSNorm",
    "AddAuxiliaryLoss",
    "DeepSeekMLP",
    "YaRNRotaryEmbedding",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_linear_ramp_mask",
    "yarn_get_mscale",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_yarn_rotary_pos_emb",
    "repeat_kv",
    "RMSEDecoderLayer",
]
