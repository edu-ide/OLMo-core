"""
RMS-E (Recursive Memory-Sparse Experts) - TRM-style Architecture

Components:
1. TRM: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
2. DeepSeekMoE: https://github.com/deepseek-ai/DeepSeek-MoE (Apache 2.0)
3. GatedDeltaNet: FLA library - https://github.com/NVlabs/GatedDeltaNet (ICLR 2025)
4. Gated Attention: https://github.com/qiuzh20/gated_attention (NeurIPS 2025 Oral)
5. YaRN RoPE: https://github.com/jquesnelle/yarn (ICLR 2024)
6. HSA-UltraLong: arxiv 2511.23319
"""

from .config import RMSEConfig
from .utils import RMSNorm, AddAuxiliaryLoss
from .rope import (
    YaRNRotaryEmbedding,
    yarn_find_correction_dim,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
    yarn_get_mscale,
    rotate_half,
    apply_rotary_pos_emb,
    apply_yarn_rotary_pos_emb,
)
from .moe import MoEGate, DeepSeekMLP, DeepSeekMoE
from .attention import GatedAttention, MLA, repeat_kv
from .layer import RMSEDecoderLayer
from .trm_model import TRMStyleRMSE, TRMStyleRMSEForCausalLM, TRMCarry


__all__ = [
    # Config
    "RMSEConfig",
    # Utils
    "RMSNorm",
    "AddAuxiliaryLoss",
    # RoPE
    "YaRNRotaryEmbedding",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_linear_ramp_mask",
    "yarn_get_mscale",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_yarn_rotary_pos_emb",
    # MoE
    "MoEGate",
    "DeepSeekMLP",
    "DeepSeekMoE",
    # Attention
    "GatedAttention",
    "MLA",
    "repeat_kv",
    # Layer
    "RMSEDecoderLayer",
    # TRM-style Model
    "TRMStyleRMSE",
    "TRMStyleRMSEForCausalLM",
    "TRMCarry",
]
