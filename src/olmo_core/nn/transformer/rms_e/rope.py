"""
YaRN RoPE - Official Implementation

Reference:
- YaRN Paper: https://arxiv.org/abs/2309.00071 (ICLR 2024)
- YaRN Code: https://github.com/jquesnelle/yarn
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    """
    Computes the correction dimension for YaRN scaling.

    Reference: DeepSeek-V3 model.py line 314-327
    """
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
) -> Tuple[int, int]:
    """
    Computes the range of correction dimensions for YaRN scaling.

    Reference: DeepSeek-V3 model.py line 329-345
    """
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """
    Computes a linear ramp function for smooth interpolation.

    Reference: DeepSeek-V3 model.py line 347-364
    """
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    """
    Computes magnitude scaling for YaRN.

    Reference:
    - YaRN official: 0.1 * math.log(scale) + 1.0
    - DeepSeek-V3: 0.1 * mscale * math.log(rope_factor) + 1.0
    """
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) Rotary Position Embedding.

    Official References:
    - YaRN Paper: https://arxiv.org/abs/2309.00071 (ICLR 2024)
    - YaRN Code: https://github.com/jquesnelle/yarn
    - DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

    Key features:
    - Extends context window efficiently with minimal fine-tuning
    - Uses NTK-aware interpolation with attention scaling
    - Smooth transition between interpolation and extrapolation
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        # YaRN specific parameters
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        rope_factor: float = 1.0,  # Scaling factor for extended contexts
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.rope_factor = rope_factor

        # Compute attention scaling for extended contexts
        self.attention_scale = 1.0
        if max_position_embeddings > original_max_position_embeddings:
            scale = yarn_get_mscale(rope_factor, mscale)
            self.attention_scale = scale * scale

        # Precompute frequency tensor
        self._compute_freqs()

    def _compute_freqs(self):
        """
        Compute frequency tensor with YaRN scaling.

        Reference: DeepSeek-V3 model.py line 366-375
        """
        # Base frequencies
        freqs = 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
        ))

        # Apply YaRN scaling if context is extended
        if self.max_position_embeddings > self.original_max_position_embeddings:
            low, high = yarn_find_correction_range(
                self.beta_fast, self.beta_slow,
                self.dim, self.base,
                self.original_max_position_embeddings
            )
            # Smooth mask for interpolation vs extrapolation
            smooth = 1 - yarn_linear_ramp_mask(low, high, self.dim // 2)
            # Blend: extrapolated frequencies * (1-smooth) + original * smooth
            freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth

        # Precompute cos/sin for all positions
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs_outer = torch.outer(t, freqs)

        # Complex exponential representation (DeepSeek-V3 style)
        freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("cos_cached", freqs_outer.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_outer.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin for rotary embedding.

        Args:
            x: Input tensor (for dtype/device)
            position_ids: Optional position indices
            seq_len: Sequence length

        Returns:
            (cos, sin) tensors for rotary embedding
        """
        if seq_len is None:
            seq_len = x.shape[1] if x.dim() > 1 else x.shape[0]

        # Extend cache if needed
        if seq_len > self.max_position_embeddings:
            self._extend_cache(seq_len, x.device, x.dtype)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin

    def _extend_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Extend cache for longer sequences."""
        self.max_position_embeddings = seq_len
        self._compute_freqs()
        self.cos_cached = self.cos_cached.to(device)
        self.sin_cached = self.sin_cached.to(device)
        self.freqs_cis = self.freqs_cis.to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_yarn_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply YaRN rotary position embedding to query and key tensors.

    Reference: DeepSeek-V3 model.py apply_rotary_emb (line 378-393)

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, head_dim//2]
        sin: Sine values [seq_len, head_dim//2]
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Rotated (q, k) tensors
    """
    # Expand cos/sin for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)  # [seq_len, 1, head_dim//2]
    sin = sin.unsqueeze(unsqueeze_dim)  # [seq_len, 1, head_dim//2]

    # Duplicate for full head_dim
    cos = torch.cat([cos, cos], dim=-1)  # [seq_len, 1, head_dim]
    sin = torch.cat([sin, sin], dim=-1)  # [seq_len, 1, head_dim]

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


__all__ = [
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_linear_ramp_mask",
    "yarn_get_mscale",
    "YaRNRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_yarn_rotary_pos_emb",
]
