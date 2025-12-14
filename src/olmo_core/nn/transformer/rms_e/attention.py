"""
Attention Modules - Official Implementations

This module contains:
- GatedAttention: NeurIPS 2025 Oral (arXiv 2505.06708)
- MLA: Multi-head Latent Attention from DeepSeek-V3

References:
- Gated Attention: https://github.com/qiuzh20/gated_attention
- DeepSeek-V3 MLA: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import RMSNorm
from .rope import (
    YaRNRotaryEmbedding,
    apply_rotary_pos_emb,
    yarn_get_mscale,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GatedAttention(nn.Module):
    """
    Gated Attention - Official Implementation
    Reference: https://github.com/qiuzh20/gated_attention (NeurIPS 2025 Oral)
    Paper: arXiv 2505.06708

    Key mechanism: Sigmoid gate AFTER Scaled Dot-Product Attention (SDPA)
    Two variants:
    - Headwise gating: scalar gate per head (efficient)
    - Elementwise gating: gate per element (more expressive)

    Now with YaRN RoPE support (DeepSeek-V3 / ICLR 2024)
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        headwise_gate: bool = True,
        elementwise_gate: bool = False,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
        # YaRN parameters (DeepSeek-V3 / ICLR 2024)
        use_yarn: bool = False,
        original_max_position_embeddings: int = 4096,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        yarn_mscale: float = 1.0,
        yarn_rope_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.headwise_gate = headwise_gate
        self.elementwise_gate = elementwise_gate
        self.use_qk_norm = use_qk_norm
        self.use_yarn = use_yarn

        # Q projection with gate (official gated attention pattern)
        if self.headwise_gate:
            # Headwise: add num_heads scalar gates
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim + self.num_heads,
                bias=False
            )
        elif self.elementwise_gate:
            # Elementwise: double the output for gates
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim * 2,
                bias=False
            )
        else:
            # No gating
            self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)

        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # QK normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # RoPE with optional YaRN
        self._init_rope(
            rope_theta, max_position_embeddings,
            use_yarn, original_max_position_embeddings,
            yarn_beta_fast, yarn_beta_slow, yarn_mscale, yarn_rope_factor
        )

    def _init_rope(
        self,
        rope_theta: float,
        max_position_embeddings: int,
        use_yarn: bool = False,
        original_max_position_embeddings: int = 4096,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        yarn_mscale: float = 1.0,
        yarn_rope_factor: float = 1.0,
    ):
        """Initialize rotary position embedding with optional YaRN."""
        self.max_position_embeddings = max_position_embeddings

        if use_yarn:
            # Use YaRN for extended context (DeepSeek-V3 / ICLR 2024)
            self.rotary_emb = YaRNRotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                original_max_position_embeddings=original_max_position_embeddings,
                beta_fast=yarn_beta_fast,
                beta_slow=yarn_beta_slow,
                mscale=yarn_mscale,
                rope_factor=yarn_rope_factor,
            )
            # Store attention scale for YaRN
            self.yarn_attention_scale = self.rotary_emb.attention_scale
        else:
            # Standard RoPE
            inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            ))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.rotary_emb = None
            self.yarn_attention_scale = 1.0

    def _compute_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Compute rotary embeddings (with YaRN support).

        Returns:
            cos, sin: shape [1, seq_len, head_dim] for use with apply_rotary_pos_emb
        """
        if self.rotary_emb is not None:
            # Use YaRN - returns [seq_len, head_dim//2]
            dummy_x = torch.empty(1, seq_len, device=device, dtype=dtype)
            cos, sin = self.rotary_emb(dummy_x, seq_len=seq_len)
            # Duplicate to full head_dim and add batch dim: [seq_len, head_dim//2] -> [1, seq_len, head_dim]
            cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)
            sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)
            return cos, sin
        else:
            # Standard RoPE - returns [1, seq_len, head_dim]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with official gated attention mechanism.
        Gate is applied AFTER SDPA with sigmoid activation.
        """
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Extract gate from Q projection (official pattern)
        gate_score = None
        if self.headwise_gate:
            query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim * self.num_key_value_groups, self.num_key_value_groups],
                dim=-1
            )
            gate_score = gate_score.reshape(bsz, q_len, -1, 1)
            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        elif self.elementwise_gate:
            query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim * self.num_key_value_groups, self.head_dim * self.num_key_value_groups],
                dim=-1
            )
            gate_score = gate_score.reshape(bsz, q_len, -1, self.head_dim)
            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # QK normalization
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = self._compute_rope(q_len, hidden_states.device, hidden_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True if attention_mask is None else False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Apply sigmoid gate AFTER SDPA (official gated attention pattern)
        if self.headwise_gate or self.elementwise_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) - DeepSeek-V3 Official Implementation

    Reference: DeepSeek-V3 model.py line 396-497

    Key innovations:
    1. Low-rank KV compression: Reduces KV cache size significantly
    2. Decoupled RoPE: Separate dimensions for positional and non-positional info
    3. Absorbed attention: Efficient computation by absorbing projections

    Architecture:
    - Q: x -> wq_a -> q_norm -> wq_b -> [q_nope, q_pe]
    - KV: x -> wkv_a -> [kv (compressed), k_pe]
    - kv -> kv_norm -> wkv_b -> [k_nope, v]
    - RoPE applied only to q_pe and k_pe
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        # MLA specific parameters (from DeepSeek-V3)
        q_lora_rank: int = 0,  # 0 means no low-rank for Q
        kv_lora_rank: int = 512,  # Low-rank dimension for KV compression
        qk_nope_head_dim: int = 128,  # Non-positional Q/K dimension
        qk_rope_head_dim: int = 64,  # Positional Q/K dimension (with RoPE)
        v_head_dim: int = 128,  # Value head dimension
        # RoPE parameters
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        # YaRN parameters
        use_yarn: bool = False,
        original_max_position_embeddings: int = 4096,
        yarn_mscale: float = 1.0,
        yarn_rope_factor: float = 1.0,
        # Gating (Gated-MLA)
        use_gate: bool = False,
        headwise_gate: bool = True,
        # Other
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attention_dropout = attention_dropout
        self.use_gate = use_gate
        self.headwise_gate = headwise_gate

        # Attention scaling (with YaRN support)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if use_yarn and max_position_embeddings > original_max_position_embeddings:
            mscale = yarn_get_mscale(yarn_rope_factor, yarn_mscale)
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # Q projection (with optional low-rank)
        if self.q_lora_rank == 0:
            # Direct projection
            self.wq = nn.Linear(hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            # Low-rank factorization: x -> wq_a -> q_norm -> wq_b
            self.wq_a = nn.Linear(hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=rms_norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV projection (always low-rank compressed)
        # Output: [kv_lora_rank (compressed KV), qk_rope_head_dim (position)]
        self.wkv_a = nn.Linear(hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=rms_norm_eps)
        # Decompress: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )

        # Output projection
        self.wo = nn.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=False)

        # Gating projection (Gated-MLA)
        if self.use_gate:
            gate_dim = self.num_heads if self.headwise_gate else self.num_heads * self.v_head_dim
            self.gate_proj = nn.Linear(hidden_size, gate_dim, bias=False)

        # RoPE for positional dimensions only
        self._init_rope(rope_theta, max_position_embeddings)

    def _init_rope(self, rope_theta: float, max_position_embeddings: int):
        """Initialize RoPE for positional dimensions."""
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32) / self.qk_rope_head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def _compute_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute RoPE cos/sin for positional dimensions."""
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to positional dimensions.

        Args:
            x: Input tensor, can be:
               - [bsz, seq_len, num_heads, head_dim] for q_pe
               - [bsz, seq_len, 1, head_dim] for k_pe
            cos, sin: [1, seq_len, head_dim] from _compute_rope
        """
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        # cos/sin: [1, seq_len, head_dim] -> [1, seq_len, 1, head_dim] for broadcasting
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return (x * cos) + (rotate_half(x) * sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for MLA.

        Reference: DeepSeek-V3 model.py MLA.forward (line 446-497)
        """
        bsz, seq_len, _ = hidden_states.size()

        # Q projection
        if self.q_lora_rank == 0:
            q = self.wq(hidden_states)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))

        # Reshape Q and split into nope (no position) and pe (position) parts
        q = q.view(bsz, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection (compressed)
        kv = self.wkv_a(hidden_states)
        # Split into compressed KV and positional key
        kv_compressed, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to positional parts
        cos, sin = self._compute_rope(seq_len, hidden_states.device, hidden_states.dtype)
        q_pe = self._apply_rope(q_pe, cos, sin)
        k_pe = self._apply_rope(k_pe.unsqueeze(2), cos, sin)  # Add head dim

        # Decompress KV
        kv_decompressed = self.wkv_b(self.kv_norm(kv_compressed))
        kv_decompressed = kv_decompressed.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_decompressed, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Combine q and k parts
        q = torch.cat([q_nope, q_pe], dim=-1)  # [bsz, seq_len, num_heads, qk_head_dim]
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)

        # Transpose for attention: [bsz, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        if q.device.type == "cuda" and attention_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True if attention_mask is None else False,
            scale=self.softmax_scale,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Apply Gating (Gated-MLA)
        if self.use_gate:
            gate_score = self.gate_proj(hidden_states) # [bsz, seq_len, gate_dim]
            if self.headwise_gate:
                # gate_score: [bsz, seq_len, num_heads] -> broadcast to [bsz, seq_len, num_heads, v_head_dim]
                gate_score = gate_score.unsqueeze(-1)
                # attn_output: [bsz, seq_len, num_heads, v_head_dim] (before view)
                attn_output = attn_output * torch.sigmoid(gate_score)
            else:
                # Elementwise: [bsz, seq_len, num_heads * v_head_dim]
                attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.v_head_dim)
                attn_output = attn_output * torch.sigmoid(gate_score)
                # Already flattened, skip view below if needed, but safe to proceed
        
        attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.v_head_dim)
        return self.wo(attn_output)


# Check for FlashAttention with sliding window support
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class GatedSWA(nn.Module):
    """
    Gated Sliding Window Attention (Gated-SWA)

    Combines:
    - Sliding Window Attention (Mistral/Longformer style)
    - Gated Attention (NeurIPS 2025, arXiv 2505.06708)

    Architecture:
    - Local attention within sliding window
    - Gating applied after attention (official pattern)
    - Uses FlashAttention window_size when available

    Reference:
    - Mamba2-NSA: https://github.com/ant-research/long-context-modeling
    - Gated Attention: https://github.com/qiuzh20/gated_attention
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: Optional[int] = None,
        sliding_window: int = 512,  # Window size for local attention
        attention_dropout: float = 0.0,
        headwise_gate: bool = True,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
        # YaRN parameters
        use_yarn: bool = False,
        original_max_position_embeddings: int = 4096,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        yarn_mscale: float = 1.0,
        yarn_rope_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.headwise_gate = headwise_gate
        self.use_qk_norm = use_qk_norm
        self.use_yarn = use_yarn

        # Q projection with gate (headwise gating)
        if self.headwise_gate:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.head_dim + self.num_heads,
                bias=False
            )
        else:
            self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)

        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # QK normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # RoPE with optional YaRN
        self._init_rope(
            rope_theta, max_position_embeddings,
            use_yarn, original_max_position_embeddings,
            yarn_beta_fast, yarn_beta_slow, yarn_mscale, yarn_rope_factor
        )

    def _init_rope(
        self,
        rope_theta: float,
        max_position_embeddings: int,
        use_yarn: bool = False,
        original_max_position_embeddings: int = 4096,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        yarn_mscale: float = 1.0,
        yarn_rope_factor: float = 1.0,
    ):
        """Initialize rotary position embedding with optional YaRN."""
        self.max_position_embeddings = max_position_embeddings

        if use_yarn:
            self.rotary_emb = YaRNRotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                original_max_position_embeddings=original_max_position_embeddings,
                beta_fast=yarn_beta_fast,
                beta_slow=yarn_beta_slow,
                mscale=yarn_mscale,
                rope_factor=yarn_rope_factor,
            )
            self.yarn_attention_scale = self.rotary_emb.attention_scale
        else:
            inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            ))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.rotary_emb = None
            self.yarn_attention_scale = 1.0

    def _compute_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute rotary embeddings."""
        if self.rotary_emb is not None:
            dummy_x = torch.empty(1, seq_len, device=device, dtype=dtype)
            cos, sin = self.rotary_emb(dummy_x, seq_len=seq_len)
            cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)
            sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)
            return cos, sin
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype)
            sin = emb.sin().to(dtype)
            return cos, sin

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Create causal sliding window attention mask."""
        # Create causal mask first
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # Upper triangular = masked

        # Add sliding window constraint
        # Mask positions beyond window_size distance
        for i in range(seq_len):
            start = max(0, i - self.sliding_window + 1)
            mask[i, :start] = True  # Mask positions before window

        # Convert to attention mask format (-inf for masked positions)
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with sliding window attention and gating.
        Uses FlashAttention with window_size when available.
        """
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Extract gate from Q projection
        gate_score = None
        if self.headwise_gate:
            query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states,
                [self.head_dim * self.num_key_value_groups, self.num_key_value_groups],
                dim=-1
            )
            gate_score = gate_score.reshape(bsz, q_len, -1, 1)
            query_states = query_states.reshape(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # QK normalization
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = self._compute_rope(q_len, hidden_states.device, hidden_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Sliding Window Attention
        if FLASH_ATTN_AVAILABLE and hidden_states.device.type == "cuda":
            # Use FlashAttention with window_size
            # flash_attn expects [bsz, seq_len, num_heads, head_dim]
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window, 0),  # (left, right) - causal so right=0
            )
            # attn_output: [bsz, seq_len, num_heads, head_dim]
        else:
            # Fallback: create sliding window mask for SDPA
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

            sw_mask = self._create_sliding_window_mask(q_len, hidden_states.device, hidden_states.dtype)

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=sw_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,  # Using custom mask
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.contiguous()

        # Apply sigmoid gate AFTER attention (official gated attention pattern)
        if self.headwise_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


__all__ = [
    "GatedAttention",
    "GatedSWA",
    "MLA",
    "repeat_kv",
]
