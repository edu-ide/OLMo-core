"""
Gated Attention: 게이트된 어텐션 메커니즘.

표준 어텐션에 게이팅을 추가하여:
- 선택적 정보 접근
- 노이즈 억제
- 핵심 정보에 집중

Reference:
- Qwen3-Next Gated Attention Implementation
- Partial RoPE (50%)
- Zero-Centered RMSNorm for Q/K
- Sigmoid Output Gate
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmo_core.config import Config
from olmo_core.nn.transformer.block import TransformerBlockBase
from olmo_core.nn.layer_norm import LayerNormConfig, RMSNorm, ZeroCenteredRMSNorm


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create position indices
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        # Apply rotary embeddings
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)


@dataclass
class GatedAttentionConfig(Config):
    """Gated Attention 설정"""
    hidden_size: int = 2048
    num_heads: int = 16
    num_key_value_heads: Optional[int] = None  # GQA support: if None, use num_heads (MHA)
    head_dim: int = 128
    use_rotary: bool = True
    rope_percentage: float = 0.5
    rope_theta: float = 10000.0
    use_gate: bool = True
    qkv_bias: bool = False # [VERIFIED] Qwen3 config says attention_bias=False.
    dropout: float = 0.0
    norm_eps: float = 1e-5
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None

    def build(self, layer_idx: Optional[int] = None) -> "GatedAttention":
        return GatedAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            use_rotary=self.use_rotary,
            rope_percentage=self.rope_percentage,
            rope_theta=self.rope_theta,
            use_gate=self.use_gate,
            qkv_bias=self.qkv_bias,
            dropout=self.dropout,
            layer_idx=layer_idx,
            norm_eps=self.norm_eps,
        )


class GatedAttention(nn.Module):
    """
    Gated Attention: 게이트된 어텐션 메커니즘.

    표준 어텐션에 게이팅을 추가하여:
    - 선택적 정보 접근
    - 노이즈 억제
    - 핵심 정보에 집중
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        num_key_value_heads: Optional[int] = None,  # GQA: if None, use num_heads (MHA)
        head_dim: int = 128,
        use_rotary: bool = True,
        use_gate: bool = True,
        rope_percentage: float = 0.5,
        rope_theta: float = 10000.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # GQA: Default to MHA if not specified
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads  # How many Q heads share one K/V head
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.kv_dim = num_key_value_heads * head_dim  # K/V dimension (smaller than Q for GQA)
        self.use_rotary = use_rotary
        self.rope_percentage = rope_percentage
        self.rope_theta = rope_theta
        self.use_gate = use_gate
        self.qkv_bias = qkv_bias
        self.layer_idx = layer_idx
        self.scale = head_dim ** -0.5

        # Projections - K/V use smaller dimension for GQA
        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=qkv_bias, **factory_kwargs)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias, **factory_kwargs)  # GQA: smaller K
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=qkv_bias, **factory_kwargs)  # GQA: smaller V
        self.o_proj = nn.Linear(self.inner_dim, hidden_size, bias=False, **factory_kwargs)

        # Gate projection
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.inner_dim, bias=False, **factory_kwargs)

        # Output normalization
        self.o_norm = RMSNorm(size=head_dim, eps=norm_eps)

        # Q/K normalization (Zero-Centered RMSNorm as per Qwen3-Next)
        # Zero-centering helps with training stability for Q/K projections
        self.q_norm = ZeroCenteredRMSNorm(size=head_dim, eps=norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(size=head_dim, eps=norm_eps)

        # Rotary embeddings (if enabled)
        if use_rotary:
            self.rotary_dim = int(head_dim * rope_percentage)
            self.rotary = RotaryEmbedding(self.rotary_dim, base=rope_theta)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.dropout_p = dropout  # [FIX] Store dropout probability directly

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (B, H, L, D) - Q uses num_heads, K/V use num_key_value_heads
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_key_value_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_key_value_heads)
        
        # GQA: Expand K/V heads to match Q heads
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)  # (B, num_heads, L, D)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)  # (B, num_heads, L, D)

        # Apply Q/K normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings (Partial RoPE)
        if self.use_rotary:
            q_rope, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
            k_rope, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
            q_rope, k_rope = self.rotary(q_rope, k_rope, seq_len=seq_len)
            q = torch.cat([q_rope, q_pass], dim=-1)
            k = torch.cat([k_rope, k_pass], dim=-1)

        # [OPTIMIZED] Attention Backend Selection
        # - Training: SDPA (stable gradients, well-tested)
        # - Inference: SageAttention3 (2.1x faster, INT8 quantization)
        if self.training:
            # Training: Use SDPA for stable gradients
            # [FIX] Use stored dropout_p instead of self.dropout.p
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True if attention_mask is None else False,
                scale=self.scale
            )
        else:
            # Inference: Use SageAttention3 for speed
            try:
                from sageattention import sageattn
                attn_output = sageattn(
                    q, k, v,
                    is_causal=True if attention_mask is None else False,
                    scale=self.scale
                )
            except ImportError:
                # Fallback to SDPA if SageAttention not available
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=True if attention_mask is None else False,
                    scale=self.scale
                )

        # Apply output normalization per head
        attn_output = self.o_norm(attn_output)

        # Apply gate
        if self.use_gate:
            g = self.g_proj(hidden_states)
            g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)
            # Use Sigmoid as per diagram (sigma')
            attn_output = attn_output * torch.sigmoid(g)

        # Reshape and project output
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        output = self.o_proj(attn_output)

        return output, None


class GatedAttentionBlock(TransformerBlockBase):
    """Gated Attention Block"""
    def __init__(
        self,
        config: GatedAttentionConfig,
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
        use_moe: bool = False,
        moe_config: Optional["MoEConfig"] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.config = config
        self.layer_idx = layer_idx
        
        self.attn_norm = layer_norm.build(config.hidden_size, init_device=init_device)
        self.attn = config.build(layer_idx=layer_idx)
        self.mlp_norm = layer_norm.build(config.hidden_size, init_device=init_device)
        
        if use_moe and moe_config is not None:
            from olmo_core.nn.transformer.moe import MoELayer
            self.mlp = MoELayer(moe_config, init_device=init_device)
        else:
            from olmo_core.nn.feed_forward import FeedForward, FeedForwardConfig
            ff_config = FeedForwardConfig(
                hidden_size=config.intermediate_size or (config.hidden_size * config.hidden_ratio),
            )
            self.mlp = ff_config.build(config.hidden_size, init_device=init_device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, **kwargs)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        mlp_out = self.mlp(x)
        aux_loss = None
        if isinstance(mlp_out, tuple):
             mlp_out, aux_loss = mlp_out
        x = residual + mlp_out

        return x, aux_loss

    def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False):
        raise NotImplementedError("TP not implemented for GatedAttentionBlock")

    def apply_cp(self, cp_mesh, load_balancer, head_stride=1):
        raise NotImplementedError("CP not implemented for GatedAttentionBlock")

    def apply_fsdp(self, dp_mesh=None, prefetch_factor=0, wrapping_strategy=None, **fsdp_kwargs):
        from torch.distributed.fsdp import fully_shard
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


class GatedAttentionBlockMoE(GatedAttentionBlock):
    """Gated Attention Block with MoE support."""
    def __init__(
        self,
        config: GatedAttentionConfig,
        moe_config: "MoEConfig",
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
    ):
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            n_layers=n_layers,
            layer_norm=layer_norm,
            init_device=init_device,
            use_moe=True,
            moe_config=moe_config,
        )
