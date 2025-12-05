"""
Triple-Hybrid Architecture: Mamba-3 + Gated DeltaNet + Gated Attention

세 가지 메모리 시스템을 결합한 하이브리드 아키텍처:
1. Mamba-3 (40%): Deep Long-term Memory - 복소수 상태 공간, 장기 의존성
2. Gated DeltaNet (40%): High-speed Working Memory - 델타 규칙 기반, 빠른 갱신
3. Gated Attention (20%): Focused Attention - 정밀한 검색, 핵심 정보 접근

레이어 배치: 4:4:2 비율
- Layer 0-3: Mamba-3 (깊은 장기 기억)
- Layer 4-7: Gated DeltaNet (고속 작업 메모리)
- Layer 8-9: Gated Attention (집중 어텐션)

Reference:
- Triple-Hybrid Architecture Design Document
- Mamba-3 for oscillatory dynamics
- Gated DeltaNet for fast recurrence
- Gated Attention for precise retrieval
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmo_core.config import Config, DType
from olmo_core.nn.transformer.block import TransformerBlockBase
from olmo_core.nn.layer_norm import LayerNormConfig, RMSNorm

# Import layer types
from olmo_core.nn.transformer.mamba_memory import Mamba3Config, Mamba3, Mamba3Block
from olmo_core.nn.transformer.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNet, GatedDeltaNetBlock


class LayerType(str, Enum):
    """Triple-Hybrid 레이어 타입"""
    MAMBA3 = "mamba3"           # Deep Long-term Memory
    GATED_DELTANET = "gated_deltanet"  # High-speed Working Memory
    GATED_ATTENTION = "gated_attention"  # Focused Attention


@dataclass
class TripleHybridConfig(Config):
    """
    Triple-Hybrid Architecture 설정.

    세 가지 레이어 타입을 4:4:2 비율로 배치:
    - Mamba-3: 40% (깊은 장기 기억)
    - Gated DeltaNet: 40% (고속 작업 메모리)
    - Gated Attention: 20% (집중 어텐션)
    """
    # Core dimensions
    hidden_size: int = 2048
    num_layers: int = 10

    # Layer distribution (default 4:4:2)
    mamba3_ratio: float = 0.4
    deltanet_ratio: float = 0.4
    attention_ratio: float = 0.2

    # Custom layer pattern (overrides ratios if set)
    layer_pattern: Optional[List[str]] = None

    # Mamba-3 config
    mamba3_d_state: int = 64
    mamba3_d_conv: int = 4
    mamba3_expand: int = 2
    mamba3_headdim: int = 128
    mamba3_use_complex: bool = True

    # Gated DeltaNet config
    deltanet_expand_v: float = 2.0
    deltanet_head_dim: int = 256
    deltanet_num_heads: int = 6
    deltanet_use_gate: bool = True

    # Gated Attention config
    attention_num_heads: int = 16
    attention_head_dim: int = 128
    attention_use_rotary: bool = True
    attention_use_gate: bool = True

    # Shared config
    hidden_ratio: int = 4
    norm_eps: float = 1e-5

    def get_layer_types(self) -> List[LayerType]:
        """
        레이어 타입 시퀀스 생성.

        Returns:
            각 레이어의 타입 리스트 (4:4:2 비율)
        """
        if self.layer_pattern is not None:
            return [LayerType(t) for t in self.layer_pattern]

        # Calculate layer counts
        n_mamba3 = int(self.num_layers * self.mamba3_ratio)
        n_deltanet = int(self.num_layers * self.deltanet_ratio)
        n_attention = self.num_layers - n_mamba3 - n_deltanet

        # Build pattern: Mamba3 → DeltaNet → Attention
        pattern = (
            [LayerType.MAMBA3] * n_mamba3 +
            [LayerType.GATED_DELTANET] * n_deltanet +
            [LayerType.GATED_ATTENTION] * n_attention
        )

        return pattern

    def build_mamba3_config(self) -> Mamba3Config:
        """Mamba-3 레이어 설정 생성"""
        return Mamba3Config(
            hidden_size=self.hidden_size,
            d_state=self.mamba3_d_state,
            d_conv=self.mamba3_d_conv,
            expand=self.mamba3_expand,
            headdim=self.mamba3_headdim,
            use_complex=self.mamba3_use_complex,
            hidden_ratio=self.hidden_ratio,
            norm_eps=self.norm_eps,
        )

    def build_deltanet_config(self) -> GatedDeltaNetConfig:
        """Gated DeltaNet 레이어 설정 생성"""
        return GatedDeltaNetConfig(
            hidden_size=self.hidden_size,
            expand_v=self.deltanet_expand_v,
            head_dim=self.deltanet_head_dim,
            num_heads=self.deltanet_num_heads,
            use_gate=self.deltanet_use_gate,
            hidden_ratio=self.hidden_ratio,
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
        head_dim: int = 128,
        use_rotary: bool = True,
        use_gate: bool = True,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.use_rotary = use_rotary
        self.use_gate = use_gate
        self.layer_idx = layer_idx
        self.scale = head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim, bias=False, **factory_kwargs)
        self.o_proj = nn.Linear(self.inner_dim, hidden_size, bias=False, **factory_kwargs)

        # Gate projection
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.inner_dim, bias=False, **factory_kwargs)

        # Output normalization
        self.o_norm = RMSNorm(head_dim, eps=norm_eps)

        # Rotary embeddings (if enabled)
        if use_rotary:
            self.rotary = RotaryEmbedding(head_dim)

        self.dropout = nn.Dropout(dropout)

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

        # Reshape to (B, H, L, D)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Apply rotary embeddings
        if self.use_rotary:
            q, k = self.rotary(q, k, seq_len=seq_len)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Apply output normalization per head
        attn_output = self.o_norm(attn_output)

        # Apply gate
        if self.use_gate:
            g = self.g_proj(hidden_states)
            g = rearrange(g, 'b l (h d) -> b h l d', h=self.num_heads)
            attn_output = attn_output * F.silu(g)

        # Reshape and project output
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        output = self.o_proj(attn_output)

        return output, None


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
    head_dim: int = 128
    use_rotary: bool = True
    use_gate: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-5
    hidden_ratio: int = 4

    def build(self, layer_idx: Optional[int] = None) -> GatedAttention:
        return GatedAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_rotary=self.use_rotary,
            use_gate=self.use_gate,
            dropout=self.dropout,
            layer_idx=layer_idx,
            norm_eps=self.norm_eps,
        )


class GatedAttentionBlock(TransformerBlockBase):
    """Gated Attention Block"""
    def __init__(
        self,
        config: GatedAttentionConfig,
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
    ):
        super().__init__(n_layers=n_layers)
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = layer_norm.build(config.hidden_size, init_device=init_device)
        self.attn = config.build(layer_idx=layer_idx)
        self.mlp_norm = layer_norm.build(config.hidden_size, init_device=init_device)

        from olmo_core.nn.feed_forward import FeedForward, FeedForwardConfig
        ff_config = FeedForwardConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * config.hidden_ratio,
            activation="swish",
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
        x = self.mlp(x)
        x = residual + x

        return x

    def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False):
        raise NotImplementedError("TP not implemented for GatedAttentionBlock")

    def apply_cp(self, cp_mesh, load_balancer, head_stride=1):
        raise NotImplementedError("CP not implemented for GatedAttentionBlock")

    def apply_fsdp(self, dp_mesh=None, prefetch_factor=0, wrapping_strategy=None, **fsdp_kwargs):
        from torch.distributed.fsdp import fully_shard
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


class TripleHybridTransformer(nn.Module):
    """
    Triple-Hybrid Transformer: Mamba-3 + Gated DeltaNet + Gated Attention

    세 가지 메모리 시스템을 4:4:2 비율로 결합:
    - 초기 레이어 (Mamba-3): 장기 기억 형성
    - 중간 레이어 (DeltaNet): 빠른 작업 메모리
    - 후기 레이어 (Attention): 정밀 검색
    """
    def __init__(
        self,
        config: TripleHybridConfig,
        layer_norm: Optional[LayerNormConfig] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.config = config

        if layer_norm is None:
            layer_norm = LayerNormConfig(eps=config.norm_eps)

        # Get layer pattern
        layer_types = config.get_layer_types()
        self.layer_types = layer_types

        # Build layers
        self.layers = nn.ModuleList()
        for idx, layer_type in enumerate(layer_types):
            if layer_type == LayerType.MAMBA3:
                layer_config = config.build_mamba3_config()
                block = Mamba3Block(
                    config=layer_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                )
            elif layer_type == LayerType.GATED_DELTANET:
                layer_config = config.build_deltanet_config()
                block = GatedDeltaNetBlock(
                    config=layer_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                )
            elif layer_type == LayerType.GATED_ATTENTION:
                layer_config = GatedAttentionConfig(
                    hidden_size=config.hidden_size,
                    num_heads=config.attention_num_heads,
                    head_dim=config.attention_head_dim,
                    use_rotary=config.attention_use_rotary,
                    use_gate=config.attention_use_gate,
                    hidden_ratio=config.hidden_ratio,
                    norm_eps=config.norm_eps,
                )
                block = GatedAttentionBlock(
                    config=layer_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.layers.append(block)

        # Final layer norm
        self.final_norm = layer_norm.build(config.hidden_size, init_device=init_device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass through Triple-Hybrid layers.

        Args:
            hidden_states: (B, L, D) input embeddings
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional cache
            use_cache: Whether to return cache

        Returns:
            hidden_states: (B, L, D) output
            new_cache: Optional cache dict
        """
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return hidden_states, None

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """각 레이어의 타입과 설정 정보 반환"""
        info = []
        for idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            info.append({
                "layer_idx": idx,
                "type": layer_type.value,
                "memory_role": self._get_memory_role(layer_type),
            })
        return info

    def _get_memory_role(self, layer_type: LayerType) -> str:
        """레이어 타입에 따른 메모리 역할 반환"""
        roles = {
            LayerType.MAMBA3: "Deep Long-term Memory (복소수 SSM, 장기 의존성)",
            LayerType.GATED_DELTANET: "High-speed Working Memory (델타 규칙, 빠른 갱신)",
            LayerType.GATED_ATTENTION: "Focused Attention (정밀 검색, 핵심 정보)",
        }
        return roles.get(layer_type, "Unknown")


def create_triple_hybrid_model(
    hidden_size: int = 2048,
    num_layers: int = 10,
    mamba3_ratio: float = 0.4,
    deltanet_ratio: float = 0.4,
    attention_ratio: float = 0.2,
    **kwargs,
) -> TripleHybridTransformer:
    """
    Triple-Hybrid 모델 생성 헬퍼 함수.

    Args:
        hidden_size: 히든 차원
        num_layers: 전체 레이어 수
        mamba3_ratio: Mamba-3 비율 (default: 0.4)
        deltanet_ratio: DeltaNet 비율 (default: 0.4)
        attention_ratio: Attention 비율 (default: 0.2)
        **kwargs: 추가 설정

    Returns:
        TripleHybridTransformer 인스턴스
    """
    config = TripleHybridConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mamba3_ratio=mamba3_ratio,
        deltanet_ratio=deltanet_ratio,
        attention_ratio=attention_ratio,
        **kwargs,
    )
    return TripleHybridTransformer(config)


# Utility function for analysis
def analyze_layer_distribution(config: TripleHybridConfig) -> Dict[str, Any]:
    """
    레이어 분포 분석.

    Returns:
        각 타입별 레이어 수와 비율
    """
    layer_types = config.get_layer_types()
    counts = {
        LayerType.MAMBA3: 0,
        LayerType.GATED_DELTANET: 0,
        LayerType.GATED_ATTENTION: 0,
    }
    for lt in layer_types:
        counts[lt] += 1

    total = len(layer_types)
    return {
        "total_layers": total,
        "distribution": {
            "mamba3": {"count": counts[LayerType.MAMBA3], "ratio": counts[LayerType.MAMBA3] / total},
            "gated_deltanet": {"count": counts[LayerType.GATED_DELTANET], "ratio": counts[LayerType.GATED_DELTANET] / total},
            "gated_attention": {"count": counts[LayerType.GATED_ATTENTION], "ratio": counts[LayerType.GATED_ATTENTION] / total},
        },
        "layer_sequence": [lt.value for lt in layer_types],
    }
