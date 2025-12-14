"""

Quad-Hybrid Architecture: ATLAS + Mamba-3 + Gated DeltaNet + Gated Attention

세 가지 메모리 시스템을 결합한 하이브리드 아키텍처:
1. ATLAS (10%): Neural Long-term Memory - 10M+ 토큰, Omega Rule
2. Mamba-3 (35%): Deep Long-term Memory - 복소수 상태 공간, 장기 의존성
3. Gated DeltaNet (35%): High-speed Working Memory - 델타 규칙 기반, 빠른 갱신
4. Gated Attention (20%): Focused Attention - 정밀한 검색, 핵심 정보 접근

레이어 배치: 1:3.5:3.5:2 비율
- Layer 0: ATLAS (초장기 기억)
- Layer 1-4: Mamba-3 (깊은 장기 기억)
- Layer 5-7: Gated DeltaNet (고속 작업 메모리)
- Layer 8-9: Gated Attention (집중 어텐션)

Reference:
- Quad-Hybrid Architecture Design Document
- ATLAS for episodic memory
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
from olmo_core.nn.transformer.moe import MoEConfig, MoELayer
from olmo_core.nn.transformer.atlas_memory import ATLASConfig, ATLASMemory, ATLASBlock, AtlasBlockMoE
from olmo_core.nn.transformer.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetBlock
from olmo_core.nn.transformer.gated_attention import (
    GatedAttention,
    GatedAttentionConfig,
    GatedAttentionBlock,
    GatedAttentionBlockMoE,
    RotaryEmbedding,
)


class LayerType(str, Enum):
    """Quad-Hybrid 레이어 타입"""
    MAMBA3 = "mamba3"           # Deep Long-term Memory
    GATED_DELTANET = "gated_deltanet"  # High-speed Working Memory
    GATED_ATTENTION = "gated_attention"  # Focused Attention
    # MoE Variants (validated in research)
    GATED_DELTANET_MOE = "gated_deltanet_moe"
    GATED_ATTENTION_MOE = "gated_attention_moe"
    MAMBA3_MOE = "mamba3_moe"
    ATLAS = "atlas"             # Episodic Memory (no MoE - not validated)


@dataclass
class QuadHybridConfig(Config):
    """
    Quad-Hybrid Architecture 설정.

    네 가지 레이어 타입을 1:3.5:3.5:2 비율로 배치:
    - ATLAS: 10% (초장기 기억)
    - Mamba-3: 35% (깊은 장기 기억)
    - Gated DeltaNet: 35% (고속 작업 메모리)
    - Gated Attention: 20% (집중 어텐션)
    """
    # Core dimensions
    hidden_size: int = 2048
    num_layers: int = 10

    # Layer distribution (default 4:4:2)
    # Layer distribution (default 4:4:2 for Triple Hybrid, or Qwen3-Next pattern)
    mamba3_ratio: float = 0.35
    deltanet_ratio: float = 0.35
    attention_ratio: float = 0.2
    atlas_ratio: float = 0.1 # 10% for Episodic Memory
    
    # Architecture Mode
    # arch_mode removed as we are fixing on Triple Hybrid + MoE

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
    attention_rope_percentage: float = 0.5  # Partial RoPE percentage

    # MoE Config (Qwen3-Next)
    use_moe: bool = True
    moe_num_experts: int = 512
    moe_num_shared_experts: int = 1
    moe_num_routed_experts: int = 10
    moe_intermediate_size: int = 14336

    # Shared config
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    norm_eps: float = 1e-5

    def get_layer_types(self) -> List[LayerType]:
        """
        레이어 타입 시퀀스 생성.

        Returns:
            각 레이어의 타입 리스트 (4:4:2 비율)
        """
        if self.layer_pattern is not None:
            return [LayerType(t) for t in self.layer_pattern]

        # Triple Hybrid + MoE Pattern (Quad-Hybrid with ATLAS)
        # 1:3.5:3.5:2 ratio (or configured) with MoE on all layers
        n_atlas = max(1, int(self.num_layers * self.atlas_ratio))
        remaining = self.num_layers - n_atlas
        
        # Re-normalize ratios for remaining layers
        total_remaining_ratio = self.mamba3_ratio + self.deltanet_ratio + self.attention_ratio
        if total_remaining_ratio == 0:
             # Fallback if ratios are messed up
             n_mamba3 = remaining // 3
             n_deltanet = remaining // 3
             n_attention = remaining - n_mamba3 - n_deltanet
        else:
             n_mamba3 = int(remaining * (self.mamba3_ratio / total_remaining_ratio))
             n_deltanet = int(remaining * (self.deltanet_ratio / total_remaining_ratio))
             n_attention = remaining - n_mamba3 - n_deltanet
        
        pattern = (
            [LayerType.ATLAS] * n_atlas +  # ATLAS without MoE (not validated)
            [LayerType.MAMBA3_MOE] * n_mamba3 +
            [LayerType.GATED_DELTANET_MOE] * n_deltanet +
            [LayerType.GATED_ATTENTION_MOE] * n_attention
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
            intermediate_size=self.intermediate_size,
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
            intermediate_size=self.intermediate_size,
        )

    def build_atlas_config(self) -> ATLASConfig:
        """ATLAS 레이어 설정 생성"""
        return ATLASConfig(
            hidden_size=self.hidden_size,
            # Use default ATLAS params or add to TripleHybridConfig if needed
            # For now using defaults from ATLASConfig but ensuring hidden_size matches
            hidden_ratio=self.hidden_ratio,
            norm_eps=self.norm_eps,
        )

    def build_moe_config(self) -> MoEConfig:
        """MoE 설정 생성"""
        return MoEConfig(
            hidden_size=self.hidden_size,
            num_experts=self.moe_num_experts,
            num_shared_experts=self.moe_num_shared_experts,
            num_routed_experts=self.moe_num_routed_experts,
            intermediate_size=self.moe_intermediate_size,
            norm_eps=self.norm_eps,
        )


class QuadHybridTransformer(nn.Module):
    """
    Quad-Hybrid Transformer: ATLAS + Mamba-3 + Gated DeltaNet + Gated Attention

    네 가지 메모리 시스템을 1:3.5:3.5:2 비율로 결합:
    - ATLAS: 초장기 기억 (Episodic Memory)
    - Mamba-3: 장기 기억 (Deep Long-term Memory)
    - DeltaNet: 작업 메모리 (Working Memory)
    - Attention: 정밀 검색 (Focused Attention)
    """
    def __init__(
        self,
        config: QuadHybridConfig,
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
                    rope_percentage=config.attention_rope_percentage,
                    use_gate=config.attention_use_gate,
                    hidden_ratio=config.hidden_ratio,
                    intermediate_size=config.intermediate_size,
                    norm_eps=config.norm_eps,
                )
                block = GatedAttentionBlock(
                    config=layer_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                    use_moe=False, # Explicitly set to False for non-MoE block
                    moe_config=None, # Explicitly set to None
                )
            elif layer_type == LayerType.GATED_DELTANET_MOE:
                layer_config = config.build_deltanet_config()
                moe_config = config.build_moe_config()
                block = GatedDeltaNetBlockMoE(
                    config=layer_config,
                    moe_config=moe_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                )
            elif layer_type == LayerType.GATED_ATTENTION_MOE:
                layer_config = GatedAttentionConfig(
                    hidden_size=config.hidden_size,
                    num_heads=config.attention_num_heads,
                    head_dim=config.attention_head_dim,
                    use_rotary=config.attention_use_rotary,
                    rope_percentage=config.attention_rope_percentage,
                    use_gate=config.attention_use_gate,
                    hidden_ratio=config.hidden_ratio,
                    intermediate_size=config.intermediate_size,
                    norm_eps=config.norm_eps,
                )
                moe_config = config.build_moe_config()
                block = GatedAttentionBlock(
                    config=layer_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                    use_moe=True,
                    moe_config=moe_config,
                )
            elif layer_type == LayerType.MAMBA3_MOE:
                layer_config = config.build_mamba3_config()
                moe_config = config.build_moe_config()
                block = Mamba3BlockMoE(
                    config=layer_config,
                    moe_config=moe_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                )
            elif layer_type == LayerType.ATLAS:
                layer_config = config.build_atlas_config()
                # [FIX] Use correct class name ATLASBlock (not AtlasBlock)
                block = ATLASBlock(
                    config=layer_config,
                    layer_idx=idx,
                    n_layers=config.num_layers,
                    layer_norm=layer_norm,
                    init_device=init_device,
                )
            # ATLAS_MOE removed - not validated in research
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
        Forward pass through Quad-Hybrid layers.

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
            out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            if isinstance(out, tuple):
                hidden_states = out[0]
            else:
                hidden_states = out

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
            LayerType.ATLAS: "Neural Long-term Memory (10M+ 토큰, Omega Rule)",
        }
        return roles.get(layer_type, "Unknown")


class GatedDeltaNetBlockMoE(GatedDeltaNetBlock):
    """
    GatedDeltaNetBlock with MoE support.
    """
    def __init__(
        self,
        config: GatedDeltaNetConfig,
        moe_config: MoEConfig,
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
    ):
        # Initialize parent to get attn and norms
        super().__init__(config, layer_idx, n_layers, layer_norm, init_device)
        
        # Override MLP with MoE
        self.mlp = MoELayer(moe_config, init_device=init_device)


class Mamba3BlockMoE(Mamba3Block):
    """
    Mamba3Block with MoE support.
    """
    def __init__(
        self,
        config: Mamba3Config,
        moe_config: MoEConfig,
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
    ):
        # Initialize parent to get attn and norms
        super().__init__(config, layer_idx, n_layers, layer_norm, init_device)
        
        # Override MLP with MoE
        self.mlp = MoELayer(moe_config, init_device=init_device)





def create_quad_hybrid_model(
    hidden_size: int = 2048,
    num_layers: int = 10,
    mamba3_ratio: float = 0.35,
    deltanet_ratio: float = 0.35,
    attention_ratio: float = 0.2,
    atlas_ratio: float = 0.1,
    **kwargs,
) -> QuadHybridTransformer:
    """
    Quad-Hybrid 모델 생성 헬퍼 함수.

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
    config = QuadHybridConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mamba3_ratio=mamba3_ratio,
        deltanet_ratio=deltanet_ratio,
        attention_ratio=attention_ratio,
        atlas_ratio=atlas_ratio,
        **kwargs,
    )
    return QuadHybridTransformer(config)


# Utility function for analysis
def analyze_layer_distribution(config: QuadHybridConfig) -> Dict[str, Any]:
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
        LayerType.ATLAS: 0,
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
            "atlas": {"count": counts[LayerType.ATLAS], "ratio": counts[LayerType.ATLAS] / total},
        },
        "layer_sequence": [lt.value for lt in layer_types],
    }
