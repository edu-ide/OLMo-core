"""
RMS-E Decoder Layer (Mamba2-NSA Style + Ouro LoopLM)

Structure (방식 A - 추가형):
- Every layer: GatedDeltaNet + MoE
- Every Nth layer: HSA → SWA → GatedDeltaNet + MoE (순차 실행)

Reference:
- Mamba2-NSA: https://github.com/ant-research/long-context-modeling
- HSA-UltraLong: arXiv 2511.23319
- Ouro/LoopLM: https://huggingface.co/ByteDance/Ouro-2.6B
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import RMSEConfig
from .utils import RMSNorm
from .moe import DeepSeekMoE, DenseFFN
from .attention import GatedSWA
from .hsa_ultralong import HSAUltraLongBlock

# Import FLA's GatedDeltaNet
try:
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    raise ImportError(
        "FLA (Flash Linear Attention) is REQUIRED.\n"
        "Install: pip install flash-linear-attention\n"
        "Reference: https://github.com/NVlabs/GatedDeltaNet"
    )


class RMSEDecoderLayer(nn.Module):
    """
    RMS-E Decoder Layer (Mamba2-NSA Style)

    Structure per layer:
    - Normal layers: GatedDeltaNet + MoE
    - Attention layers (every Nth): HSA → SWA → GatedDeltaNet + MoE

    Pattern (N=6, 48 layers):
    - Layer 1-5:   DeltaNet + MoE
    - Layer 6:     HSA → SWA → DeltaNet + MoE  (combined)
    - Layer 7-11:  DeltaNet + MoE
    - Layer 12:    HSA → SWA → DeltaNet + MoE  (combined)
    - ...
    """
    def __init__(self, config: RMSEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine if this layer has HSA + SWA
        self.has_attention = ((layer_idx + 1) % config.hybrid_attention_ratio == 0)

        # ===== 1. HSA-UltraLong (Every Nth layer, FIRST) =====
        if self.has_attention and getattr(config, 'use_hsa', True):
            # Set HSA config attributes
            config.chunk_size = getattr(config, 'hsa_chunk_size', 64)
            config.chunk_topk = getattr(config, 'hsa_chunk_topk', 8)
            config.retrieval_dim = getattr(config, 'hsa_retrieval_dim', 256)

            # HSA has internal pre_norm + residual
            self.hsa = HSAUltraLongBlock(config, layer_idx=layer_idx)
        else:
            self.hsa = None

        # ===== 2. Gated-SWA (Every Nth layer, SECOND) =====
        if self.has_attention and getattr(config, 'use_swa', True):
            self.swa_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.swa = GatedSWA(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                sliding_window=getattr(config, 'sliding_window', 512),
                headwise_gate=config.headwise_gate,
                use_qk_norm=config.use_qk_norm,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                use_yarn=config.use_yarn,
                original_max_position_embeddings=config.original_max_position_embeddings,
                yarn_beta_fast=config.yarn_beta_fast,
                yarn_beta_slow=config.yarn_beta_slow,
                yarn_mscale=config.yarn_mscale,
                yarn_rope_factor=config.yarn_rope_factor,
            )
        else:
            self.swa = None

        # ===== 3. GatedDeltaNet (EVERY layer, THIRD/ALWAYS) =====
        self.deltanet_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.deltanet = FLAGatedDeltaNet(
            hidden_size=config.hidden_size,
            num_heads=config.deltanet_num_heads,
            head_dim=config.deltanet_head_dim,
            expand_v=config.deltanet_expand_v,
            mode='chunk',
        )

        # ===== 4. FFN: MoE or Dense (EVERY layer) =====
        self.moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_moe = getattr(config, 'use_moe', True)

        if self.use_moe:
            # MoE (DeepSeek-style)
            self.ffn = DeepSeekMoE(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                aux_loss_alpha=config.aux_loss_alpha,
                use_loss_free_balancing=config.use_loss_free_balancing,
                bias_update_rate=config.bias_update_rate,
                route_scale=config.route_scale,
                score_func=config.score_func,
            )
        else:
            # Dense FFN (SwiGLU)
            self.ffn = DenseFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )

        # Gradient checkpointing (disabled by default)
        self.gradient_checkpointing = False

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        current_ut: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal forward implementation for gradient checkpointing."""
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # ===== 1. HSA-UltraLong (global sparse attention) =====
        if self.hsa is not None:
            hidden_states, _ = self.hsa(hidden_states)

        # ===== 2. Gated-SWA (local window attention) =====
        if self.swa is not None:
            residual = hidden_states
            hidden_states = self.swa_norm(hidden_states)
            hidden_states = self.swa(hidden_states)
            hidden_states = residual + hidden_states

        # ===== 3. GatedDeltaNet (linear attention) =====
        residual = hidden_states
        hidden_states = self.deltanet_norm(hidden_states)
        deltanet_output = self.deltanet(hidden_states)
        hidden_states = deltanet_output[0] if isinstance(deltanet_output, tuple) else deltanet_output
        hidden_states = residual + hidden_states

        # ===== 4. FFN (MoE or Dense) =====
        residual = hidden_states
        hidden_states = self.moe_norm(hidden_states)
        hidden_states, aux_loss = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        if aux_loss is not None:
            total_aux_loss = total_aux_loss + aux_loss

        return hidden_states, total_aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        current_ut: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: [HSA → SWA →] DeltaNet → MoE

        Mamba2-NSA style: HSA and SWA are added BEFORE DeltaNet at certain layers.
        Each component uses pre-norm + residual pattern.
        """
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            hidden_states, aux_loss = checkpoint(
                self._forward_impl,
                hidden_states,
                current_ut,
                use_reentrant=False,
            )
        else:
            hidden_states, aux_loss = self._forward_impl(hidden_states, current_ut)

        return hidden_states, aux_loss


__all__ = ["RMSEDecoderLayer"]
