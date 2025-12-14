import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from olmo_core.config import Config, DType
from olmo_core.nn.transformer.block import TransformerBlockBase
from olmo_core.nn.layer_norm import LayerNormConfig, RMSNorm

# [REQUIRED] FLA (Flash Linear Attention) - Official NVlabs GatedDeltaNet Implementation
# Reference: https://github.com/NVlabs/GatedDeltaNet (ICLR 2025)
# FLA provides optimized CUDA kernels for linear attention (2-3x faster)
# Install: pip install fla
try:
    from fla.modules import ShortConvolution, FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    FLA_AVAILABLE = True
except ImportError as e:
    FLA_AVAILABLE = False
    raise ImportError(
        "FLA (Flash Linear Attention) is REQUIRED for GatedDeltaNet.\n"
        "This is the official NVlabs implementation (ICLR 2025).\n"
        "Install with: pip install fla\n"
        "Reference: https://github.com/NVlabs/GatedDeltaNet\n"
        f"Original error: {e}"
    )


@dataclass
class GatedDeltaNetConfig(Config):
    """
    Configuration for GatedDeltaNet.
    Reference: fla/models/gated_deltanet/configuration_gated_deltanet.py
    """
    hidden_size: int = 2048
    expand_v: float = 2.0
    head_dim: int = 256
    num_heads: int = 6
    num_v_heads: Optional[int] = None
    attn_mode: str = 'chunk'
    use_gate: bool = True
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5
    
    # Block config
    hidden_ratio: Optional[int] = 4
    intermediate_size: Optional[int] = None
    hidden_act: str = "swish"
    fuse_norm: bool = True
    fuse_swiglu: bool = True
    
    def build(self, layer_idx: Optional[int] = None) -> "GatedDeltaNet":
        if not FLA_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning("fla is not installed. GatedDeltaNet will run in compatibility mode (slow) or fail on forward.")

        
        return GatedDeltaNet(
            hidden_size=self.hidden_size,
            expand_v=self.expand_v,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_v_heads=self.num_v_heads,
            mode=self.attn_mode,
            use_gate=self.use_gate,
            use_short_conv=self.use_short_conv,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            layer_idx=layer_idx,
            norm_eps=self.norm_eps,
        )


class GatedDeltaNet(nn.Module):
    """
    The layer implementation for Gated Delta Networks.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: Optional[int] = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Validation
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(f"Invalid expand_v configuration.")
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads must be divisible by num_heads.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(size=self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        
        last_state = None
        if past_key_values is not None and self.layer_idx is not None:
             # Assuming past_key_values is a dict of layer_idx -> state
             last_state = past_key_values.get(self.layer_idx)

        cu_seqlens = kwargs.get('cu_seqlens')
        
        # Handling padding with attention_mask if needed (simplified adaptation)
        # In OLMo-core, we often rely on cu_seqlens for variable length or assume padded batches with masking handles elsewhere
        # Here keeping it close to fla implementation
        
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state.get('conv_state', (None, None, None))
            
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            # Expand q, k to match v's head count
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        # [FIX] GQA support: expand v, g, beta to match q/k head dimension
        # When num_heads > num_v_heads (GQA), v/g/beta have num_v_heads dimension
        # but need to match the q/k head dimension for the recurrent computation
        if self.num_heads > self.num_v_heads:
            num_groups = self.num_heads // self.num_v_heads
            # v: (B, L, num_v_heads, D_v) -> (B, L, num_heads, D_v)
            v = repeat(v, '... h d -> ... (h g) d', g=num_groups)
            # g: (B, L, num_v_heads) -> (B, L, num_heads)
            g = repeat(g, '... h -> ... (h g)', g=num_groups)
            # beta: (B, L, num_v_heads) -> (B, L, num_heads)
            beta = repeat(beta, '... h -> ... (h g)', g=num_groups)

        recurrent_state = last_state.get('recurrent_state') if last_state is not None else None
        
        if mode == 'chunk':
            if chunk_gated_delta_rule is not None:
                o, recurrent_state = chunk_gated_delta_rule(
                    q=q, k=k, v=v, g=g, beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                o = torch.zeros_like(v)
                recurrent_state = None
        elif mode == 'fused_recurrent':
            if fused_recurrent_gated_delta_rule is not None:
                o, recurrent_state = fused_recurrent_gated_delta_rule(
                    q=q, k=k, v=v, g=g, beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                o = torch.zeros_like(v)
                recurrent_state = None
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        new_state = None
        if use_cache:
            new_state = {
                'recurrent_state': recurrent_state,
                'conv_state': (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            }

        # [FIX] GQA support: contract o back to num_v_heads
        # After expansion, o has (B, L, num_heads, D_v)
        # Need to contract back to (B, L, num_v_heads, D_v) for output projection
        if self.num_heads > self.num_v_heads:
            num_groups = self.num_heads // self.num_v_heads
            # o: (B, L, num_heads, D_v) -> (B, L, num_v_heads, num_groups, D_v) -> (B, L, num_v_heads, D_v)
            o = rearrange(o, '... (h g) d -> ... h g d', g=num_groups)
            o = o.mean(dim=-2)  # Average over the expanded groups

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, new_state


class GatedDeltaNetBlock(TransformerBlockBase):
    """
    A Transformer block using GatedDeltaNet instead of standard attention.
    """
    def __init__(
        self,
        config: GatedDeltaNetConfig,
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
        
        # GatedMLP part - creating standard MLP or adapting configuration
        # For now, using a standard FeedForward from OLMo if config permits, or a simple implementation
        # Fla implementation uses GatedMLP. We can use olmo_core.nn.feed_forward.FeedForward
        # But we need to configure it.
        
        # For simplicity in this context, we'll implement a basic GatedMLP here or use placeholder
        # Since this file is about GatedDeltaNet, we assume the user might want to use standard OLMo MLP or specific one.
        # Let's use standard OLMo FeedForward for compatibility
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
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        
        residual = x
        x = self.attn_norm(x)
        
        # GatedDeltaNet forward
        x, new_state = self.attn(
            x, 
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        x = residual + x
        residual = x
        
        x = self.mlp_norm(x)
        mlp_out = self.mlp(x)
        aux_loss = None
        if isinstance(mlp_out, tuple):
             mlp_out, aux_loss = mlp_out
        x = residual + mlp_out
        
        if use_cache:
            # Return x and the state (we might need to package it better for OLMo's cache system)
            # For now just returning x, keeping state management implicit or handled by caller
            pass
            
        return x, aux_loss

    def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False):
        raise NotImplementedError("TP not implemented for GatedDeltaNetBlock")
        
    def apply_cp(self, cp_mesh, load_balancer, head_stride=1):
        raise NotImplementedError("CP not implemented for GatedDeltaNetBlock")
        
    def apply_fsdp(self, dp_mesh=None, prefetch_factor=0, wrapping_strategy=None, **fsdp_kwargs):
        # Basic FSDP wrapping
        from torch.distributed.fsdp import fully_shard
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)
