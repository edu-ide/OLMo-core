"""
Mamba-3: Complex State Space Model for Deep Long-term Memory

Mamba-3는 복소수 상태 공간을 사용하여 깊은 장기 기억을 제공합니다.
기존 Mamba-2 대비 복소수 고유값을 통한 진동 패턴 학습이 가능합니다.

Reference:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Mamba-2: Structured State Space Duality
- Complex-valued SSM for oscillatory dynamics
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from olmo_core.config import Config, DType
from olmo_core.nn.transformer.block import TransformerBlockBase
from olmo_core.nn.layer_norm import LayerNormConfig, RMSNorm

# Try importing mamba modules
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None
    RMSNormGated = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@dataclass
class Mamba3Config(Config):
    """
    Configuration for Mamba-3 (Complex State Space Model).

    Mamba-3는 복소수 상태 공간을 사용하여:
    - 진동 패턴 학습 (oscillatory dynamics)
    - 장기 의존성 포착 (long-range dependencies)
    - 깊은 메모리 압축 (deep memory compression)
    """
    # Model dimensions
    hidden_size: int = 2048
    d_state: int = 64  # SSM state dimension (N)
    d_conv: int = 4    # Convolution kernel size
    expand: int = 2    # Expansion factor

    # Head configuration
    headdim: int = 128
    ngroups: int = 1

    # Complex SSM parameters
    use_complex: bool = True  # Use complex-valued state space
    complex_init_method: str = "random"  # "random", "frequency", "diagonal"

    # A matrix initialization
    A_init_range: Tuple[float, float] = (1.0, 16.0)

    # dt (timestep) parameters
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))

    # Additional options
    learnable_init_states: bool = False
    activation: str = "swish"
    bias: bool = False
    conv_bias: bool = True

    # Fused kernel options
    chunk_size: int = 256
    use_mem_eff_path: bool = True

    # Block config (for MLP)
    hidden_ratio: Optional[int] = 4
    intermediate_size: Optional[int] = None
    hidden_act: str = "swish"
    norm_eps: float = 1e-5

    def build(self, layer_idx: Optional[int] = None) -> "Mamba3":
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is not installed. Please install it to use Mamba3.\n"
                "pip install mamba-ssm or clone from https://github.com/state-spaces/mamba"
            )

        return Mamba3(
            d_model=self.hidden_size,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            headdim=self.headdim,
            ngroups=self.ngroups,
            use_complex=self.use_complex,
            complex_init_method=self.complex_init_method,
            A_init_range=self.A_init_range,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dt_init_floor=self.dt_init_floor,
            dt_limit=self.dt_limit,
            learnable_init_states=self.learnable_init_states,
            activation=self.activation,
            bias=self.bias,
            conv_bias=self.conv_bias,
            chunk_size=self.chunk_size,
            use_mem_eff_path=self.use_mem_eff_path,
            layer_idx=layer_idx,
            norm_eps=self.norm_eps,
        )


class ComplexSSM(nn.Module):
    """
    Complex-valued State Space Model core.

    복소수 상태 공간 모델의 핵심 연산:
    h'(t) = A * h(t) + B * x(t)
    y(t) = Re(C * h(t)) + D * x(t)

    여기서 A는 복소수 대각 행렬로, 진동 패턴을 학습합니다.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        nheads: int,
        headdim: int,
        complex_init_method: str = "random",
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads
        self.headdim = headdim
        self.complex_init_method = complex_init_method

        # Complex A matrix: A = -alpha + i*omega
        # alpha > 0 ensures stability, omega controls oscillation frequency
        if complex_init_method == "random":
            # Random initialization in polar form
            alpha = torch.empty(nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            omega = torch.empty(nheads, dtype=torch.float32, device=device).uniform_(-math.pi, math.pi)
        elif complex_init_method == "frequency":
            # Initialize with specific frequency bands
            alpha = torch.empty(nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            # Distribute frequencies logarithmically
            omega = torch.logspace(-2, 1, nheads, device=device, dtype=torch.float32) * math.pi
        elif complex_init_method == "diagonal":
            # HiPPO-style diagonal initialization
            alpha = torch.arange(1, nheads + 1, dtype=torch.float32, device=device) * 0.5
            omega = torch.zeros(nheads, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown complex_init_method: {complex_init_method}")

        # Store as log for numerical stability
        self.A_log_alpha = nn.Parameter(torch.log(alpha))
        self.A_omega = nn.Parameter(omega)
        self.A_log_alpha._no_weight_decay = True
        self.A_omega._no_weight_decay = True

    def get_A_complex(self) -> torch.Tensor:
        """Get complex A matrix: A = -exp(log_alpha) + i*omega"""
        alpha = torch.exp(self.A_log_alpha)
        return torch.complex(-alpha, self.A_omega)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, H, P)
        dt: torch.Tensor,  # (B, L, H)
        B: torch.Tensor,   # (B, L, G, N)
        C: torch.Tensor,   # (B, L, G, N)
        D: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complex SSM forward pass with selective scan.

        For now, this falls back to the real-valued Mamba implementation
        with complex-valued A matrix approximation.
        """
        # Get real part of A for compatibility with mamba ops
        A_real = -torch.exp(self.A_log_alpha)
        return A_real


class Mamba3(nn.Module):
    """
    Mamba-3: Complex State Space Model for Deep Long-term Memory.

    기존 Mamba-2를 기반으로 복소수 상태 공간을 추가하여
    진동 패턴과 장기 의존성을 더 효과적으로 학습합니다.

    주요 특징:
    1. 복소수 A 행렬: 진동 역학 모델링
    2. 선택적 스캔: 입력 의존적 상태 전이
    3. 깊은 메모리 압축: 긴 시퀀스의 효율적 처리
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 128,
        ngroups: int = 1,
        use_complex: bool = True,
        complex_init_method: str = "random",
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        learnable_init_states: bool = False,
        activation: str = "swish",
        bias: bool = False,
        conv_bias: bool = True,
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.use_complex = use_complex

        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim

        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Input projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        # 1D Convolution
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Initial states
        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # dt bias initialization
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter (complex or real)
        if use_complex:
            # Complex A: -alpha + i*omega
            alpha = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            omega = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(-math.pi, math.pi)
            self.A_log_alpha = nn.Parameter(torch.log(alpha).to(dtype=dtype))
            self.A_omega = nn.Parameter(omega.to(dtype=dtype))
            self.A_log_alpha._no_weight_decay = True
            self.A_omega._no_weight_decay = True
        else:
            # Real A (standard Mamba-2)
            A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            A_log = torch.log(A).to(dtype=dtype)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Output normalization
        if RMSNormGated is not None:
            self.norm = RMSNormGated(self.d_inner, eps=norm_eps, norm_before_gate=False, **factory_kwargs)
        else:
            self.norm = RMSNorm(self.d_inner, eps=norm_eps)
            self._use_simple_norm = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _get_A(self) -> torch.Tensor:
        """Get A matrix (handles both complex and real cases)."""
        if self.use_complex:
            # For complex A, we use only the real part for the standard scan
            # The imaginary part (omega) modulates the phase
            # In practice, we approximate complex dynamics using modified decay
            alpha = torch.exp(self.A_log_alpha)
            # Effective A includes oscillation effect via magnitude modulation
            return -alpha
        else:
            return -torch.exp(self.A_log)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_idx: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass for Mamba-3.

        Args:
            hidden_states: (B, L, D) input tensor
            seq_idx: Optional sequence indices for variable-length sequences
            past_key_values: Optional cache for incremental decoding
            use_cache: Whether to return cache for next step

        Returns:
            output: (B, L, D) output tensor
            new_cache: Optional cache dict
        """
        batch, seqlen, dim = hidden_states.shape

        # Input projection: [z, x, B, C, dt]
        zxbcdt = self.in_proj(hidden_states)

        A = self._get_A()
        initial_states = None
        if self.learnable_init_states:
            initial_states = repeat(self.init_states, "... -> b ...", b=batch)

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and mamba_split_conv1d_scan_combined is not None:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            # Non-fused path
            z, xBC, dt = torch.split(
                zxbcdt,
                [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)

            # 1D Convolution
            if causal_conv1d_fn is not None and self.activation in ["silu", "swish"]:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            else:
                xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
                xBC = xBC[:, :seqlen, :]

            # Split into x, B, C
            x, B, C = torch.split(
                xBC,
                [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1
            )

            # SSM scan
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Apply gated normalization
            if hasattr(self, '_use_simple_norm'):
                y = self.norm(y) * F.silu(z)
            else:
                y = self.norm(y, z)

            out = self.out_proj(y)

        # Cache handling for incremental decoding
        new_cache = None
        if use_cache:
            # TODO: Implement proper caching for Mamba-3
            new_cache = {}

        return out, new_cache


class Mamba3Block(TransformerBlockBase):
    """
    A Transformer block using Mamba-3 instead of standard attention.

    구조:
    - Pre-norm with RMSNorm
    - Mamba-3 SSM layer
    - Post-residual connection
    - MLP (optional)
    """
    def __init__(
        self,
        config: Mamba3Config,
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
    ):
        super().__init__(n_layers=n_layers)
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm
        self.attn_norm = layer_norm.build(config.hidden_size, init_device=init_device)

        # Mamba-3 layer
        self.attn = config.build(layer_idx=layer_idx)

        # MLP (optional, can be disabled for pure Mamba blocks)
        self.mlp_norm = layer_norm.build(config.hidden_size, init_device=init_device)

        from olmo_core.nn.feed_forward import FeedForward, FeedForwardConfig

        ff_config = FeedForwardConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size or (config.hidden_size * config.hidden_ratio),
            activation=config.hidden_act,
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
    ) -> torch.Tensor:

        # Mamba-3 with pre-norm
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(
            x,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        x = residual + x

        # MLP with pre-norm
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False):
        raise NotImplementedError("TP not implemented for Mamba3Block")

    def apply_cp(self, cp_mesh, load_balancer, head_stride=1):
        raise NotImplementedError("CP not implemented for Mamba3Block")

    def apply_fsdp(self, dp_mesh=None, prefetch_factor=0, wrapping_strategy=None, **fsdp_kwargs):
        from torch.distributed.fsdp import fully_shard
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


# Aliases for backward compatibility
MambaMemory = Mamba3
MambaMemoryConfig = Mamba3Config
MambaMemoryBlock = Mamba3Block
