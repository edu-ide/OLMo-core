"""
ATLAS: Deep Neural Long-term Memory for Transformers

ATLAS (DeepTransformer)는 Titans의 한계를 극복한 차세대 신경 메모리 시스템입니다.
10M+ 토큰 컨텍스트에서도 80%+ 정확도를 유지합니다 (Titans는 70%).

핵심 혁신:
1. Omega Rule: 슬라이딩 윈도우 최적화 (토큰별 아님)
2. Muon Optimizer: 2차 최적화로 빠른 수렴
3. Polynomial Kernels: 메모리 용량 O(d_k^p)로 확장
4. Deep MLP Memory: 2+ 레이어 메모리 구조

Reference:
- ATLAS: Learning to Optimally Memorize the Context at Test-time
  (https://arxiv.org/abs/2505.23735)
- Titans: Learning to Memorize at Test Time
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from olmo_core.config import Config, DType
from olmo_core.nn.transformer.moe import MoEConfig, MoELayer
from olmo_core.nn.layer_norm import LayerNormConfig, RMSNorm
from olmo_core.nn.transformer.block import TransformerBlockBase


class OptimizationRule(str, Enum):
    """메모리 최적화 규칙"""
    DELTA = "delta"      # Delta rule (Titans)
    OMEGA = "omega"      # Omega rule (ATLAS) - sliding window
    TTT = "ttt"          # Test-Time Training (TTT-Linear)


class KernelType(str, Enum):
    """커널 타입"""
    LINEAR = "linear"           # Linear kernel (standard)
    POLYNOMIAL = "polynomial"   # Polynomial kernel (ATLAS)
    RBF = "rbf"                # RBF kernel


@dataclass
class ATLASConfig(Config):
    """
    ATLAS Neural Memory Configuration.

    ATLAS는 다음 핵심 요소로 구성됩니다:
    - Omega Rule: 윈도우 기반 최적화 (W 토큰마다 업데이트)
    - Polynomial Kernel: 용량 확장 (degree p로 O(d_k^p) 용량)
    - Deep MLP: 깊은 메모리 구조 (2+ layers)
    - Muon Optimizer: 2차 최적화 (빠른 수렴)
    """
    # Core dimensions
    hidden_size: int = 2048
    memory_dim: int = 512       # Memory key/value dimension
    num_heads: int = 8          # Number of memory heads

    # Omega Rule parameters
    window_size: int = 64       # W: 슬라이딩 윈도우 크기
    optimization_steps: int = 3  # 윈도우당 최적화 스텝 수
    optimization_rule: str = "omega"  # "omega", "delta", "ttt"

    # Polynomial Kernel parameters
    kernel_type: str = "polynomial"   # "linear", "polynomial", "rbf"
    polynomial_degree: int = 2        # p: 다항 차수 (capacity = O(d_k^p))
    kernel_gamma: float = 1.0         # RBF 커널 감마

    # Deep MLP Memory parameters
    memory_layers: int = 2            # MLP 레이어 수
    memory_expansion: int = 4         # MLP 확장 비율
    memory_activation: str = "gelu"   # 활성화 함수

    # Muon Optimizer parameters (2차 최적화)
    use_muon: bool = True             # Muon 옵티마이저 사용
    muon_momentum: float = 0.95       # Muon 모멘텀
    muon_beta: float = 0.95           # Muon 2차 모멘텀
    learning_rate: float = 0.01       # 메모리 학습률

    # Gating and integration
    use_gate: bool = True             # 게이트 사용
    use_residual: bool = True         # 잔차 연결
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # Memory persistence
    max_memory_tokens: int = 10_000_000  # 최대 10M 토큰 메모리
    memory_decay: float = 0.9999         # 메모리 감쇠율

    # Block configuration
    hidden_ratio: int = 4

    def build(self, layer_idx: Optional[int] = None) -> "ATLASMemory":
        return ATLASMemory(
            hidden_size=self.hidden_size,
            memory_dim=self.memory_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            optimization_steps=self.optimization_steps,
            optimization_rule=OptimizationRule(self.optimization_rule),
            kernel_type=KernelType(self.kernel_type),
            polynomial_degree=self.polynomial_degree,
            kernel_gamma=self.kernel_gamma,
            memory_layers=self.memory_layers,
            memory_expansion=self.memory_expansion,
            memory_activation=self.memory_activation,
            use_muon=self.use_muon,
            muon_momentum=self.muon_momentum,
            muon_beta=self.muon_beta,
            learning_rate=self.learning_rate,
            use_gate=self.use_gate,
            use_residual=self.use_residual,
            dropout=self.dropout,
            norm_eps=self.norm_eps,
            memory_decay=self.memory_decay,
            layer_idx=layer_idx,
        )


class PolynomialKernel(nn.Module):
    """
    Polynomial Feature Kernel for ATLAS.

    다항 커널 φ(x) = [1, x, x⊗x, ..., x^⊗p]로 메모리 용량을 확장합니다.
    용량: O(d_k) → O(d_k^p)

    degree=2일 때: φ(x) = [x_i * x_j for all i,j] (외적)
    """
    def __init__(
        self,
        input_dim: int,
        degree: int = 2,
        output_dim: Optional[int] = None,
        learnable_weights: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree
        self.learnable_weights = learnable_weights

        # Calculate expanded dimension
        # For degree=2: d + d*(d+1)/2 (linear + quadratic terms)
        self.expanded_dim = self._calculate_expanded_dim(input_dim, degree)
        self.output_dim = output_dim or self.expanded_dim

        # Optional learnable projection
        if output_dim is not None and output_dim != self.expanded_dim:
            self.proj = nn.Linear(self.expanded_dim, output_dim, bias=False, **factory_kwargs)
        else:
            self.proj = None

        # Learnable scaling per degree
        if learnable_weights:
            self.degree_scales = nn.Parameter(
                torch.ones(degree, **factory_kwargs) / math.sqrt(degree)
            )

    def _calculate_expanded_dim(self, d: int, p: int) -> int:
        """다항 확장 후 차원 계산"""
        # degree 1: d
        # degree 2: d + d*(d+1)/2
        # degree p: sum of combinations
        total = 0
        for deg in range(1, p + 1):
            # 조합 수 계산 (중복 허용)
            from math import comb
            total += comb(d + deg - 1, deg)
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        다항 특징 확장.

        Args:
            x: (B, L, D) 입력 텐서

        Returns:
            (B, L, expanded_dim or output_dim) 확장된 텐서
        """
        # Normalize input for stability
        x_norm = F.normalize(x, dim=-1)

        # Expand features using polynomial basis
        expanded = self._expand_features(x_norm)

        # Optional projection to reduce dimensionality
        if self.proj is not None:
            expanded = self.proj(expanded)

        return expanded

    def _expand_features(self, x):
        """
        Polynomial feature expansion up to specified degree.

        Supports:
        - Degree 1: Linear features (x)
        - Degree 2: Quadratic features (x_i * x_j for i <= j)
        - Degree 3: Cubic features (x_i * x_j * x_k for i <= j <= k)
        - Degree > 3: Falls back to element-wise power (approximation)

        Note: Full polynomial expansion for degree > 3 has O(D^p) complexity,
        which becomes prohibitive for large dimensions. The power approximation
        captures self-interactions but not cross-term interactions.
        """
        batch, seq, dim = x.shape
        features = [x]

        if self.degree >= 2:
            # Degree 2: Full quadratic expansion
            # x_i * x_j for all i <= j (upper triangular of outer product)
            outer = torch.einsum('bli,blj->blij', x, x)
            triu_idx = torch.triu_indices(dim, dim, device=x.device)
            deg2 = outer[:, :, triu_idx[0], triu_idx[1]]
            if self.learnable_weights:
                deg2 = deg2 * self.degree_scales[1]
            features.append(deg2)

        if self.degree >= 3:
            # Degree 3: Full cubic expansion
            # x_i * x_j * x_k for all i <= j <= k
            # Memory: O(D^3) - manageable for D <= 128
            if dim <= 128:
                outer3 = torch.einsum('blij,blk->blijk', outer, x)  # (B, L, D, D, D)

                # Create mask for i <= j <= k (computed once, cached as buffer)
                if not hasattr(self, 'triu_idx_3_mask') or self.triu_idx_3_mask.device != x.device:
                    i, j, k = torch.meshgrid(
                        torch.arange(dim, device=x.device),
                        torch.arange(dim, device=x.device),
                        torch.arange(dim, device=x.device),
                        indexing='ij'
                    )
                    mask = (i <= j) & (j <= k)
                    self.register_buffer('triu_idx_3_mask', mask.flatten())

                deg3 = outer3.reshape(batch, seq, -1)[:, :, self.triu_idx_3_mask]
            else:
                # Fallback for large dimensions: element-wise power
                deg3 = x ** 3

            if self.learnable_weights:
                deg3 = deg3 * self.degree_scales[2]
            features.append(deg3)

        # Degree > 3: Fallback to element-wise power approximation
        # Full expansion would be O(D^p) which is prohibitive
        for deg in range(4, self.degree + 1):
            # Element-wise power captures self-interactions: x_i^deg
            # This is an approximation - true polynomial basis would include
            # all cross-term products x_i * x_j * x_k * ... for i <= j <= k <= ...
            deg_features = x ** deg

            if self.learnable_weights and deg - 1 < len(self.degree_scales):
                deg_features = deg_features * self.degree_scales[deg - 1]

            features.append(deg_features)

        return torch.cat(features, dim=-1)


class DeepMLPMemory(nn.Module):
    """
    Deep MLP Memory for ATLAS.

    2+ 레이어 MLP로 메모리를 구성하여 복잡한 패턴을 학습합니다.
    Titans의 단일 레이어 메모리보다 표현력이 높습니다.

    M(x) = W_L * σ(W_{L-1} * ... * σ(W_1 * x + b_1) + ... + b_{L-1}) + b_L
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        expansion: int = 4,
        activation: str = "gelu",
        use_bias: bool = False,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = input_dim * expansion

        # Activation function
        self.activation = self._get_activation(activation)

        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [input_dim] + [self.hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            self.layers.append(
                nn.Linear(dims[i], dims[i + 1], bias=use_bias, **factory_kwargs)
            )
            if i < num_layers - 1:  # No norm after last layer
                self.norms.append(RMSNorm(size=dims[i + 1], eps=norm_eps))

        self._reset_parameters()

    def _get_activation(self, name: str) -> Callable:
        activations = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
            "swish": F.silu,
            "tanh": torch.tanh,
        }
        return activations.get(name, F.gelu)

    def _reset_parameters(self):
        """Xavier initialization for stable training"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through deep MLP memory.

        Args:
            x: (B, L, input_dim) 입력

        Returns:
            (B, L, output_dim) 메모리 출력
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                x = self.activation(self.norms[i](x))

        return x

    def get_weights(self) -> List[torch.Tensor]:
        """Get all layer weights for optimization"""
        return [layer.weight for layer in self.layers]

    def set_weights(self, weights: List[torch.Tensor]):
        """Set all layer weights"""
        for layer, w in zip(self.layers, weights):
            layer.weight.data.copy_(w)


class MuonOptimizer:
    """
    Muon Optimizer for ATLAS Memory.

    2차 최적화로 메모리 업데이트를 가속합니다.
    Newton-Raphson의 근사로, 곡률 정보를 사용합니다.

    업데이트: θ = θ - lr * (1 / sqrt(v + ε)) * m
    여기서:
    - m: 1차 모멘텀 (gradient의 EMA)
    - v: 2차 모멘텀 (gradient^2의 EMA)
    """
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps

        # Initialize momentum buffers
        self.state = {}
        for p in self.params:
            self.state[p] = {'momentum': torch.zeros_like(p)}

    def step(self):
        """
        Perform a single optimization step using Muon (Newton-Schulz).
        """
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]
            buf = state['momentum']
            
            # Lazy device move
            if buf.device != grad.device:
                buf = buf.to(grad.device)
                state['momentum'] = buf

            # Update momentum
            buf.mul_(self.momentum).add_(grad)

            if self.nesterov:
                g = grad.add(buf, alpha=self.momentum)
            else:
                g = buf

            # Newton-Schulz Iteration for Orthogonalization
            # Reshape to 2D if necessary (e.g. for Conv2d or higher dim tensors)
            if g.ndim > 2:
                original_shape = g.shape
                g_2d = g.view(g.size(0), -1)
            elif g.ndim == 1:
                 # For 1D vectors, just normalize
                 original_shape = g.shape
                 g_2d = g.unsqueeze(1)
            else:
                original_shape = g.shape
                g_2d = g

            # Ensure dimensions are compatible for matmul (rows >= cols)
            transposed = False
            if g_2d.size(0) < g_2d.size(1):
                g_2d = g_2d.t()
                transposed = True

            # Normalize spectral norm
            norm = g_2d.norm() + 1e-8
            X = g_2d / norm

            # Newton-Schulz iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k * X_k^T * X_k
            # Or more stable: X_{k+1} = 1.5 * X_k - 0.5 * X_k * (X_k^T * X_k)
            for _ in range(self.ns_steps):
                A = X.T @ X
                B = 1.5 * X - 0.5 * X @ A
                X = B

            # Restore scale (optional, but Muon usually replaces Adam's adaptive scaling with this)
            # Muon update: theta = theta - lr * O_t
            update = X

            if transposed:
                update = update.t()

            if g.ndim != 2:
                update = update.view(original_shape)

            p.data.add_(update, alpha=-self.lr)

    def reset(self):
        """Reset optimizer state"""
        for p in self.params:
            self.state[p]['momentum'].zero_()


class OmegaRule(nn.Module):
    """
    Omega Rule: Window-based Memory Optimization for ATLAS.
    Reference: arXiv:2505.23735 - 'ATLAS: Learning to Optimally Memorize the Context at Test Time'

    Paper Equations:
    - Omega Rule (Eq. 10): ℳₜ = αₜℳₜ₋₁ - ∇∑γᵢ‖ℳ(kᵢ) - vᵢ‖²
    - Linear Memory Closed-Form (Eq. 11):
      ℳₜ = (αₜI - ∑γᵢkᵢkᵢᵀ)ℳₜ₋₁ + ∑γᵢvᵢkᵢᵀ

    Key insight from paper: "all other parameters from the model are considered
    hyperparameters and are fixed and not optimized" during inner loop.

    This implementation uses:
    1. Linear Memory: Closed-form update (no gradients needed, fast)
    2. Deep Memory: torch.no_grad() for inner optimization (isolated from outer backward)
    """
    def __init__(
        self,
        memory: DeepMLPMemory,
        window_size: int = 64,
        optimization_steps: int = 3,
        learning_rate: float = 0.01,
        use_muon: bool = True,
        muon_momentum: float = 0.95,
        muon_beta: float = 0.95,
        use_linear_memory: bool = True,  # Use closed-form linear memory by default
    ):
        super().__init__()

        self.memory = memory
        self.window_size = window_size
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate
        self.use_muon = use_muon
        self.use_linear_memory = use_linear_memory

        # Linear Memory State (closed-form, paper Eq. 11)
        # M is a matrix of shape (D, D) acting as associative memory
        if use_linear_memory:
            input_dim = memory.input_dim
            output_dim = memory.output_dim
            self.register_buffer('M', torch.zeros(output_dim, input_dim))
            # Snapshot tensors for CUDA graph / autograd safety.
            # We keep one independent tensor per window to avoid version-counter
            # conflicts when updating multiple windows in a single forward.
            self._M_snapshots: List[torch.Tensor] = []
            self.alpha = nn.Parameter(torch.tensor(0.9))  # Decay factor

        # Initialize optimizer for deep memory
        if not use_linear_memory and use_muon:
            self.optimizer = MuonOptimizer(
                params=list(memory.parameters()),
                lr=learning_rate,
                momentum=muon_momentum,
            )
        else:
            self.optimizer = None

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        gammas: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Omega Rule forward pass.

        Args:
            keys: (B, L, D) input keys
            values: (B, L, D) input values
            queries: (B, L, D) queries for retrieval
            gammas: (B, L, 1) input-dependent gates (Optional)

        Returns:
            memory_output: (B, L, D) retrieved values from memory
            info: debugging info
        """
        batch_size, seq_len, dim = keys.shape
        info = {"optimization_steps_run": 0, "windows_processed": 0}

        if self.use_linear_memory:
            # Linear Memory with closed-form update (Paper Eq. 11)
            return self._forward_linear(keys, values, queries, gammas, info)
        else:
            # Deep Memory with gradient-based optimization
            return self._forward_deep(keys, values, queries, gammas, info)

    def _forward_linear(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        gammas: Optional[torch.Tensor],
        info: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Linear Memory with closed-form Omega Rule update.
        Paper Equation 11:
            ℳₜ = (αₜI - ∑γᵢkᵢkᵢᵀ)ℳₜ₋₁ + ∑γᵢvᵢkᵢᵀ

        This is computed WITHOUT gradients - purely matrix operations.
        """
        batch_size, seq_len, dim = keys.shape
        outputs = []

        # Number of windows for this sequence length
        num_windows = (seq_len + self.window_size - 1) // self.window_size

        # Ensure snapshot list is large enough and matches device/dtype.
        # Each window gets its own tensor so later snapshots don't bump the
        # version counter of earlier ones (autograd safety).
        if self.training:
            need_rebuild = (
                len(self._M_snapshots) < num_windows
                or (len(self._M_snapshots) > 0 and (
                    self._M_snapshots[0].device != self.M.device
                    or self._M_snapshots[0].dtype != self.M.dtype
                ))
            )
            if need_rebuild:
                self._M_snapshots = [self.M.new_zeros(self.M.shape) for _ in range(num_windows)]

        # Process in windows
        for win_idx, start in enumerate(range(0, seq_len, self.window_size)):
            end = min(start + self.window_size, seq_len)

            window_keys = keys[:, start:end].detach()  # (B, W, D)
            window_values = values[:, start:end].detach()  # (B, W, D)
            window_queries = queries[:, start:end]  # Keep gradient for output
            window_gammas = gammas[:, start:end].detach() if gammas is not None else None

            # Update memory with closed-form solution (no gradients)
            if self.training:
                with torch.no_grad():
                    self._update_linear_memory(window_keys, window_values, window_gammas)
                    info["windows_processed"] += 1

            # Retrieve from memory: output = M @ query.
            # Snapshot per window so later updates don't mutate saved tensors.
            # M: (D_out, D_in), queries: (B, W, D_in) -> output: (B, W, D_out)
            if self.training:
                with torch.no_grad():
                    self._M_snapshots[win_idx].copy_(self.M)
                M_snapshot = self._M_snapshots[win_idx]
            else:
                M_snapshot = self.M

            window_output = torch.matmul(window_queries, M_snapshot.T)
            outputs.append(window_output)

        memory_output = torch.cat(outputs, dim=1)
        return memory_output, info

    def _update_linear_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        gammas: Optional[torch.Tensor],
    ):
        """
        Closed-form linear memory update (Paper Eq. 11).
        ℳₜ = (αₜI - ∑γᵢkᵢkᵢᵀ)ℳₜ₋₁ + ∑γᵢvᵢkᵢᵀ

        All operations are matrix computations, no autograd needed.
        """
        batch_size, window_len, dim = keys.shape
        # Do update math in float32 for stability, then cast back
        keys_f = keys.float()
        values_f = values.float()
        alpha = torch.sigmoid(self.alpha.float())  # Bound to (0, 1)

        # Default gamma = 1 if not provided
        if gammas is None:
            gammas_f = torch.ones(batch_size, window_len, 1, device=keys.device, dtype=torch.float32)
        else:
            gammas_f = gammas.float()

        # Compute ∑γᵢkᵢkᵢᵀ: outer product sum weighted by gamma
        # keys: (B, W, D), gammas: (B, W, 1)
        # k_weighted: (B, W, D) with gamma scaling
        k_weighted = keys_f * gammas_f  # (B, W, D)

        # ∑γᵢkᵢkᵢᵀ = k_weightedᵀ @ k = (D, B*W) @ (B*W, D) = (D, D)
        k_flat = keys_f.reshape(-1, dim)  # (B*W, D)
        k_weighted_flat = k_weighted.reshape(-1, dim)  # (B*W, D)
        kk_sum = torch.matmul(k_weighted_flat.T, k_flat) / (batch_size * window_len)  # (D, D)

        # Compute ∑γᵢvᵢkᵢᵀ
        # values: (B, W, D_out), keys: (B, W, D_in)
        v_weighted = values_f * gammas_f  # (B, W, D_out)
        v_flat = v_weighted.reshape(-1, values_f.shape[-1])  # (B*W, D_out)
        vk_sum = torch.matmul(v_flat.T, k_flat) / (batch_size * window_len)  # (D_out, D_in)

        # Update: ℳₜ = (αI - kk_sum) @ ℳₜ₋₁ + vk_sum
        # For stability, use: ℳₜ = α*ℳₜ₋₁ - kk_sum @ ℳₜ₋₁ + vk_sum
        D_out, D_in = self.M.shape
        M_f = self.M.float()
        new_M_f = alpha * M_f - torch.matmul(kk_sum[:D_out, :D_in], M_f) + vk_sum
        self.M.copy_(new_M_f.to(dtype=self.M.dtype))

    def _forward_deep(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        gammas: Optional[torch.Tensor],
        info: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Deep Memory with gradient-based optimization.
        Uses torch.no_grad() to isolate inner optimization from outer backward.
        """
        batch_size, seq_len, dim = keys.shape
        outputs = []

        for start in range(0, seq_len, self.window_size):
            end = min(start + self.window_size, seq_len)

            window_keys = keys[:, start:end]
            window_values = values[:, start:end]
            window_queries = queries[:, start:end]
            window_gammas = gammas[:, start:end] if gammas is not None else None

            # Optimize memory on this window (isolated from outer backward)
            if self.training:
                self._optimize_window_isolated(
                    window_keys.detach(),
                    window_values.detach(),
                    window_gammas.detach() if window_gammas is not None else None
                )
                info["optimization_steps_run"] += self.optimization_steps
                info["windows_processed"] += 1

            # Retrieve from memory (this is part of the forward graph)
            window_output = self.memory(window_queries)
            outputs.append(window_output)

        memory_output = torch.cat(outputs, dim=1)
        return memory_output, info

    def _optimize_window_isolated(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        gammas: Optional[torch.Tensor] = None,
    ):
        """
        Window optimization ISOLATED from outer backward pass.

        Paper: "all other parameters from the model are considered hyperparameters
        and are fixed and not optimized" during inner loop.

        Uses torch.no_grad() + manual param updates to prevent affecting outer autograd.
        """
        # Temporarily enable gradients only for memory parameters
        for _ in range(self.optimization_steps):
            # Forward pass with local gradient computation
            with torch.enable_grad():
                # Create fresh computation graph for this optimization step
                predictions = self.memory(keys)

                # Compute loss
                if gammas is not None:
                    squared_error = (predictions - values).pow(2)
                    weighted_error = gammas * squared_error
                    loss = weighted_error.mean()
                else:
                    loss = F.mse_loss(predictions, values)

                # Compute gradients ONLY for memory parameters
                grads = torch.autograd.grad(
                    loss,
                    list(self.memory.parameters()),
                    create_graph=False,  # Don't create graph for grad computation
                    retain_graph=False,  # Don't retain - fresh graph each step
                )

            # Update parameters manually with no_grad (isolated from outer backward)
            with torch.no_grad():
                if self.use_muon and self.optimizer is not None:
                    for param, grad in zip(self.memory.parameters(), grads):
                        param.grad = grad
                    self.optimizer.step()
                    for param in self.memory.parameters():
                        param.grad = None
                else:
                    # Simple gradient descent
                    for param, grad in zip(self.memory.parameters(), grads):
                        param.add_(grad, alpha=-self.learning_rate)


class ATLASMemory(nn.Module):
    """
    ATLAS Neural Memory Module.

    세 가지 핵심 요소를 결합:
    1. Polynomial Kernel: 메모리 용량 확장
    2. Deep MLP Memory: 깊은 메모리 구조
    3. Omega Rule: 윈도우 기반 최적화

    데이터 흐름:
    Input → Kernel Expansion → Deep MLP Memory → Omega Optimization → Output
    """
    def __init__(
        self,
        hidden_size: int,
        memory_dim: int = 512,
        num_heads: int = 8,
        window_size: int = 64,
        optimization_steps: int = 3,
        optimization_rule: OptimizationRule = OptimizationRule.OMEGA,
        kernel_type: KernelType = KernelType.POLYNOMIAL,
        polynomial_degree: int = 2,
        kernel_gamma: float = 1.0,
        memory_layers: int = 2,
        memory_expansion: int = 4,
        memory_activation: str = "gelu",
        use_muon: bool = True,
        muon_momentum: float = 0.95,
        muon_beta: float = 0.95,
        learning_rate: float = 0.01,
        use_gate: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        memory_decay: float = 0.9999,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.layer_idx = layer_idx
        self.optimization_rule = optimization_rule

        # Input projections
        self.q_proj = nn.Linear(hidden_size, memory_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(hidden_size, memory_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(hidden_size, memory_dim, bias=False, **factory_kwargs)
        self.o_proj = nn.Linear(memory_dim, hidden_size, bias=False, **factory_kwargs)

        # Gate projection
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, memory_dim, bias=False, **factory_kwargs)
        
        # Gamma projection for Omega Rule Gating
        self.gamma_proj = nn.Linear(hidden_size, 1, bias=True, **factory_kwargs)

        # Polynomial kernel (if enabled)
        if kernel_type == KernelType.POLYNOMIAL:
            self.kernel = PolynomialKernel(
                input_dim=self.head_dim,
                degree=polynomial_degree,
                output_dim=self.head_dim,  # Project back to original dim
                **factory_kwargs,
            )
        else:
            self.kernel = None

        # Deep MLP Memory (per head)
        self.memories = nn.ModuleList([
            DeepMLPMemory(
                input_dim=self.head_dim,
                output_dim=self.head_dim,
                num_layers=memory_layers,
                expansion=memory_expansion,
                activation=memory_activation,
                norm_eps=norm_eps,
                **factory_kwargs,
            )
            for _ in range(num_heads)
        ])

        # Omega Rule optimizer (per head)
        if optimization_rule == OptimizationRule.OMEGA:
            self.omega_rules = nn.ModuleList([
                OmegaRule(
                    memory=memory,
                    window_size=window_size,
                    optimization_steps=optimization_steps,
                    learning_rate=learning_rate,
                    use_muon=use_muon,
                    muon_momentum=muon_momentum,
                    muon_beta=muon_beta,
                )
                for memory in self.memories
            ])
        else:
            self.omega_rules = None

        # Normalization and dropout
        self.norm = RMSNorm(size=hidden_size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

        # Memory decay for long-term forgetting
        self.memory_decay = memory_decay

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
        ATLAS Memory forward pass.

        Args:
            hidden_states: (B, L, D) 입력 텐서
            attention_mask: 어텐션 마스크 (사용 안 함, 호환성용)
            past_key_values: 이전 키/값 캐시
            use_cache: 캐시 반환 여부

        Returns:
            output: (B, L, D) 출력 텐서
            cache: 새 캐시 (use_cache=True일 때)
        """
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Normalize input
        hidden_states = self.norm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Compute gammas for gating
        # Sigmoid to ensure [0, 1] range
        gammas = torch.sigmoid(self.gamma_proj(hidden_states)) # (B, L, 1)

        # Split into heads: (B, L, H, D/H)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Broadcast gammas to heads: (B, L, 1) -> (B, H, L, 1)
        gammas = rearrange(gammas, 'b l 1 -> b 1 l 1').expand(-1, self.num_heads, -1, -1)

        # Apply polynomial kernel (if enabled)
        if self.kernel is not None:
            # Reshape for kernel: (B*H, L, D/H)
            k_flat = rearrange(k, 'b h l d -> (b h) l d')
            k_expanded = self.kernel(k_flat)
            k = rearrange(k_expanded, '(b h) l d -> b h l d', b=batch_size, h=self.num_heads)

            q_flat = rearrange(q, 'b h l d -> (b h) l d')
            q_expanded = self.kernel(q_flat)
            q = rearrange(q_expanded, '(b h) l d -> b h l d', b=batch_size, h=self.num_heads)

        # Process each head with its memory
        outputs = []
        info = {"heads": []}

        for h in range(self.num_heads):
            q_h = q[:, h]  # (B, L, D/H)
            k_h = k[:, h]
            v_h = v[:, h]
            gamma_h = gammas[:, h] # (B, L, 1)

            if self.optimization_rule == OptimizationRule.OMEGA:
                head_out, head_info = self.omega_rules[h](k_h, v_h, q_h, gamma_h)
                info["heads"].append(head_info)
            else:
                # Fallback
                head_out = torch.zeros_like(q_h)
            
            outputs.append(head_out)

        # Merge heads: (B, H, L, D/H) -> (B, L, H*D/H)
        output = torch.stack(outputs, dim=1)  # (B, H, L, D/H)
        output = rearrange(output, 'b h l d -> b l (h d)')

        # Apply gate (if enabled)
        if self.use_gate:
            g = self.g_proj(residual if self.use_residual else hidden_states)
            output = output * F.silu(g)

        # Project output
        output = self.o_proj(output)
        output = self.dropout(output)

        # Residual connection
        if self.use_residual:
            output = output + residual

        # Cache handling
        new_cache = None
        if use_cache:
            new_cache = {"layer_idx": self.layer_idx}

        return output, new_cache

    def reset_memory(self):
        """메모리 초기화"""
        # Reset deep memory parameters (used in deep-memory mode).
        for memory in self.memories:
            memory._reset_parameters()

        if self.omega_rules is not None:
            for rule in self.omega_rules:
                # Linear-memory mode stores its state in OmegaRule.M (buffer), so we
                # must clear it explicitly or it will persist across batches.
                if getattr(rule, "use_linear_memory", False) and hasattr(rule, "M"):
                    with torch.no_grad():
                        rule.M.zero_()
                    # Drop snapshots so they get re-created with correct shapes/devices.
                    if hasattr(rule, "_M_snapshots"):
                        rule._M_snapshots = []

                if rule.optimizer is not None:
                    rule.optimizer.reset()


class ATLASBlock(TransformerBlockBase):
    """
    ATLAS Memory Block.

    구조:
    - Pre-norm with RMSNorm
    - ATLAS Memory layer
    - Post-residual connection
    - MLP (optional)
    """
    def __init__(
        self,
        config: ATLASConfig,
        layer_idx: int,
        n_layers: int,
        layer_norm: LayerNormConfig,
        init_device: str = "cpu",
    ):
        super().__init__(n_layers=n_layers)
        self.config = config
        self.layer_idx = layer_idx

        # ATLAS Memory layer
        self.attn = config.build(layer_idx=layer_idx)

        # MLP
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
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        # ATLAS Memory (includes pre-norm)
        x_memory, _ = self.attn(x, past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        x = x_memory

        # MLP with pre-norm
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def apply_tp(self, tp_mesh, *, input_layout, float8_enabled=False):
        raise NotImplementedError("TP not implemented for ATLASBlock")

    def apply_cp(self, cp_mesh, load_balancer, head_stride=1):
        raise NotImplementedError("CP not implemented for ATLASBlock")

    def apply_fsdp(self, dp_mesh=None, prefetch_factor=0, wrapping_strategy=None, **fsdp_kwargs):
        from torch.distributed.fsdp import fully_shard
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

class AtlasBlockMoE(ATLASBlock):
    """
    AtlasBlock with MoE support.
    """
    def __init__(
        self,
        config: ATLASConfig,
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
