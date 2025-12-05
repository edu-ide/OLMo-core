"""
Block Diffusion: AR-Diffusion Hybrid for Language Models

BD3-LM (Block Discrete Denoising Diffusion Language Model) + Fast-dLLM 패턴을 통합합니다.

Based on:
- BD3-LM: arXiv:2503.09573 (ICLR 2025 Oral)
- Fast-dLLM: arXiv:2505.22618 (NVIDIA)
- D2F: arXiv:2508.09192 (Discrete Diffusion Forcing)
- LLaDA: arXiv:2502.09992

핵심 아이디어:
1. Block 내부: Bidirectional attention + Diffusion denoising (병렬 생성)
2. Block 간: Causal AR + KV Cache (효율적 연결)
3. replace_position: 특정 위치의 KV cache만 업데이트 (Fast-dLLM)

이점:
- 2.5x faster than AR (LLaMA3, Qwen2.5) on GSM8K
- 50x faster than vanilla dLLMs (LLaDA, Dream)
- Global coherence (bidirectional within block)
- KV cache 활용 (AR between blocks)
"""

import abc
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from olmo_core.config import Config

log = logging.getLogger(__name__)

__all__ = [
    # Config
    "BlockDiffusionConfig",
    "NoiseScheduleType",
    # Noise Schedules (from BD3-LM)
    "NoiseSchedule",
    "LogLinearNoise",
    "CosineNoise",
    "ExpNoise",
    "get_noise_schedule",
    # Core modules
    "MaskDiffusion",
    "BlockDiffusionDecoder",
    "BlockDiffAttentionMask",
    "ReplacePositionKVCache",
    # Integration
    "BlockDiffusionWrapper",
    "wrap_model_with_block_diffusion",
    # Utilities
    "create_block_diffusion_model",
]


# =============================================================================
# Noise Schedules (Based on BD3-LM noise_schedule.py)
# =============================================================================


class NoiseScheduleType(str, Enum):
    """Diffusion noise schedule types."""

    LOGLINEAR = "loglinear"
    """Log-linear schedule (BD3-LM default)"""

    COSINE = "cosine"
    """Cosine schedule"""

    SQUARE = "square"
    """Square schedule (exp=2)"""

    SQRT = "sqrt"
    """Square root schedule (exp=0.5)"""


class NoiseSchedule(abc.ABC, nn.Module):
    """
    Abstract base class for noise schedules.

    Based on BD3-LM's noise schedule implementation.
    Returns (loss_scaling, move_chance) at timestep t.
    """

    @abc.abstractmethod
    def compute_loss_scaling_and_move_chance(
        self, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute loss scaling and move chance at timestep t.

        Args:
            t: Timestep in [0, 1], shape [B] or [B, num_blocks]

        Returns:
            loss_scaling: Loss scaling factor
            move_chance: Probability of masking (0 = original, 1 = fully masked)
        """
        raise NotImplementedError

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.compute_loss_scaling_and_move_chance(t)


class LogLinearNoise(NoiseSchedule):
    """
    Log Linear noise schedule (BD3-LM default).

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t varies from 0 to 1.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t: Tensor) -> Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: Tensor) -> Tensor:
        return -torch.log1p(-(1 - self.eps) * t)

    def compute_loss_scaling_and_move_chance(
        self, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        loss_scaling = -1 / t.clamp(min=self.eps)
        move_chance = t
        return loss_scaling, move_chance


class CosineNoise(NoiseSchedule):
    """Cosine noise schedule."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def compute_loss_scaling_and_move_chance(
        self, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        cos = -(1 - self.eps) * torch.cos(t * math.pi / 2)
        sin = -(1 - self.eps) * torch.sin(t * math.pi / 2)
        move_chance = cos + 1
        loss_scaling = sin / (move_chance + self.eps) * math.pi / 2
        return loss_scaling, move_chance


class ExpNoise(NoiseSchedule):
    """Exponential noise schedule (square, sqrt, etc.)."""

    def __init__(self, exp: float = 2.0, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.exp = exp

    def compute_loss_scaling_and_move_chance(
        self, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        move_chance = torch.pow(t, self.exp)
        move_chance = torch.clamp(move_chance, min=self.eps)
        loss_scaling = -(self.exp * torch.pow(t, self.exp - 1)) / move_chance
        return loss_scaling, move_chance


def get_noise_schedule(
    schedule_type: NoiseScheduleType,
    eps: float = 1e-3,
) -> NoiseSchedule:
    """Get noise schedule by type."""
    if schedule_type == NoiseScheduleType.LOGLINEAR:
        return LogLinearNoise(eps)
    elif schedule_type == NoiseScheduleType.COSINE:
        return CosineNoise(eps)
    elif schedule_type == NoiseScheduleType.SQUARE:
        return ExpNoise(exp=2.0, eps=eps)
    elif schedule_type == NoiseScheduleType.SQRT:
        return ExpNoise(exp=0.5, eps=eps)
    else:
        raise ValueError(f"Unknown noise schedule: {schedule_type}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BlockDiffusionConfig(Config):
    """
    Block Diffusion 설정.

    BD3-LM과 Fast-dLLM의 핵심 설정을 통합합니다.
    """

    # Block 설정
    block_size: int = 64
    """각 블록의 토큰 수 (권장: 32-128)"""

    max_blocks: int = 32
    """최대 블록 수"""

    # Diffusion 설정
    num_diffusion_steps: int = 8
    """블록 내 diffusion steps (적을수록 빠름, 권장: 4-16)"""

    noise_schedule: NoiseScheduleType = NoiseScheduleType.LOGLINEAR
    """Noise schedule (BD3-LM default: loglinear)"""

    mask_token_id: int = -1
    """Mask token ID (vocabulary size로 자동 설정 가능)"""

    noise_eps: float = 1e-3
    """Noise schedule epsilon"""

    # Sampling 설정
    sampling_eps: float = 1e-3
    """Sampling epsilon (최소 noise level)"""

    nucleus_p: float = 1.0
    """Nucleus sampling p (1.0 = no nucleus)"""

    use_first_hitting: bool = False
    """Use first-hitting sampler (BD3-LM)"""

    # KV Cache 설정 (Fast-dLLM)
    use_kv_cache: bool = True
    """Block 간 KV cache 사용"""

    use_replace_position: bool = True
    """replace_position 패턴 사용 (Fast-dLLM)"""

    # Attention 설정
    attn_backend: str = "sdpa"
    """Attention backend: 'sdpa', 'flex', 'flash'"""

    # Training
    antithetic_sampling: bool = True
    """Use antithetic sampling for variance reduction"""

    training_resample: bool = True
    """Resample if mask ratio outside bounds"""

    # Cross attention (for x0 conditioning)
    use_cross_attn: bool = False
    """Use x0 as cross-attention context (BD3-LM)"""


# =============================================================================
# Block Diffusion Attention Mask (Based on BD3-LM)
# =============================================================================


class BlockDiffAttentionMask(nn.Module):
    """
    Block Diffusion Attention Mask.

    Based on BD3-LM's block_diff_mask function:
    - M_BD (Block Diagonal): Self-attention within noised blocks
    - M_OBC (Offset Block Causal): Cross-attention for conditional context
    - M_BC (Block Causal): Attention to update x0

    For FlexAttention, returns a mask function.
    For SDPA, returns a boolean mask tensor.
    """

    def __init__(
        self,
        seq_len: int,
        block_size: int,
        attn_backend: str = "sdpa",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.block_size = block_size
        self.attn_backend = attn_backend

        if attn_backend == "sdpa":
            # Pre-compute SDPA mask
            mask = self._create_sdpa_mask(seq_len, block_size, device)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def _create_sdpa_mask(
        self,
        seq_len: int,
        block_size: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Create SDPA-compatible attention mask."""
        n = seq_len  # Length of xt (noised)

        # Create indices
        q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = kv_idx // block_size

        # Block Diagonal Mask: same block = bidirectional
        block_diagonal = block_q == block_kv

        # Block Causal Mask: can attend to previous blocks
        block_causal = block_q >= block_kv

        # Combined: within block = bidirectional, between blocks = causal
        mask = block_diagonal | block_causal

        return mask.float()

    @staticmethod
    def flex_mask_fn(
        b: int,
        h: int,
        q_idx: int,
        kv_idx: int,
        block_size: int,
        n: int,
    ) -> bool:
        """
        FlexAttention mask function for BD3-LM style training.

        When using cross-attention with x0:
        - Sequence is [xt, x0] where xt is noised, x0 is clean
        - M_BD: Self-attention within noised blocks (xt)
        - M_OBC: xt can attend to previous blocks of x0
        - M_BC: x0 uses block-causal attention
        """
        # Indicate whether token belongs to xt or x0
        x0_flag_q = q_idx >= n
        x0_flag_kv = kv_idx >= n

        # Compute block indices
        if x0_flag_q:
            block_q = (q_idx - n) // block_size
        else:
            block_q = q_idx // block_size

        if x0_flag_kv:
            block_kv = (kv_idx - n) // block_size
        else:
            block_kv = kv_idx // block_size

        # 1. Block Diagonal Mask (M_BD)
        block_diagonal = (block_q == block_kv) and (x0_flag_q == x0_flag_kv)

        # 2. Offset Block-Causal Mask (M_OBC)
        offset_block_causal = (
            (block_q > block_kv)
            and x0_flag_kv
            and (not x0_flag_q)
        )

        # 3. Block-Causal Mask (M_BC)
        block_causal = (block_q >= block_kv) and x0_flag_kv and x0_flag_q

        return block_diagonal or offset_block_causal or block_causal

    def get_mask(
        self,
        seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Tensor]:
        """Get attention mask for SDPA."""
        if self.mask is not None:
            if seq_len is not None and seq_len != self.seq_len:
                # Recompute for different length
                return self._create_sdpa_mask(seq_len, self.block_size, device)
            return self.mask
        return None


# =============================================================================
# Replace Position KV Cache (Based on Fast-dLLM)
# =============================================================================


class ReplacePositionKVCache:
    """
    KV Cache with replace_position support.

    Based on Fast-dLLM's replace_position mechanism:
    - Instead of concatenating new K/V, update specific positions
    - Enables efficient block-wise generation without full recomputation

    Usage:
        cache = ReplacePositionKVCache(num_layers, ...)

        # First forward (initialize)
        outputs = model(x, past_key_values=None, use_cache=True)
        cache.initialize(outputs.past_key_values)

        # Update specific positions
        replace_mask = torch.zeros(B, seq_len, dtype=torch.bool)
        replace_mask[:, block_start:block_end] = True
        cache.update(new_kv, replace_mask)
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Initialize empty cache
        self.k_cache: List[Optional[Tensor]] = [None] * num_layers
        self.v_cache: List[Optional[Tensor]] = [None] * num_layers
        self.seq_len = 0

    def initialize(
        self,
        past_key_values: List[Tuple[Tensor, Tensor]],
    ) -> None:
        """Initialize cache from model outputs."""
        for layer_idx, (k, v) in enumerate(past_key_values):
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        self.seq_len = past_key_values[0][0].size(2)

    def update(
        self,
        layer_idx: int,
        new_k: Tensor,
        new_v: Tensor,
        replace_position: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update KV cache at specific positions.

        Args:
            layer_idx: Layer index
            new_k: New keys [B, n_kv_heads, update_len, head_dim]
            new_v: New values [B, n_kv_heads, update_len, head_dim]
            replace_position: [B, seq_len] boolean mask indicating positions to replace

        Returns:
            Updated (K, V) for this layer
        """
        past_k = self.k_cache[layer_idx]
        past_v = self.v_cache[layer_idx]

        if past_k is None:
            # First time, just store
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
            return new_k, new_v

        B = replace_position.shape[0]

        # Replace positions (based on Fast-dLLM pattern)
        for batch_idx in range(B):
            # Get indices for this batch
            replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
            if len(replace_indices) > 0:
                num_replace = min(len(replace_indices), new_k.size(2))
                past_k[batch_idx, :, replace_indices[:num_replace]] = new_k[
                    batch_idx, :, :num_replace
                ]
                past_v[batch_idx, :, replace_indices[:num_replace]] = new_v[
                    batch_idx, :, :num_replace
                ]

        return past_k, past_v

    def get_cache(
        self, layer_idx: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Get cached K/V for a layer."""
        if self.k_cache[layer_idx] is None:
            return None
        return (self.k_cache[layer_idx], self.v_cache[layer_idx])

    def reset(self) -> None:
        """Reset the cache."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self.seq_len = 0


# =============================================================================
# Mask Diffusion Process
# =============================================================================


class MaskDiffusion(nn.Module):
    """
    Masked Diffusion Process.

    Based on BD3-LM's diffusion implementation with proper noise scheduling.
    Supports antithetic sampling for variance reduction.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        noise_schedule: NoiseSchedule,
        block_size: int = 64,
        antithetic_sampling: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.noise = noise_schedule
        self.block_size = block_size
        self.antithetic_sampling = antithetic_sampling
        self.neg_infinity = -1000000.0

    def sample_t(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
    ) -> Tensor:
        """
        Sample timesteps for training.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device
            sampling_eps_min: Minimum sampling epsilon
            sampling_eps_max: Maximum sampling epsilon

        Returns:
            t: Sampled timesteps [B, num_blocks]
        """
        num_blocks = seq_len // self.block_size
        _eps_b = torch.rand((batch_size, num_blocks), device=device)

        # Antithetic sampling for variance reduction
        if self.antithetic_sampling:
            offset_b = torch.arange(
                batch_size * num_blocks, device=device
            ) / (batch_size * num_blocks)
            offset_b = offset_b.view(batch_size, num_blocks)
            _eps_b = (_eps_b / (batch_size * num_blocks) + offset_b) % 1

        t = _eps_b * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
        return t

    def q_xt(
        self,
        x: Tensor,
        t: Tensor,
        block_size: Optional[int] = None,
    ) -> Tensor:
        """
        Forward diffusion: sample x_t given x_0 and t.

        Args:
            x: Original tokens [B, S]
            t: Timesteps [B] or [B, num_blocks]

        Returns:
            x_t: Noised tokens [B, S]
        """
        if block_size is None:
            block_size = self.block_size

        # Get move_chance from noise schedule
        _, move_chance = self.noise(t)

        # Expand move_chance to token level
        if move_chance.dim() == 1:
            move_chance = move_chance.unsqueeze(-1)
        if move_chance.size(-1) != x.size(-1):
            # Per-block masking: expand to per-token
            move_chance = move_chance.repeat_interleave(block_size, dim=-1)
            if move_chance.size(-1) > x.size(-1):
                move_chance = move_chance[:, :x.size(-1)]

        # Sample mask
        mask = torch.rand_like(x.float()) < move_chance
        x_t = torch.where(mask, self.mask_token_id, x)

        return x_t

    def ddpm_update(
        self,
        x: Tensor,
        logits: Tensor,
        t: Tensor,
        dt: float,
        nucleus_p: float = 1.0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        DDPM-style reverse step.

        Based on BD3-LM's _ddpm_caching_update.

        Args:
            x: Current noised tokens [B, S]
            logits: Model predictions [B, S, V]
            t: Current timestep
            dt: Time step size

        Returns:
            x_new: Updated tokens
            p_x0: Predicted x0 distribution (for caching)
        """
        _, move_chance_t = self.noise(t)
        _, move_chance_s = self.noise(t - dt)

        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        mask_prob = move_chance_s / move_chance_t

        # Get p(x0 | xt)
        logits = logits.to(torch.float64)
        logits[:, :, self.mask_token_id] = self.neg_infinity
        p_x0 = F.softmax(logits, dim=-1)

        # Nucleus sampling
        if nucleus_p < 1.0:
            p_x0 = self._nucleus_sample(p_x0, nucleus_p)

        # Sample new tokens
        q_xs = p_x0 * (1 - mask_prob)
        q_xs[:, :, self.mask_token_id] = mask_prob.squeeze(-1)

        # Gumbel-max trick for sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(q_xs) + 1e-10) + 1e-10)
        x_new = (q_xs / (gumbel_noise + 1e-10)).argmax(dim=-1)

        # Keep already unmasked tokens
        is_masked = x == self.mask_token_id
        x_new = torch.where(is_masked, x_new, x)

        return x_new, p_x0

    def _nucleus_sample(self, probs: Tensor, p: float) -> Tensor:
        """Apply nucleus sampling."""
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= p
        nucleus_mask[..., 0] = True  # Keep at least one

        sorted_probs = sorted_probs * nucleus_mask
        probs_new = probs.clone()
        probs_new.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        probs_new = probs_new / probs_new.sum(-1, keepdim=True)

        return probs_new


# =============================================================================
# Block Diffusion Decoder
# =============================================================================


class BlockDiffusionDecoder(nn.Module):
    """
    Block Diffusion Decoder.

    Block 단위로 AR 진행하면서, 각 block 내부는
    diffusion으로 병렬 생성합니다.

    Based on BD3-LM's _semi_ar_sampler.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        config: BlockDiffusionConfig,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.config = config

        # Mask token
        mask_id = config.mask_token_id if config.mask_token_id >= 0 else vocab_size
        self.mask_token_id = mask_id

        # Noise schedule
        self.noise = get_noise_schedule(
            config.noise_schedule,
            eps=config.noise_eps,
        )

        # Diffusion process
        self.diffusion = MaskDiffusion(
            vocab_size=vocab_size,
            mask_token_id=mask_id,
            noise_schedule=self.noise,
            block_size=config.block_size,
            antithetic_sampling=config.antithetic_sampling,
        )

        # Attention mask
        self.attn_mask: Optional[BlockDiffAttentionMask] = None

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _get_sigma_from_p(self, p: Tensor) -> Tensor:
        """Convert move_chance to sigma for model conditioning."""
        if isinstance(self.noise, LogLinearNoise):
            return torch.minimum(
                -torch.log(1 - p),
                torch.tensor(self.noise.sigma_max, device=p.device),
            )
        return p  # For other schedules, use p directly

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: Tensor,
        max_new_tokens: int,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Semi-AR generation: block-wise AR with diffusion within blocks.

        Based on BD3-LM's _semi_ar_sampler.

        Args:
            model: Transformer model
            input_ids: [B, prefix_len] input tokens
            max_new_tokens: Number of new tokens to generate
            num_steps: Diffusion steps per block (default from config)

        Returns:
            generated: [B, prefix_len + new_tokens] full sequence
        """
        B = input_ids.size(0)
        device = input_ids.device
        block_size = self.config.block_size

        if num_steps is None:
            num_steps = self.config.num_diffusion_steps

        num_blocks = (max_new_tokens + block_size - 1) // block_size

        # Initialize with input
        x = input_ids.clone()

        # KV cache
        kv_cache = None
        if self.config.use_kv_cache:
            # Will be initialized on first forward
            pass

        for block_idx in range(num_blocks):
            # Append masked block
            new_block = torch.full(
                (B, block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            x = torch.cat([x, new_block], dim=1)

            # Block boundaries
            block_start = input_ids.size(1) + block_idx * block_size
            block_end = block_start + block_size

            # Diffusion loop within block
            dt = 1.0 / num_steps
            t = 1.0

            for step in range(num_steps):
                # Check if all unmasked
                if self.mask_token_id not in x[:, block_start:block_end]:
                    break

                # Prepare input for model
                if self.config.use_kv_cache and kv_cache is not None:
                    # Only forward the current block
                    model_input = x[:, block_start:block_end]

                    if self.config.use_replace_position:
                        # Create replace_position mask
                        replace_position = torch.zeros(
                            B, x.size(1), dtype=torch.bool, device=device
                        )
                        replace_position[:, block_start:block_end] = True

                        outputs = model(
                            model_input,
                            past_key_values=kv_cache,
                            use_cache=True,
                            replace_position=replace_position,
                        )
                    else:
                        outputs = model(
                            model_input,
                            past_key_values=kv_cache,
                            use_cache=True,
                        )
                else:
                    # Full forward
                    outputs = model(x, use_cache=self.config.use_kv_cache)

                # Get logits for current block
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    logits = self.output_proj(hidden)

                # Extract block logits
                if self.config.use_kv_cache and kv_cache is not None:
                    block_logits = logits
                else:
                    block_logits = logits[:, block_start:block_end]

                # Update cache
                if self.config.use_kv_cache:
                    if hasattr(outputs, 'past_key_values'):
                        kv_cache = outputs.past_key_values

                # DDPM update
                t_tensor = torch.full((B,), t, device=device)
                x_block = x[:, block_start:block_end]

                x_new, _ = self.diffusion.ddpm_update(
                    x_block,
                    block_logits,
                    t_tensor,
                    dt,
                    nucleus_p=self.config.nucleus_p,
                )

                x[:, block_start:block_end] = x_new
                t = t - dt

            # Final denoising: unmask any remaining
            if self.mask_token_id in x[:, block_start:block_end]:
                # One more forward for final predictions
                if self.config.use_kv_cache and kv_cache is not None:
                    model_input = x[:, block_start:block_end]
                    outputs = model(
                        model_input,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
                    logits = outputs.logits if hasattr(outputs, 'logits') else self.output_proj(outputs[0])
                else:
                    outputs = model(x)
                    logits = outputs.logits if hasattr(outputs, 'logits') else self.output_proj(outputs[0])
                    logits = logits[:, block_start:block_end]

                # Argmax for remaining masked
                predictions = logits.argmax(dim=-1)
                is_masked = x[:, block_start:block_end] == self.mask_token_id
                x[:, block_start:block_end] = torch.where(
                    is_masked, predictions, x[:, block_start:block_end]
                )

            # Store KV cache for completed block
            if self.config.use_kv_cache and kv_cache is not None:
                # Cache is automatically updated by the model
                pass

        # Truncate to exact length
        target_len = input_ids.size(1) + max_new_tokens
        if x.size(1) > target_len:
            x = x[:, :target_len]

        return x


# =============================================================================
# Block Diffusion Wrapper
# =============================================================================


class BlockDiffusionWrapper(nn.Module):
    """
    기존 모델에 Block Diffusion을 추가하는 wrapper.

    AR 모델을 AR-Diffusion 하이브리드로 변환합니다.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: BlockDiffusionConfig,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # Get model dimensions
        if hasattr(base_model, 'd_model'):
            d_model = base_model.d_model
        elif hasattr(base_model, 'config'):
            d_model = getattr(
                base_model.config, 'hidden_size',
                getattr(base_model.config, 'd_model', 4096)
            )
        else:
            d_model = 4096

        if vocab_size is None:
            if hasattr(base_model, 'vocab_size'):
                vocab_size = base_model.vocab_size
            elif hasattr(base_model, 'config'):
                vocab_size = base_model.config.vocab_size
            else:
                vocab_size = 32000

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Block diffusion decoder
        self.block_decoder = BlockDiffusionDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            config=config,
        )

        # Mask token embedding (if needed)
        mask_id = config.mask_token_id if config.mask_token_id >= 0 else vocab_size
        self.mask_token_id = mask_id

        if mask_id == vocab_size:
            # Add mask token to embedding
            self.mask_embedding = nn.Parameter(
                torch.randn(1, 1, d_model) * 0.02
            )
        else:
            self.mask_embedding = None

        # Noise schedule for training
        self.noise = get_noise_schedule(
            config.noise_schedule,
            eps=config.noise_eps,
        )

    def __getattr__(self, name: str):
        """base_model의 속성에 접근 가능하도록 위임"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Training forward pass.

        Diffusion training: random masking + denoising loss.
        """
        if inputs_embeds is None:
            if hasattr(self.base_model, 'get_input_embeddings'):
                embed = self.base_model.get_input_embeddings()
            else:
                embed = self.base_model.embed_tokens
            inputs_embeds = embed(input_ids)

        # Random masking for training
        if self.training and input_ids is not None:
            B, S = input_ids.shape

            # Sample timesteps
            if t is None:
                t = self.block_decoder.diffusion.sample_t(
                    B, S, input_ids.device,
                    sampling_eps_min=self.config.sampling_eps,
                    sampling_eps_max=1.0,
                )

            # Get masked tokens
            x_t = self.block_decoder.diffusion.q_xt(input_ids, t)

            # Replace embeddings for masked tokens
            if self.mask_embedding is not None:
                mask_embed = self.mask_embedding.expand(B, S, -1)
            else:
                mask_embed = embed(
                    torch.full((B, S), self.mask_token_id, device=input_ids.device)
                )

            is_masked = (x_t == self.mask_token_id).unsqueeze(-1)
            inputs_embeds = torch.where(is_masked, mask_embed, inputs_embeds)

        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Compute loss
        if labels is not None:
            # outputs가 dict, namedtuple, 또는 tuple 형태일 수 있음
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs[0]

            # Get loss scaling from noise schedule
            t_tensor = t if t is not None else torch.ones(1, device=logits.device, dtype=logits.dtype)
            loss_scaling, _ = self.noise(t_tensor)
            loss_scaling = loss_scaling.to(logits.device)

            # Cross entropy loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            ).view(logits.size(0), -1)

            # Apply loss scaling (per-block if available)
            if loss_scaling.dim() > 1:
                loss_scaling = loss_scaling.repeat_interleave(
                    self.config.block_size, dim=-1
                )
                if loss_scaling.size(-1) > loss_per_token.size(-1):
                    loss_scaling = loss_scaling[:, :loss_per_token.size(-1)]

            scaled_loss = loss_scaling.abs() * loss_per_token

            # Only compute loss on masked positions (if available)
            if attention_mask is not None:
                loss = (scaled_loss * attention_mask).sum() / attention_mask.sum()
            else:
                loss = scaled_loss.mean()
        else:
            loss = None

        # logits 추출
        if hasattr(outputs, 'logits'):
            final_logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            final_logits = outputs['logits']
        else:
            final_logits = logits  # 이미 위에서 추출한 logits 사용

        return {
            "loss": loss,
            "logits": final_logits,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Block-wise AR-Diffusion 생성.
        """
        return self.block_decoder.generate(
            self.base_model,
            input_ids,
            max_new_tokens,
            num_steps=num_steps,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def wrap_model_with_block_diffusion(
    model: nn.Module,
    config: Optional[BlockDiffusionConfig] = None,
    vocab_size: Optional[int] = None,
) -> BlockDiffusionWrapper:
    """
    기존 모델에 Block Diffusion을 추가합니다.

    Args:
        model: Base transformer model
        config: Block diffusion configuration
        vocab_size: Vocabulary size

    Returns:
        BlockDiffusionWrapper: Wrapped model

    Example:
        ```python
        from olmo_core.nn.transformer import (
            BlockDiffusionConfig,
            wrap_model_with_block_diffusion,
        )

        config = BlockDiffusionConfig(
            block_size=64,
            num_diffusion_steps=8,
            noise_schedule=NoiseScheduleType.LOGLINEAR,
        )

        model = wrap_model_with_block_diffusion(base_model, config)

        # Generate
        output = model.generate(input_ids, max_new_tokens=256)
        ```
    """
    if config is None:
        config = BlockDiffusionConfig()

    return BlockDiffusionWrapper(model, config, vocab_size)


def create_block_diffusion_model(
    base_model: nn.Module,
    block_size: int = 64,
    num_diffusion_steps: int = 8,
    noise_schedule: NoiseScheduleType = NoiseScheduleType.LOGLINEAR,
    use_kv_cache: bool = True,
    **kwargs,
) -> BlockDiffusionWrapper:
    """
    간편한 Block Diffusion 모델 생성.

    Args:
        base_model: Base transformer model
        block_size: Block size for generation
        num_diffusion_steps: Diffusion steps per block
        noise_schedule: Noise schedule type
        use_kv_cache: Use KV cache for efficiency

    Example:
        ```python
        model = create_block_diffusion_model(
            base_model,
            block_size=64,
            num_diffusion_steps=8,
        )
        ```
    """
    config = BlockDiffusionConfig(
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
        noise_schedule=noise_schedule,
        use_kv_cache=use_kv_cache,
        **kwargs,
    )

    return wrap_model_with_block_diffusion(base_model, config)
