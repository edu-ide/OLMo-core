"""
LaDiR: Latent Diffusion Reasoning Module

VAE + Flow Matching Diffusion + Prophet Early Exit을 통합한
Latent Reasoning 구현입니다.

핵심 아이디어:
- VAE로 reasoning text를 compact latent representation으로 압축
- Flow Matching diffusion으로 latent space에서 reasoning 생성
- Prophet: Confidence gap 기반 조기 종료로 추론 가속

References:
- LaDiR: arXiv:2510.08558 (Latent Diffusion Reasoning)
- Prophet: arXiv:2508.19982 (Early Commit Decoding)

Deprecated (removed):
- CoCoNut: arXiv:2412.06769 - 원본, curriculum 필요
- CODI: EMNLP 2025 - Self-distillation 기반
- PCCoT: EMNLP 2025 - Jacobi iteration 병렬화
- KaVa: October 2025 - KV-cache distillation
- SoftCoT: Frozen LLM + projection
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from olmo_core.config import Config

log = logging.getLogger(__name__)

__all__ = [
    # Config
    "LatentReasoningConfig",
    "LatentReasoningMode",
    "LaDiRConfig",
    "ProphetConfig",
    # LaDiR + Prophet
    "LaDiRVAE",
    "FlowMatchingScheduler",
    "LaDiRDiffusion",
    "ProphetEarlyExit",
    # Integration
    "LatentThinkBlock",
    "LatentReasoningWrapper",
    "wrap_model_with_latent_reasoning",
    # Utilities
    "decode_latent_to_tokens",
    "compute_superposition_entropy",
    "create_latent_reasoning_pipeline",
]


class LatentReasoningMode(str, Enum):
    """Latent reasoning 모드."""

    LADIR = "ladir"
    """LaDiR: VAE + Flow Matching Diffusion + Prophet early exit (recommended)"""


@dataclass
class ProphetConfig(Config):
    """
    Prophet Early Exit 설정.

    Confidence gap (top1 - top2 logits) 기반 조기 종료.
    Phase-aware thresholds로 denoising 진행도에 따라 다른 threshold 적용.
    """

    enabled: bool = True
    """Prophet early exit 활성화"""

    threshold_early: float = 7.5
    """Early phase (0-33%) threshold"""

    threshold_mid: float = 5.0
    """Mid phase (33-67%) threshold"""

    threshold_late: float = 2.5
    """Late phase (67-100%) threshold"""

    check_interval: int = 5
    """Early exit 체크 간격 (steps)"""

    min_steps: int = 10
    """최소 denoising steps (이전에는 early exit 안함)"""


@dataclass
class LaDiRConfig(Config):
    """
    LaDiR (Latent Diffusion Reasoning) 설정.

    VAE로 reasoning을 latent space로 압축하고,
    Flow Matching diffusion으로 생성합니다.
    """

    # VAE 설정
    latent_dim: int = 128
    """VAE latent dimension"""

    num_memory_slots: int = 3
    """Memory slot 개수 (압축된 thought token 수)"""

    compression_rate: int = 512
    """압축 비율 (tokens → latent)"""

    vae_beta: float = 1e-5
    """VAE KL loss weight"""

    # Diffusion 설정
    num_train_timesteps: int = 1000
    """Training timesteps"""

    num_inference_steps: int = 50
    """Inference denoising steps"""

    noise_scale: float = 2.5
    """Initial noise scale"""

    # Flow Matching 설정
    prediction_type: str = "flow"
    """Prediction type: 'flow', 'epsilon', 'v_prediction'"""

    # Prophet 설정
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    """Prophet early exit 설정"""

    # 학습 설정
    freeze_vae: bool = True
    """VAE freeze (diffusion 학습 시)"""

    diffusion_batch_mul: int = 8
    """Diffusion batch multiplier"""


@dataclass
class LatentReasoningConfig(Config):
    """
    LaDiR Latent Reasoning 설정.

    LaDiR: VAE + Flow Matching + Prophet을 통합합니다.
    """

    mode: LatentReasoningMode = LatentReasoningMode.LADIR
    """Latent reasoning 모드 (현재 LADIR만 지원)"""

    # LaDiR 설정
    ladir: LaDiRConfig = field(default_factory=LaDiRConfig)
    """LaDiR (VAE + Flow Matching + Prophet) 설정"""

    # Superposition 분석
    track_superposition: bool = False
    """Superposition entropy 추적"""

    # Interpretability
    enable_decoding: bool = True
    """Latent → token decoding 활성화"""

    decoding_top_k: int = 5
    """Decoding시 top-k tokens"""


# ============================================================================
# LaDiR + Prophet Modules
# ============================================================================

class FlowMatchingScheduler:
    """
    Flow Matching Euler Discrete Scheduler.

    LaDiR에서 사용하는 Flow Matching 기반 noise scheduler.
    Diffusers의 FlowMatchEulerDiscreteScheduler와 호환.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "flow",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timesteps = None
        self.sigmas = None
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """Set inference timesteps."""
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps,
            device=device
        )
        self.timesteps = timesteps.long()

        # Compute sigmas for flow matching
        sigmas = timesteps / self.num_train_timesteps
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        self._step_index = None

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Add noise to samples (forward diffusion)."""
        # Flow matching: x_t = (1 - t) * x_0 + t * noise
        t = timesteps.float() / self.num_train_timesteps
        t = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
        noisy_samples = (1 - t) * original_samples + t * noise
        return noisy_samples

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> "FlowMatchingSchedulerOutput":
        """Perform one denoising step."""
        if self._step_index is None:
            self._step_index = 0

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        # Flow matching step: x_{t-1} = x_t - (sigma - sigma_next) * v
        dt = sigma - sigma_next
        prev_sample = sample - dt * model_output

        self._step_index += 1

        return FlowMatchingSchedulerOutput(prev_sample=prev_sample)


class FlowMatchingSchedulerOutput:
    """Output of FlowMatchingScheduler.step()"""
    def __init__(self, prev_sample: Tensor):
        self.prev_sample = prev_sample


class ProphetEarlyExit(nn.Module):
    """
    Prophet: Training-free Early Exit for Diffusion LMs.

    Confidence gap (top1 - top2 logits) 기반으로 denoising을
    조기 종료하여 추론 속도를 높입니다.

    Phase-aware thresholds:
    - Early (0-33%): 높은 threshold (7.5) - 신중하게 종료
    - Mid (33-67%): 중간 threshold (5.0)
    - Late (67-100%): 낮은 threshold (2.5) - 적극적 종료
    """

    def __init__(self, config: ProphetConfig):
        super().__init__()
        self.config = config
        self.enabled = config.enabled

    def get_threshold(self, progress: float) -> float:
        """Get phase-aware threshold based on denoising progress."""
        if progress < 0.33:
            return self.config.threshold_early
        elif progress < 0.67:
            return self.config.threshold_mid
        else:
            return self.config.threshold_late

    def compute_confidence_gap(self, logits: Tensor) -> Tensor:
        """
        Compute confidence gap: top1_logit - top2_logit.

        Args:
            logits: [B, seq_len, vocab_size]

        Returns:
            gap: [B] average gap across sequence
        """
        top2_vals, _ = torch.topk(logits, k=2, dim=-1)  # [B, seq_len, 2]
        gaps = top2_vals[..., 0] - top2_vals[..., 1]    # [B, seq_len]
        return gaps.mean(dim=-1)  # [B]

    def should_exit(
        self,
        logits: Tensor,
        step: int,
        total_steps: int,
    ) -> Tuple[bool, float]:
        """
        Check if early exit condition is met.

        Args:
            logits: Model output logits
            step: Current denoising step
            total_steps: Total denoising steps

        Returns:
            (should_exit, confidence_gap)
        """
        if not self.enabled:
            return False, 0.0

        if step < self.config.min_steps:
            return False, 0.0

        if step % self.config.check_interval != 0:
            return False, 0.0

        progress = step / total_steps
        threshold = self.get_threshold(progress)

        gap = self.compute_confidence_gap(logits)
        avg_gap = gap.mean().item()

        return avg_gap >= threshold, avg_gap


class LaDiRVAE(nn.Module):
    """
    LaDiR VAE: Text → Latent 압축 및 복원.

    Reasoning text를 compact latent representation으로 압축하고,
    다시 text로 복원합니다.

    구조:
    - Encoder: LLM + memory tokens → mean, log_var
    - Reparameterization: z = mean + eps * std
    - Decoder: z → LLM → text
    """

    def __init__(
        self,
        d_model: int,
        config: LaDiRConfig,
        base_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_memory_slots = config.num_memory_slots

        # Encoder projections
        self.mean_proj = nn.Linear(d_model, config.latent_dim)
        self.logvar_proj = nn.Linear(d_model, config.latent_dim)

        # Decoder projection
        self.decompress = nn.Linear(config.latent_dim, d_model)

        # Memory token embeddings
        self.memory_embed = nn.Embedding(config.num_memory_slots, config.latent_dim)

        # Base model reference (for encoding/decoding)
        self.base_model = base_model

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mean + eps * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode hidden states to latent.

        Args:
            hidden_states: [B, S, d_model] from encoder

        Returns:
            z: [B, num_memory_slots, latent_dim]
            mean: [B, num_memory_slots, latent_dim]
            logvar: [B, num_memory_slots, latent_dim]
        """
        B, S, D = hidden_states.shape

        # Use last num_memory_slots positions as memory
        memory_hidden = hidden_states[:, -self.num_memory_slots:, :]  # [B, M, D]

        mean = self.mean_proj(memory_hidden)      # [B, M, latent_dim]
        logvar = self.logvar_proj(memory_hidden)  # [B, M, latent_dim]

        z = self.reparameterize(mean, logvar)

        return z, mean, logvar

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent to hidden states.

        Args:
            z: [B, num_memory_slots, latent_dim]

        Returns:
            hidden_states: [B, num_memory_slots, d_model]
        """
        return self.decompress(z)

    def compute_kl_loss(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence loss."""
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Dict[str, Tensor]:
        """
        VAE forward pass.

        Returns:
            dict with 'z', 'mean', 'logvar', 'reconstructed', 'kl_loss'
        """
        z, mean, logvar = self.encode(hidden_states)
        reconstructed = self.decode(z)
        kl_loss = self.compute_kl_loss(mean, logvar)

        return {
            'z': z,
            'mean': mean,
            'logvar': logvar,
            'reconstructed': reconstructed,
            'kl_loss': kl_loss,
        }

    def latent_to_logits(
        self,
        z: Tensor,
        model: nn.Module,
    ) -> Tensor:
        """
        Convert latent to logits for Prophet confidence check.

        Args:
            z: [B, num_memory_slots, latent_dim]
            model: LLM for decoding

        Returns:
            logits: [B, seq_len, vocab_size]
        """
        hidden = self.decode(z)  # [B, M, d_model]

        # Forward through model to get logits
        with torch.no_grad():
            outputs = model(inputs_embeds=hidden, output_hidden_states=False)
            if hasattr(outputs, 'logits'):
                return outputs.logits
            return outputs[0]


class LaDiRDiffusion(nn.Module):
    """
    LaDiR Diffusion: Flow Matching 기반 latent denoising.

    VAE로 압축된 latent space에서 Flow Matching diffusion을 수행하고,
    Prophet early exit으로 추론을 가속합니다.
    """

    def __init__(
        self,
        d_model: int,
        config: LaDiRConfig,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.latent_dim = config.latent_dim

        # Latent to model dimension projection
        self.latent_to_model = nn.Linear(config.latent_dim, d_model)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Output projection with AdaLN
        self.output_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.output_proj = nn.Linear(d_model, config.latent_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )

        # Schedulers
        self.noise_scheduler = FlowMatchingScheduler(
            num_train_timesteps=config.num_train_timesteps,
            prediction_type=config.prediction_type,
        )
        self.sample_scheduler = FlowMatchingScheduler(
            num_train_timesteps=config.num_train_timesteps,
            prediction_type=config.prediction_type,
        )

        # Prophet early exit
        self.prophet = ProphetEarlyExit(config.prophet)

    def get_time_embed(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        """Get sinusoidal time embedding."""
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device, dtype=torch.float32) * -emb)
        emb = timestep.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb.to(dtype))

    def forward_diffusion(
        self,
        model: nn.Module,
        z: Tensor,
        condition_embeds: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """
        Forward pass for diffusion training.

        Args:
            model: Base LLM for processing
            z: Noisy latent [B, M, latent_dim]
            condition_embeds: Condition embeddings [B, S, d_model]
            timestep: Diffusion timestep [B]

        Returns:
            model_pred: Predicted velocity/noise [B, M, latent_dim]
        """
        B = z.shape[0]

        # Project latent to model dimension
        z_proj = self.latent_to_model(z)  # [B, M, d_model]

        # Get time embedding
        t_emb = self.get_time_embed(timestep, z.dtype)  # [B, d_model]

        # Concatenate condition and noisy latent
        combined = torch.cat([condition_embeds, z_proj], dim=1)

        # Forward through model
        outputs = model(inputs_embeds=combined, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        # Extract latent positions
        latent_hidden = hidden[:, -z.shape[1]:, :]  # [B, M, d_model]

        # AdaLN modulation
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        latent_hidden = self.output_norm(latent_hidden) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Project to latent dimension
        model_pred = self.output_proj(latent_hidden)  # [B, M, latent_dim]

        return model_pred

    def denoise(
        self,
        model: nn.Module,
        condition_embeds: Tensor,
        target_shape: Tuple[int, int, int],
        vae: Optional[LaDiRVAE] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Denoise with Prophet early exit.

        Args:
            model: Base LLM
            condition_embeds: Condition embeddings [B, S, d_model]
            target_shape: (B, num_memory_slots, latent_dim)
            vae: VAE for logits computation (Prophet)
            verbose: Print progress

        Returns:
            dict with 'output', 'early_exit', 'exit_step', 'total_steps', 'speedup'
        """
        B, M, D = target_shape
        device = condition_embeds.device
        dtype = condition_embeds.dtype

        # Initialize with scaled noise
        z = torch.randn(B, M, D, dtype=dtype, device=device) * self.config.noise_scale

        # Setup scheduler
        self.sample_scheduler.set_timesteps(self.config.num_inference_steps, device=device)
        total_steps = len(self.sample_scheduler.timesteps)

        early_exit = False
        exit_step = total_steps
        final_gap = 0.0

        for step_idx, t in enumerate(self.sample_scheduler.timesteps):
            timestep = torch.full((B,), t, device=device, dtype=torch.long)

            # Forward diffusion step
            model_pred = self.forward_diffusion(model, z, condition_embeds, timestep)

            # Scheduler step
            z = self.sample_scheduler.step(
                model_output=model_pred,
                timestep=t.item(),
                sample=z,
            ).prev_sample

            # Prophet early exit check
            if vae is not None and self.prophet.enabled:
                if (step_idx + 1) % self.config.prophet.check_interval == 0:
                    try:
                        logits = vae.latent_to_logits(z, model)
                        should_exit, gap = self.prophet.should_exit(
                            logits, step_idx + 1, total_steps
                        )
                        final_gap = gap

                        if should_exit:
                            early_exit = True
                            exit_step = step_idx + 1
                            if verbose:
                                log.info(f"Prophet early exit at step {exit_step}/{total_steps} (gap={gap:.3f})")
                            break
                    except Exception as e:
                        log.warning(f"Prophet check failed: {e}")

            if verbose and (step_idx + 1) % 10 == 0:
                log.info(f"Denoising step {step_idx + 1}/{total_steps}")

        return {
            'output': z,
            'early_exit': early_exit,
            'exit_step': exit_step,
            'total_steps': total_steps,
            'speedup': total_steps / exit_step if exit_step > 0 else 1.0,
            'confidence_gap': final_gap,
        }

    def compute_loss(
        self,
        model: nn.Module,
        z_clean: Tensor,
        condition_embeds: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute diffusion training loss.

        Args:
            model: Base LLM
            z_clean: Clean latent from VAE [B, M, latent_dim]
            condition_embeds: Condition embeddings [B, S, d_model]

        Returns:
            dict with 'loss', 'mse_loss'
        """
        B = z_clean.shape[0]
        device = z_clean.device

        # Sample random timesteps
        timestep = torch.randint(0, self.config.num_train_timesteps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(z_clean)
        z_noisy = self.noise_scheduler.add_noise(z_clean, noise, timestep)

        # Predict
        model_pred = self.forward_diffusion(model, z_noisy, condition_embeds, timestep)

        # Flow matching target: velocity = noise - clean
        if self.config.prediction_type == "flow":
            target = noise - z_clean
        elif self.config.prediction_type == "epsilon":
            target = noise
        else:
            target = noise - z_clean

        # MSE loss
        mse_loss = F.mse_loss(model_pred, target)

        return {
            'loss': mse_loss,
            'mse_loss': mse_loss,
        }


class LaDiRModule(nn.Module):
    """
    LaDiR Module: 시퀀스 끝에서 Latent Reasoning 수행.

    CoCoNut과 동일한 위치 (시퀀스 END)에서 작동하지만,
    AR 대신 VAE + Flow Matching Diffusion으로 생성합니다.

    구조:
    1. Input: [Question Tokens] + [MEM₁] [MEM₂] [MEM₃]
    2. VAE: Memory Slots → Latent z (압축)
    3. Diffusion: Noise → Latent z (생성)
    4. Prophet: Confidence gap 기반 조기 종료
    5. Output: Reasoning이 담긴 Memory Slots
    """

    def __init__(
        self,
        d_model: int,
        config: LatentReasoningConfig,
        base_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.base_model = base_model
        self.num_memory_slots = config.ladir.num_memory_slots

        # Memory slot embeddings (learnable)
        self.memory_embeddings = nn.Parameter(
            torch.randn(config.ladir.num_memory_slots, d_model) * 0.02
        )

        # LaDiR: VAE + Flow Matching Diffusion + Prophet
        self.ladir_vae = LaDiRVAE(
            d_model=d_model,
            config=config.ladir,
            base_model=base_model,
        )
        self.ladir_diffusion = LaDiRDiffusion(
            d_model=d_model,
            config=config.ladir,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

        # Superposition tracking
        if config.track_superposition:
            self.register_buffer('superposition_entropy', torch.zeros(1))

    def append_memory_slots(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        시퀀스 끝에 Memory Slots를 추가합니다.

        Args:
            hidden_states: [B, S, D] 입력 hidden states
            attention_mask: [B, S] attention mask

        Returns:
            extended_hidden: [B, S + num_memory_slots, D]
            extended_mask: [B, S + num_memory_slots]
        """
        B, S, D = hidden_states.shape

        # Expand memory embeddings for batch
        memory_slots = self.memory_embeddings.unsqueeze(0).expand(B, -1, -1)

        # Append memory slots to end of sequence
        extended_hidden = torch.cat([hidden_states, memory_slots], dim=1)

        # Extend attention mask if provided
        extended_mask = None
        if attention_mask is not None:
            memory_mask = torch.ones(
                B, self.num_memory_slots,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            extended_mask = torch.cat([attention_mask, memory_mask], dim=1)

        return extended_hidden, extended_mask

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cot_hidden_states: Optional[Tensor] = None,
        return_info: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        LaDiR forward pass - 시퀀스 끝에서 작동.

        Training Mode:
            - cot_hidden_states가 제공되면 VAE로 압축 학습
            - Memory slots 위치에 압축된 reasoning 정보 생성

        Inference Mode:
            - Diffusion으로 Memory slots 내용 생성
            - Prophet으로 조기 종료 가능

        Args:
            hidden_states: [B, S, D] 모델 출력 (ETD 통과 후)
            attention_mask: [B, S] attention mask
            cot_hidden_states: [B, S_cot, D] CoT hidden states (학습용)
            return_info: statistics 반환 여부

        Returns:
            output: [B, S + num_memory_slots, D] Memory slots가 추가된 출력
            info: (optional) statistics
        """
        B, S, D = hidden_states.shape
        info = {}

        if self.training and cot_hidden_states is not None:
            # ============================================
            # Training Mode: VAE로 CoT 압축 학습
            # ============================================

            # CoT의 마지막 부분을 Memory slots로 압축
            # cot_hidden_states: [B, S_cot, D] → z: [B, num_memory_slots, latent_dim]
            vae_output = self.ladir_vae(cot_hidden_states)
            z_clean = vae_output['z']
            kl_loss = vae_output['kl_loss']

            # Memory slots 위치의 hidden states로 복원
            reconstructed = vae_output['reconstructed']  # [B, num_memory_slots, D]

            # Diffusion loss 계산 (condition: input hidden states)
            diff_output = self.ladir_diffusion.compute_loss(
                model=self.base_model,
                z_clean=z_clean,
                condition_embeds=hidden_states,
            )

            # Output: input + memory slots (reconstructed)
            output = torch.cat([hidden_states, reconstructed], dim=1)

            info['kl_loss'] = kl_loss
            info['diffusion_loss'] = diff_output['loss']
            info['mse_loss'] = diff_output['mse_loss']
            info['z'] = z_clean

        else:
            # ============================================
            # Inference Mode: Diffusion으로 Memory slots 생성
            # ============================================

            target_shape = (
                B,
                self.num_memory_slots,
                self.config.ladir.latent_dim,
            )

            # Prophet early exit과 함께 denoising
            denoise_result = self.ladir_diffusion.denoise(
                model=self.base_model,
                condition_embeds=hidden_states,
                target_shape=target_shape,
                vae=self.ladir_vae,
                verbose=False,
            )

            # Latent z를 hidden states로 decode
            z_denoised = denoise_result['output']
            memory_hidden = self.ladir_vae.decode(z_denoised)  # [B, num_memory_slots, D]

            # Output: input + generated memory slots
            output = torch.cat([hidden_states, memory_hidden], dim=1)

            info['early_exit'] = denoise_result['early_exit']
            info['exit_step'] = denoise_result['exit_step']
            info['total_steps'] = denoise_result['total_steps']
            info['speedup'] = denoise_result['speedup']
            info['confidence_gap'] = denoise_result['confidence_gap']

        # Normalize output
        output = self.output_norm(output)

        if return_info:
            return output, info
        return output


# Alias for backward compatibility
LatentThinkBlock = LaDiRModule


class LatentReasoningWrapper(nn.Module):
    """
    기존 모델에 LaDiR Latent Reasoning을 추가하는 wrapper.

    중요: LaDiR은 시퀀스 END에서 작동합니다 (CoCoNut과 동일한 위치).

    데이터 흐름:
    1. Input → Base Model (with ETD) → Hidden States
    2. Hidden States → LaDiR (시퀀스 END에 Memory Slots 추가) → Output
    3. Output → LM Head → Logits

    CoCoNut vs LaDiR:
    - 위치: 동일 (시퀀스 END)
    - 생성 방식: AR (CoCoNut) vs Diffusion (LaDiR)
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: LatentReasoningConfig,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config

        # Get model dimension
        if hasattr(base_model, 'd_model'):
            d_model = base_model.d_model
        elif hasattr(base_model, 'config'):
            d_model = getattr(base_model.config, 'hidden_size',
                           getattr(base_model.config, 'd_model', 4096))
        else:
            d_model = 4096

        self.d_model = d_model

        # LaDiR Module (시퀀스 END에서 작동)
        self.ladir_module = LaDiRModule(
            d_model=d_model,
            config=config,
            base_model=base_model,
        )

        # LM Head projection (optional, if needed)
        self.has_lm_head = hasattr(base_model, 'lm_head') or hasattr(base_model, 'output')

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
        cot_hidden_states: Optional[Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with LaDiR latent reasoning.

        핵심: Base Model을 먼저 통과한 후, 시퀀스 END에 LaDiR 적용.

        Args:
            input_ids: [B, S] input token ids
            inputs_embeds: [B, S, D] input embeddings (alternative to input_ids)
            attention_mask: [B, S] attention mask
            labels: [B, S] labels for training
            cot_hidden_states: [B, S_cot, D] CoT hidden states (학습용)
            return_dict: return dict instead of tuple

        Returns:
            dict with 'logits', 'loss', 'hidden_states', 'ladir_info'
        """
        # ============================================
        # Step 1: Base Model Forward (with ETD if configured)
        # ============================================
        base_outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Extract hidden states from base model output
        if hasattr(base_outputs, 'hidden_states'):
            hidden_states = base_outputs.hidden_states[-1]  # Last layer
        elif isinstance(base_outputs, dict):
            hidden_states = base_outputs.get('hidden_states', [None])[-1]
        elif isinstance(base_outputs, tuple) and len(base_outputs) > 1:
            # (logits, hidden_states)
            hidden_states = base_outputs[1][-1] if isinstance(base_outputs[1], (list, tuple)) else base_outputs[1]
        else:
            raise ValueError("Cannot extract hidden states from base model output")

        # ============================================
        # Step 2: LaDiR at Sequence END
        # ============================================
        # hidden_states: [B, S, D] → [B, S + num_memory_slots, D]
        ladir_output, ladir_info = self.ladir_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cot_hidden_states=cot_hidden_states,
            return_info=True,
        )

        # ============================================
        # Step 3: Compute Logits (LM Head)
        # ============================================
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(ladir_output)
        elif hasattr(self.base_model, 'output'):
            logits = self.base_model.output(ladir_output)
        elif hasattr(base_outputs, 'logits'):
            # Cannot apply LM head directly, just use base logits
            # (memory slots won't have valid logits)
            logits = base_outputs.logits
        else:
            logits = None

        # ============================================
        # Step 4: Compute Loss (if labels provided)
        # ============================================
        loss = None
        if labels is not None and logits is not None:
            # Shift logits and labels for LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Add LaDiR losses if training
            if self.training:
                kl_loss = ladir_info.get('kl_loss', 0.0)
                diff_loss = ladir_info.get('diffusion_loss', 0.0)

                # Total loss
                loss = lm_loss + self.config.ladir.vae_beta * kl_loss + diff_loss
                ladir_info['lm_loss'] = lm_loss
            else:
                loss = lm_loss

        result = {
            "logits": logits,
            "loss": loss,
            "hidden_states": ladir_output,
            "ladir_info": ladir_info,
        }

        return result

    def generate_with_latent_reasoning(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Tensor:
        """
        Latent reasoning을 사용한 생성.

        1. Base Model로 input 처리
        2. LaDiR로 Memory Slots 생성 (시퀀스 END)
        3. Memory Slots 기반으로 생성
        """
        self.eval()

        with torch.no_grad():
            # Forward through model with LaDiR
            outputs = self.forward(input_ids=input_ids, **kwargs)

            # Get hidden states with memory slots
            hidden_with_memory = outputs['hidden_states']

            # Generate from memory-augmented representation
            if hasattr(self.base_model, 'generate'):
                # Note: Most HF models don't support inputs_embeds in generate
                # This is a simplified implementation
                return self.base_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            else:
                # Manual AR generation
                generated = self._generate_ar(
                    hidden_with_memory,
                    max_new_tokens=max_new_tokens,
                )
                return generated

    def _generate_ar(
        self,
        hidden_states: Tensor,
        max_new_tokens: int = 100,
    ) -> Tensor:
        """Simple AR generation from hidden states."""
        # Get LM head
        if hasattr(self.base_model, 'lm_head'):
            lm_head = self.base_model.lm_head
        elif hasattr(self.base_model, 'output'):
            lm_head = self.base_model.output
        else:
            raise NotImplementedError("No LM head found")

        B = hidden_states.shape[0]
        generated_ids = []

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = lm_head(hidden_states[:, -1:, :])  # [B, 1, V]
            next_token = logits.argmax(dim=-1)  # [B, 1]
            generated_ids.append(next_token)

            # Get embedding for next token
            if hasattr(self.base_model, 'get_input_embeddings'):
                embed = self.base_model.get_input_embeddings()
            else:
                embed = self.base_model.embed_tokens
            next_embed = embed(next_token)  # [B, 1, D]

            # Append to hidden states
            hidden_states = torch.cat([hidden_states, next_embed], dim=1)

        return torch.cat(generated_ids, dim=1)


def wrap_model_with_latent_reasoning(
    model: nn.Module,
    config: Optional[LatentReasoningConfig] = None,
) -> LatentReasoningWrapper:
    """
    기존 모델에 LaDiR Latent Reasoning을 추가합니다.

    Args:
        model: Base transformer model
        config: Latent reasoning configuration

    Returns:
        LatentReasoningWrapper: Wrapped model

    Example:
        ```python
        from olmo_core.nn.transformer import (
            LatentReasoningConfig,
            wrap_model_with_latent_reasoning,
        )

        config = LatentReasoningConfig()
        model = wrap_model_with_latent_reasoning(base_model, config)

        outputs = model(input_ids)
        ```
    """
    if config is None:
        config = LatentReasoningConfig()

    return LatentReasoningWrapper(model, config)


# ============================================================================
# Utility Functions
# ============================================================================

def decode_latent_to_tokens(
    latent_thoughts: Tensor,
    embed_matrix: Tensor,
    top_k: int = 5,
) -> List[List[int]]:
    """
    Latent thoughts를 discrete tokens으로 decoding.

    Interpretability를 위해 continuous thought가
    어떤 token들과 유사한지 확인합니다.
    """
    # Compute similarity with embedding matrix
    similarities = torch.matmul(latent_thoughts, embed_matrix.T)  # [B, T, V]

    # Get top-k tokens for each thought
    top_k_values, top_k_indices = similarities.topk(top_k, dim=-1)

    return top_k_indices.tolist()


def compute_superposition_entropy(
    thoughts: Tensor,
    num_bins: int = 100,
) -> Tensor:
    """
    Superposition entropy 계산.

    Thought vector의 "spread"를 측정하여
    얼마나 많은 reasoning paths가 인코딩되어 있는지 추정합니다.
    """
    # Normalize thoughts
    thoughts_norm = F.normalize(thoughts, dim=-1)

    # Compute pairwise cosine similarities
    B, T, D = thoughts_norm.shape

    # Flatten batch and thoughts
    flat = thoughts_norm.view(-1, D)

    # Self-similarity matrix
    sim_matrix = torch.matmul(flat, flat.T)

    # Convert to probabilities (softmax)
    probs = F.softmax(sim_matrix, dim=-1)

    # Compute entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()

    return entropy


def create_latent_reasoning_pipeline(
    base_model: nn.Module,
    **kwargs,
) -> LatentReasoningWrapper:
    """
    간편한 LaDiR Latent Reasoning 파이프라인 생성.

    Args:
        base_model: Base transformer model
        **kwargs: LaDiRConfig 옵션

    Example:
        ```python
        # LaDiR + Prophet
        model = create_latent_reasoning_pipeline(base_model)

        # Custom config
        model = create_latent_reasoning_pipeline(
            base_model,
            ladir=LaDiRConfig(
                latent_dim=256,
                num_memory_slots=4,
            ),
        )
        ```
    """
    config = LatentReasoningConfig(**kwargs)
    return wrap_model_with_latent_reasoning(base_model, config)
