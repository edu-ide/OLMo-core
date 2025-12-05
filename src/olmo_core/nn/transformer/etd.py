"""
ETD (Encode-Think-Decode) + MoDr + Dr.LLM + Triple-Hybrid 통합 모듈

이 모듈은 네 가지 최신 아키텍처를 통합합니다:
- ETD: 레이어를 Encoder/Think/Decoder로 구조화, Think 블록 반복
- MoDr: LoRA 기반 전문가 분기로 사고의 다양성 확보
- Dr.LLM: Skip/Execute/Repeat 라우터로 동적 깊이 제어
- Triple-Hybrid: Mamba-3 + Gated DeltaNet + Gated Attention 백본

Based on Huginn reference (seal-rg/recurrent-pretraining):
- Prelude + Adapter + Core Block + Coda architecture
- HuginnDynamicCache for step-aware KV caching
- SandwichBlock with 4 layer norms

References:
- ETD: arXiv:2510.07358 (Encode, Think, Decode)
- Huginn: ICML 2025 Spotlight (seal-rg/recurrent-pretraining)
- MoDr: OpenReview (Mixture-of-Depth-Recurrent Transformers)
- Dr.LLM: arXiv:2510.12773 (Dynamic Layer Routing in LLMs)
- Triple-Hybrid: Mamba-3 + Gated DeltaNet + Gated Attention Architecture
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import Config

log = logging.getLogger(__name__)

__all__ = [
    "LayerRouterConfig",
    "LayerRouter",
    "LoRAExpertConfig",
    "LoRAExpert",
    "MoDrExpertRouter",
    "ETDConfig",
    "RouterAction",
    "ThinkBlockController",
    "ETDTransformer",
    "wrap_transformer_with_etd",
    # Huginn-style components
    "HuginnDynamicCache",
    "SandwichBlock",
    "HuginnAdapter",
    # Triple-Hybrid integration
    "BackboneType",
    "ETDTripleHybridConfig",
    "create_etd_triple_hybrid_model",
]


class BackboneType(str, Enum):
    """백본 레이어 타입"""
    STANDARD = "standard"      # 기존 Transformer 블록 사용
    TRIPLE_HYBRID = "triple_hybrid"  # Triple-Hybrid (Mamba-3 + DeltaNet + Attention)


class RouterAction(Enum):
    """Dr.LLM 스타일 라우터 액션"""
    SKIP = 0      # 레이어 건너뛰기 (residual만 통과)
    EXECUTE = 1   # 레이어 1회 실행
    REPEAT = 2    # 레이어 2회 실행


@dataclass
class LayerRouterConfig(Config):
    """
    Dr.LLM 스타일 레이어 라우터 설정.

    각 레이어 앞에 배치되어 Skip/Execute/Repeat 결정을 내립니다.
    Windowed mean pooling으로 긴 시퀀스에서도 안정적인 라우팅을 제공합니다.
    """

    hidden_size: int = 256
    """라우터 MLP의 hidden dimension"""

    window_size: int = 8
    """Windowed mean pooling의 윈도우 크기"""

    num_actions: int = 3
    """액션 수 (Skip=0, Execute=1, Repeat=2)"""

    temperature: float = 1.0
    """Softmax 온도 (낮을수록 더 sharp한 결정)"""

    use_gumbel: bool = False
    """학습 시 Gumbel-Softmax 사용 여부"""

    def build(
        self,
        d_model: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> "LayerRouter":
        return LayerRouter(
            d_model=d_model,
            hidden_size=self.hidden_size,
            window_size=self.window_size,
            num_actions=self.num_actions,
            temperature=self.temperature,
            use_gumbel=self.use_gumbel,
            dtype=dtype,
            init_device=init_device,
        )


class LayerRouter(nn.Module):
    """
    Dr.LLM 스타일 레이어 라우터.

    입력 hidden states를 분석하여 해당 레이어를 Skip/Execute/Repeat할지 결정합니다.
    Windowed mean pooling을 사용하여 긴 시퀀스에서도 안정적인 결정을 내립니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int = 256,
        window_size: int = 8,
        num_actions: int = 3,
        temperature: float = 1.0,
        use_gumbel: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_actions = num_actions
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Bottleneck MLP: Linear -> GELU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size, bias=True, dtype=dtype, device=init_device),
            nn.GELU(),
            nn.Linear(hidden_size, num_actions, bias=True, dtype=dtype, device=init_device),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Hidden states, shape (B, T, D)
            return_probs: 확률 분포도 반환할지 여부

        Returns:
            action: 선택된 액션 (B,) - 0=Skip, 1=Execute, 2=Repeat
            probs: (optional) 액션 확률 분포 (B, num_actions)
        """
        B, T, D = x.shape

        # Windowed mean pooling
        if T <= self.window_size:
            pooled = x.mean(dim=1)  # (B, D)
        else:
            # 윈도우별 평균 후 전체 평균
            num_windows = (T + self.window_size - 1) // self.window_size
            padded_len = num_windows * self.window_size
            if padded_len > T:
                x_padded = F.pad(x, (0, 0, 0, padded_len - T), value=0)
            else:
                x_padded = x
            x_windowed = x_padded.view(B, num_windows, self.window_size, D)
            pooled = x_windowed.mean(dim=2).mean(dim=1)  # (B, D)

        # MLP로 액션 logits 계산
        logits = self.mlp(pooled)  # (B, num_actions)

        if self.training and self.use_gumbel:
            # Gumbel-Softmax for differentiable sampling
            probs = F.gumbel_softmax(logits / self.temperature, tau=1.0, hard=True)
            action = probs.argmax(dim=-1)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            action = probs.argmax(dim=-1)

        if return_probs:
            return action, probs
        return action


@dataclass
class LoRAExpertConfig(Config):
    """
    MoDr 스타일 LoRA 전문가 설정.

    공유 가중치에 LoRA 델타를 더해 다양한 전문가 분기를 생성합니다.
    """

    rank: int = 16
    """LoRA rank (낮을수록 파라미터 효율적)"""

    alpha: float = 32.0
    """LoRA scaling factor"""

    dropout: float = 0.0
    """LoRA dropout"""

    def build(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> "LoRAExpert":
        return LoRAExpert(
            in_features=in_features,
            out_features=out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            dtype=dtype,
            init_device=init_device,
        )


class LoRAExpert(nn.Module):
    """
    MoDr 스타일 LoRA 전문가.

    기존 Linear 레이어에 저랭크 어댑터를 추가하여 전문가 분기를 생성합니다.
    W' = W + (alpha/rank) * B @ A
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: W' = W + (alpha/r) * B @ A
        self.lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=init_device)
        self.lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=init_device)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        # A는 Kaiming uniform, B는 zero 초기화 (LoRA 표준)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA 델타를 계산합니다.

        Args:
            x: 입력 텐서 (*, in_features)

        Returns:
            LoRA 델타 (*, out_features) - 기존 출력에 더해야 함
        """
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class MoDrExpertRouter(nn.Module):
    """
    MoDr 스타일 전문가 라우터.

    Think 블록 내에서 매 반복마다 어떤 LoRA 전문가를 사용할지 결정합니다.
    Hard-gate 라우팅으로 효율성을 유지하면서 다양한 사고 패턴을 가능하게 합니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # 간단한 linear router
        self.router = nn.Linear(d_model, num_experts, bias=False, dtype=dtype, device=init_device)

        # Load balancing을 위한 bias (auxiliary-loss-free)
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=dtype, device=init_device),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Hidden states, shape (B, T, D)

        Returns:
            expert_idx: 선택된 전문가 인덱스 (B,)
            probs: (optional) 전문가 선택 확률 (B, num_experts)
        """
        # Sequence mean pooling
        pooled = x.mean(dim=1)  # (B, D)

        # Router logits + bias (DeepSeek-V3 스타일 auxiliary-loss-free)
        logits = self.router(pooled) + self.expert_bias
        probs = F.softmax(logits, dim=-1)

        # Hard gate: top-1 selection
        expert_idx = probs.argmax(dim=-1)

        if return_probs:
            return expert_idx, probs
        return expert_idx

    @torch.no_grad()
    def update_bias(self, expert_counts: torch.Tensor, lr: float = 0.01):
        """
        Auxiliary-loss-free load balancing: bias 업데이트로 부하 분산.

        Args:
            expert_counts: 각 전문가가 선택된 횟수 (num_experts,)
            lr: 업데이트 학습률
        """
        # 평균보다 많이 선택된 전문가는 bias 감소, 적게 선택된 전문가는 증가
        mean_count = expert_counts.float().mean()
        delta = (mean_count - expert_counts.float()) / (mean_count + 1e-8)
        self.expert_bias.add_(delta * lr)


@dataclass
class ETDConfig(Config):
    """
    ETD (Encode-Think-Decode) + MoDr + Dr.LLM 통합 설정.

    세 가지 아키텍처를 결합하여:
    - ETD: 구조적 레이어 분할 (어디서 생각할지)
    - MoDr: LoRA 전문가 분기 (어떻게 생각할지)
    - Dr.LLM: 동적 깊이 라우팅 (얼마나 생각할지)
    """

    # ETD 구조 설정
    n_encoder_layers: int = 7
    """Encoder 블록 레이어 수 (입력을 latent space로 매핑)"""

    n_think_layers: int = 4
    """Think 블록 레이어 수 (반복되는 추론 블록)"""

    n_decoder_layers: int = 5
    """Decoder 블록 레이어 수 (latent에서 출력으로 매핑)"""

    max_think_iterations: int = 5
    """Think 블록 최대 반복 횟수"""

    adaptive_depth: bool = True
    """적응형 깊이 사용 여부 (신뢰도 기반 조기 종료)"""

    confidence_threshold: float = 0.9
    """조기 종료를 위한 신뢰도 임계값"""

    # Dr.LLM 라우터 설정
    use_layer_router: bool = True
    """레이어별 Skip/Execute/Repeat 라우터 사용 여부"""

    layer_router: LayerRouterConfig = field(default_factory=LayerRouterConfig)
    """레이어 라우터 설정"""

    # MoDr 전문가 설정
    use_lora_experts: bool = True
    """Think 블록에 LoRA 전문가 사용 여부"""

    num_lora_experts: int = 4
    """LoRA 전문가 수"""

    lora_expert: LoRAExpertConfig = field(default_factory=LoRAExpertConfig)
    """LoRA 전문가 설정"""

    # 학습 설정
    router_warmup_steps: int = 1000
    """라우터 웜업 스텝 (초기에는 모두 Execute)"""

    expert_warmup_steps: int = 500
    """전문가 웜업 스텝 (초기에는 균등 선택)"""

    def validate(self, n_layers: int):
        """총 레이어 수와 ETD 분할이 맞는지 검증"""
        total = self.n_encoder_layers + self.n_think_layers + self.n_decoder_layers
        if total != n_layers:
            raise ValueError(
                f"ETD 레이어 분할({total})이 모델 레이어 수({n_layers})와 불일치. "
                f"E={self.n_encoder_layers}, T={self.n_think_layers}, D={self.n_decoder_layers}"
            )

    def get_layer_ranges(self) -> Dict[str, Tuple[int, int]]:
        """각 블록의 레이어 인덱스 범위 반환"""
        enc_end = self.n_encoder_layers
        think_end = enc_end + self.n_think_layers
        dec_end = think_end + self.n_decoder_layers
        return {
            "encoder": (0, enc_end),
            "think": (enc_end, think_end),
            "decoder": (think_end, dec_end),
        }


class ThinkBlockController(nn.Module):
    """
    Think 블록 반복을 제어하는 컨트롤러.

    적응형 깊이 전략: 각 반복 후 신뢰도를 평가하여 조기 종료 결정.
    """

    def __init__(
        self,
        *,
        d_model: int,
        max_iterations: int = 5,
        confidence_threshold: float = 0.9,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # 종료 확률을 예측하는 분류기
        self.exit_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4, dtype=dtype, device=init_device),
            nn.GELU(),
            nn.Linear(d_model // 4, 1, dtype=dtype, device=init_device),
            nn.Sigmoid(),
        )

    def should_exit(self, h: torch.Tensor, iteration: int) -> Tuple[bool, torch.Tensor]:
        """
        현재 반복에서 종료할지 결정.

        Args:
            h: 현재 hidden states (B, T, D)
            iteration: 현재 반복 횟수 (0-indexed)

        Returns:
            should_exit: 종료 여부
            confidence: 신뢰도 점수 (B,)
        """
        # 최대 반복 도달 시 강제 종료
        if iteration >= self.max_iterations - 1:
            return True, torch.ones(h.shape[0], device=h.device)

        # 시퀀스 평균으로 종료 확률 계산
        pooled = h.mean(dim=1)  # (B, D)
        confidence = self.exit_classifier(pooled).squeeze(-1)  # (B,)

        # 배치 전체의 평균 신뢰도로 결정 (배치 동기화 유지)
        mean_confidence = confidence.mean()
        should_exit = mean_confidence >= self.confidence_threshold

        return should_exit.item(), confidence


class ETDTransformer(nn.Module):
    """
    ETD (Encode-Think-Decode) + MoDr + Dr.LLM 통합 Transformer.

    기존 Transformer를 감싸서 ETD 구조를 구현합니다:
    1. Encoder: 입력을 latent space로 매핑 (1회 실행)
    2. Think: 추론 레이어를 반복 실행 (MoDr 전문가 + Dr.LLM 라우터)
    3. Decoder: latent에서 출력으로 매핑 (1회 실행)

    Args:
        base_blocks: 기존 Transformer의 블록들 (nn.ModuleDict)
        etd_config: ETD 설정
        d_model: 모델 차원
        dtype: 데이터 타입
        init_device: 초기화 디바이스
    """

    def __init__(
        self,
        *,
        base_blocks: nn.ModuleDict,
        etd_config: ETDConfig,
        d_model: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.etd_config = etd_config
        self.d_model = d_model

        # ETD 설정 검증
        n_layers = len(base_blocks)
        etd_config.validate(n_layers)
        layer_ranges = etd_config.get_layer_ranges()

        # 블록들을 Encoder/Think/Decoder로 분할
        self.encoder_blocks = nn.ModuleDict()
        self.think_blocks = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()

        enc_start, enc_end = layer_ranges["encoder"]
        think_start, think_end = layer_ranges["think"]
        dec_start, dec_end = layer_ranges["decoder"]

        for i, (key, block) in enumerate(base_blocks.items()):
            if enc_start <= i < enc_end:
                self.encoder_blocks[key] = block
            elif think_start <= i < think_end:
                self.think_blocks[key] = block
            elif dec_start <= i < dec_end:
                self.decoder_blocks[key] = block

        # Dr.LLM 스타일 레이어 라우터 (Think 블록용)
        self.layer_routers: Optional[nn.ModuleDict] = None
        if etd_config.use_layer_router:
            self.layer_routers = nn.ModuleDict()
            for key in self.think_blocks.keys():
                self.layer_routers[key] = etd_config.layer_router.build(
                    d_model=d_model,
                    dtype=dtype,
                    init_device=init_device,
                )

        # MoDr 스타일 LoRA 전문가 (Think 블록용)
        self.lora_experts: Optional[nn.ModuleDict] = None
        self.expert_router: Optional[MoDrExpertRouter] = None
        if etd_config.use_lora_experts:
            self.lora_experts = nn.ModuleDict()
            for key in self.think_blocks.keys():
                experts = nn.ModuleList([
                    etd_config.lora_expert.build(
                        in_features=d_model,
                        out_features=d_model,
                        dtype=dtype,
                        init_device=init_device,
                    )
                    for _ in range(etd_config.num_lora_experts)
                ])
                self.lora_experts[key] = experts

            self.expert_router = MoDrExpertRouter(
                d_model=d_model,
                num_experts=etd_config.num_lora_experts,
                dtype=dtype,
                init_device=init_device,
            )

        # Think 블록 컨트롤러 (적응형 깊이)
        self.think_controller: Optional[ThinkBlockController] = None
        if etd_config.adaptive_depth:
            self.think_controller = ThinkBlockController(
                d_model=d_model,
                max_iterations=etd_config.max_think_iterations,
                confidence_threshold=etd_config.confidence_threshold,
                dtype=dtype,
                init_device=init_device,
            )

        # 학습 상태 추적
        self._step_count = 0
        self._router_stats: Dict[str, List[int]] = {
            "skip": [], "execute": [], "repeat": []
        }
        self._expert_counts: Optional[torch.Tensor] = None
        if etd_config.use_lora_experts:
            self._expert_counts = torch.zeros(
                etd_config.num_lora_experts, dtype=torch.long
            )

    def forward(
        self,
        h: torch.Tensor,
        **block_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        ETD forward pass.

        Args:
            h: 임베딩 출력 (B, T, D)
            **block_kwargs: 블록에 전달할 추가 인자

        Returns:
            h: 최종 hidden states (B, T, D)
            metrics: 라우팅/전문가 사용 통계
        """
        metrics: Dict[str, Any] = {
            "think_iterations": 0,
            "router_actions": [],
            "expert_selections": [],
            "early_exit": False,
        }

        # 1. Encoder: 입력을 latent space로 매핑
        for block_key, block in self.encoder_blocks.items():
            h = block(h, **block_kwargs)

        # 2. Think: 반복적 추론 (MoDr + Dr.LLM)
        think_iteration = 0
        max_iterations = self.etd_config.max_think_iterations

        while think_iteration < max_iterations:
            h_before_think = h

            # Think 블록 1회 실행
            for block_key, block in self.think_blocks.items():
                # Dr.LLM 라우터 결정
                if self.layer_routers is not None and self._should_use_router():
                    router = self.layer_routers[block_key]
                    action, probs = router(h, return_probs=True)

                    # 배치의 첫 번째 샘플 기준으로 결정 (학습 시 동기화)
                    action_val = action[0].item()
                    metrics["router_actions"].append(action_val)

                    if action_val == RouterAction.SKIP.value:
                        # Skip: residual만 통과
                        self._router_stats["skip"].append(int(block_key))
                        continue
                    elif action_val == RouterAction.REPEAT.value:
                        # Repeat: 2회 실행
                        self._router_stats["repeat"].append(int(block_key))
                        h = block(h, **block_kwargs)
                        h = block(h, **block_kwargs)
                    else:
                        # Execute: 1회 실행
                        self._router_stats["execute"].append(int(block_key))
                        h = block(h, **block_kwargs)
                else:
                    # 라우터 없이 항상 실행
                    h = block(h, **block_kwargs)

                # MoDr LoRA 전문가 적용
                if self.lora_experts is not None and self._should_use_experts():
                    expert_idx = self.expert_router(h)
                    expert_val = expert_idx[0].item()
                    metrics["expert_selections"].append(expert_val)

                    # 선택된 전문가의 LoRA 델타 적용
                    experts = self.lora_experts[block_key]
                    lora_delta = experts[expert_val](h)
                    h = h + lora_delta

                    # 전문가 사용 통계 업데이트
                    if self._expert_counts is not None:
                        self._expert_counts[expert_val] += 1

            think_iteration += 1
            metrics["think_iterations"] = think_iteration

            # 적응형 깊이: 조기 종료 검사
            if self.think_controller is not None and think_iteration < max_iterations:
                should_exit, confidence = self.think_controller.should_exit(
                    h, think_iteration
                )
                if should_exit:
                    metrics["early_exit"] = True
                    metrics["exit_confidence"] = confidence.mean().item()
                    break

        # 3. Decoder: latent에서 출력으로 매핑
        for block_key, block in self.decoder_blocks.items():
            h = block(h, **block_kwargs)

        self._step_count += 1
        return h, metrics

    def _should_use_router(self) -> bool:
        """웜업 기간 동안 라우터 비활성화"""
        return self._step_count >= self.etd_config.router_warmup_steps

    def _should_use_experts(self) -> bool:
        """웜업 기간 동안 전문가 라우팅 비활성화"""
        return self._step_count >= self.etd_config.expert_warmup_steps

    def update_expert_bias(self, lr: float = 0.01):
        """배치 후 전문가 bias 업데이트 (load balancing)"""
        if self.expert_router is not None and self._expert_counts is not None:
            self.expert_router.update_bias(self._expert_counts, lr)
            self._expert_counts.zero_()

    def get_router_stats(self) -> Dict[str, Any]:
        """라우터 통계 반환"""
        return {
            "skip_count": len(self._router_stats["skip"]),
            "execute_count": len(self._router_stats["execute"]),
            "repeat_count": len(self._router_stats["repeat"]),
            "skip_layers": self._router_stats["skip"][-10:],  # 최근 10개
            "repeat_layers": self._router_stats["repeat"][-10:],
        }

    def reset_stats(self):
        """통계 초기화"""
        self._router_stats = {"skip": [], "execute": [], "repeat": []}


# =============================================================================
# Huginn-style Components (from seal-rg/recurrent-pretraining reference)
# =============================================================================


class HuginnDynamicCache:
    """
    Huginn-style step-aware KV cache.

    Stores KV cache per recurrent step, enabling different lookup strategies
    for recurrent transformers. Based on recpre/raven_modeling_minimal.py.

    Args:
        lookup_strategy: Strategy for handling missing entries
            - "full": Materialize all entries (default)
            - "latest-m4": Use latest entry with modulo-4 step matching
            - "skip": Only use entries that exist
    """

    def __init__(self, lookup_strategy: str = "full") -> None:
        self._seen_tokens = 0
        # Structure: cache[step_idx][seq_position] = tensor
        self.key_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self.value_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self.lookup_strategy = lookup_strategy

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx: int,
        lookup_strategy: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache and return materialized past states.

        Args:
            key_states: New key states [B, num_heads, seq_len, head_dim]
            value_states: New value states [B, num_heads, seq_len, head_dim]
            step_idx: Current step index (layer or recurrent iteration)
            lookup_strategy: Override default lookup strategy

        Returns:
            Tuple of (past_keys, past_values) tensors
        """
        lookup_strategy = lookup_strategy or self.lookup_strategy

        # Initialize cache for this step if needed
        if step_idx not in self.key_cache:
            self.key_cache[step_idx] = {}
            self.value_cache[step_idx] = {}

        # Track tokens seen (only count on step 0)
        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Add entries to cache
        start_pos = self._seen_tokens - key_states.shape[-2]
        for idx, (k_entry, v_entry) in enumerate(
            zip(key_states.unbind(dim=-2), value_states.unbind(dim=-2))
        ):
            pos = start_pos + idx
            self.key_cache[step_idx][pos] = k_entry
            self.value_cache[step_idx][pos] = v_entry

        # Materialize based on strategy
        if lookup_strategy == "full" or len(self.key_cache[step_idx]) == self._seen_tokens:
            return (
                torch.stack(list(self.key_cache[step_idx].values()), dim=-2),
                torch.stack(list(self.value_cache[step_idx].values()), dim=-2),
            )
        elif lookup_strategy.startswith("latest-m4"):
            # Use latest entry matching modulo-4 pattern (Huginn block structure)
            latest_keys, latest_values = [], []
            for pos in range(self._seen_tokens):
                if step_idx >= 2:
                    valid_steps = [s for s in range(step_idx + 1) if pos in self.key_cache.get(s, {})]
                    max_step = max([s for s in valid_steps if s >= 2 and s % 4 == step_idx % 4], default=0)
                else:
                    max_step = step_idx if pos in self.key_cache[step_idx] else 0
                latest_keys.append(self.key_cache[max_step][pos])
                latest_values.append(self.value_cache[max_step][pos])
            return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
        elif lookup_strategy == "skip":
            # Only return entries that exist
            existing_keys = [self.key_cache[step_idx][pos] for pos in range(self._seen_tokens)
                           if pos in self.key_cache[step_idx]]
            existing_values = [self.value_cache[step_idx][pos] for pos in range(self._seen_tokens)
                             if pos in self.value_cache[step_idx]]
            return torch.stack(existing_keys, dim=-2), torch.stack(existing_values, dim=-2)
        else:
            raise ValueError(f"Unknown lookup strategy: {lookup_strategy}")

    def reset(self) -> None:
        """Reset the cache state."""
        self._seen_tokens = 0
        self.key_cache.clear()
        self.value_cache.clear()

    def get_seq_length(self, step_idx: int = 0) -> int:
        """Return number of tokens seen."""
        return self._seen_tokens


class SandwichBlock(nn.Module):
    """
    Huginn-style Sandwich Block with 4 layer norms.

    Unlike standard pre-norm or post-norm, this uses:
    - norm_1 before attention, norm_2 after attention residual
    - norm_3 before MLP, norm_4 after MLP residual

    Pattern:
        x = norm_2(attn(norm_1(x)) + x)
        x = norm_4(mlp(norm_3(x)) + x)

    This provides better gradient flow and stability for recurrent models.
    Based on recpre/model_dynamic.py SandwichBlock.
    """

    def __init__(
        self,
        *,
        d_model: int,
        attention: nn.Module,
        mlp: nn.Module,
        norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model

        # 4 layer norms (sandwich pattern)
        self.norm_1 = nn.RMSNorm(d_model, eps=norm_eps, device=init_device)
        self.norm_2 = nn.RMSNorm(d_model, eps=norm_eps, device=init_device)
        self.norm_3 = nn.RMSNorm(d_model, eps=norm_eps, device=init_device)
        self.norm_4 = nn.RMSNorm(d_model, eps=norm_eps, device=init_device)

        self.attn = attention
        self.mlp = mlp

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sandwich forward pass.

        Args:
            x: Input tensor [B, T, D]
            **kwargs: Additional arguments for attention (mask, freqs_cis, etc.)
        """
        # Attention with sandwich norms
        x = self.norm_2(self.attn(self.norm_1(x), **kwargs) + x)
        # MLP with sandwich norms
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x


class HuginnAdapter(nn.Module):
    """
    Huginn-style adapter for combining recurrent state with input embeddings.

    Based on recpre/model_dynamic.py injection_type options:
    - "linear": Linear projection of concatenated [state, input]
    - "ffn": MLP projection of concatenated [state, input]
    - "add": Simple addition
    - "gate": Element-wise multiplication (gating)

    Args:
        d_model: Model dimension
        injection_type: Type of injection ("linear", "ffn", "add", "gate")
        dtype: Data type
        init_device: Initialization device
    """

    def __init__(
        self,
        *,
        d_model: int,
        injection_type: str = "linear",
        intermediate_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.injection_type = injection_type

        if injection_type == "linear":
            self.adapter = nn.Linear(
                d_model * 2, d_model, bias=False,
                dtype=dtype, device=init_device
            )
        elif injection_type == "ffn":
            intermediate = intermediate_size or d_model * 4
            self.adapter = nn.Sequential(
                nn.Linear(d_model * 2, intermediate, bias=False, dtype=dtype, device=init_device),
                nn.SiLU(),
                nn.Linear(intermediate, d_model, bias=False, dtype=dtype, device=init_device),
            )
        elif injection_type in ("add", "gate"):
            self.adapter = nn.Identity()
        else:
            raise ValueError(f"Unknown injection_type: {injection_type}")

    def forward(
        self,
        state: torch.Tensor,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine recurrent state with input embeddings.

        Args:
            state: Recurrent state [B, T, D]
            input_embeds: Input embeddings [B, T, D]

        Returns:
            Combined representation [B, T, D]
        """
        if self.injection_type == "linear" or self.injection_type == "ffn":
            return self.adapter(torch.cat([state, input_embeds], dim=-1))
        elif self.injection_type == "add":
            return state + input_embeds
        elif self.injection_type == "gate":
            return state * input_embeds
        else:
            return self.adapter(state)


class HuginnRecurrentTransformer(nn.Module):
    """
    Huginn-style Recurrent Transformer.

    Architecture: Prelude → Adapter → Core Block (recurrent) → Coda

    Based on recpre/model_dynamic.py RecurrentGPT:
    - Prelude: Initial non-recurrent layers
    - Adapter: Combines recurrent state with input embeddings
    - Core Block: Recurrent layers (repeated N times)
    - Coda: Final non-recurrent layers

    Args:
        prelude_blocks: Initial transformer blocks
        core_blocks: Recurrent transformer blocks
        coda_blocks: Final transformer blocks
        adapter: HuginnAdapter for state injection
        d_model: Model dimension
        mean_recurrence: Expected number of recurrent iterations
        state_init: Initialization for recurrent state ("zero", "normal", "embed")
    """

    def __init__(
        self,
        *,
        prelude_blocks: nn.ModuleList,
        core_blocks: nn.ModuleList,
        coda_blocks: nn.ModuleList,
        adapter: HuginnAdapter,
        d_model: int,
        mean_recurrence: int = 8,
        mean_backprop_depth: int = 4,
        state_init: str = "zero",
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.prelude = prelude_blocks
        self.core_block = core_blocks
        self.coda = coda_blocks
        self.adapter = adapter
        self.d_model = d_model
        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth
        self.state_init = state_init

        # Final norm
        self.ln_f = nn.RMSNorm(d_model, eps=1e-6, device=init_device)

        # Training state
        self._step = 0

    def forward(
        self,
        input_embeds: torch.Tensor,
        num_steps: Optional[int] = None,
        **block_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with recurrent iteration.

        Args:
            input_embeds: Input embeddings [B, T, D]
            num_steps: Override number of recurrent steps
            **block_kwargs: Arguments for transformer blocks

        Returns:
            output: Final hidden states [B, T, D]
            metrics: Training metrics
        """
        metrics: Dict[str, Any] = {
            "num_steps_no_grad": 0,
            "num_steps_with_grad": 0,
        }

        # Prelude
        x = input_embeds
        for block in self.prelude:
            x = block(x, **block_kwargs)

        # Initialize recurrent state
        state = self._initialize_state(input_embeds)

        # Determine iterations
        if num_steps is not None:
            n_no_grad, n_with_grad = num_steps, 0
        elif self.training:
            n_no_grad, n_with_grad = self._sample_iterations()
        else:
            n_no_grad, n_with_grad = self.mean_recurrence, 0

        metrics["num_steps_no_grad"] = n_no_grad
        metrics["num_steps_with_grad"] = n_with_grad

        # Recurrent iterations without gradient
        with torch.no_grad():
            for _ in range(n_no_grad):
                state = self._core_forward(state, x, **block_kwargs)

        # Recurrent iterations with gradient
        for _ in range(n_with_grad):
            state = self._core_forward(state, x, **block_kwargs)

        # Apply final norm to recurrent output
        x = self.ln_f(state)

        # Coda
        for block in self.coda:
            x = block(x, **block_kwargs)

        self._step += 1
        return x, metrics

    def _core_forward(
        self,
        state: torch.Tensor,
        input_embeds: torch.Tensor,
        **block_kwargs,
    ) -> torch.Tensor:
        """Single iteration of core block."""
        # Combine state with input via adapter
        x = self.adapter(state, input_embeds)

        # Apply core blocks
        for block in self.core_block:
            x = block(x, **block_kwargs)

        return x

    def _initialize_state(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Initialize recurrent state."""
        if self.state_init == "zero":
            return torch.zeros_like(input_embeds)
        elif self.state_init == "normal":
            return torch.randn_like(input_embeds)
        elif self.state_init == "embed":
            # Scaled like embeddings
            return torch.randn_like(input_embeds) / math.sqrt(input_embeds.shape[-1])
        else:
            return torch.zeros_like(input_embeds)

    def _sample_iterations(self) -> Tuple[int, int]:
        """Sample number of iterations for training (Huginn-style)."""
        t = max(self.mean_recurrence - self.mean_backprop_depth, 0)
        s = self.mean_backprop_depth

        # Simple uniform sampling (can be extended with other schemes)
        n = torch.randint(0, 2 * t + 1, (1,)).item() if t > 0 else 0
        k = torch.randint(1, 2 * s + 1, (1,)).item()

        return int(n), int(k)


def wrap_transformer_with_etd(
    transformer: nn.Module,
    etd_config: ETDConfig,
) -> nn.Module:
    """
    기존 Transformer를 ETD로 래핑하는 헬퍼 함수.

    이 함수는 기존 Transformer의 blocks를 ETDTransformer로 교체합니다.
    Transformer.forward()를 수정하여 ETD 흐름을 사용하게 합니다.

    Args:
        transformer: 래핑할 기존 Transformer
        etd_config: ETD 설정

    Returns:
        수정된 Transformer (in-place)
    """
    # 기존 블록들 추출
    base_blocks = transformer.blocks
    d_model = transformer.d_model
    dtype = transformer.dtype

    # ETDTransformer 생성
    etd_wrapper = ETDTransformer(
        base_blocks=base_blocks,
        etd_config=etd_config,
        d_model=d_model,
        dtype=dtype,
        init_device="cpu",  # 이미 초기화된 블록 사용
    )

    # 원래 블록들을 빈 ModuleDict로 교체
    transformer.blocks = nn.ModuleDict()

    # ETD wrapper 추가
    transformer.add_module("etd", etd_wrapper)

    # forward 메서드 패치
    original_forward = transformer.forward

    def etd_forward(
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        *,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 입력 준비 (원래 메서드 사용)
        if input_ids is not None:
            (
                input_ids,
                labels,
                all_block_kwargs,
                per_block_kwargs,
                lm_head_kwargs,
            ) = transformer._prepare_inputs(
                input_ids,
                labels,
                **kwargs,
            )
        else:
            all_block_kwargs = {}
            per_block_kwargs = {}
            lm_head_kwargs = {"ignore_index": -100, "loss_reduction": "mean", "return_logits": True}

        # 임베딩 (inputs_embeds가 주어지면 그대로 사용)
        if inputs_embeds is not None:
            h = inputs_embeds
        elif input_ids is not None:
            h = transformer.embeddings(input_ids) if transformer.embeddings is not None else input_ids
        else:
            raise ValueError("input_ids 또는 inputs_embeds 중 하나는 제공해야 합니다.")

        # ETD forward
        h, metrics = transformer.etd(h, **all_block_kwargs)

        # 선택적: 메트릭을 로깅하거나 저장
        if hasattr(transformer, "_etd_metrics"):
            transformer._etd_metrics = metrics

        # LM head
        if transformer.lm_head is not None:
            # lm_head가 단순 nn.Linear인 경우 kwargs 없이 호출
            if isinstance(transformer.lm_head, nn.Linear):
                logits = transformer.lm_head(h)
                loss = None
                if labels is not None:
                    ignore_index = lm_head_kwargs.get("ignore_index", -100)
                    loss = nn.functional.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=ignore_index,
                    )
                return {"loss": loss, "logits": logits}
            else:
                if labels is not None:
                    lm_head_kwargs["labels"] = labels
                return transformer.lm_head(h, **lm_head_kwargs)
        else:
            return h

    # forward 교체
    transformer.forward = etd_forward
    transformer._etd_metrics = {}

    log.info(
        f"Transformer wrapped with ETD: "
        f"E={etd_config.n_encoder_layers}, "
        f"T={etd_config.n_think_layers}, "
        f"D={etd_config.n_decoder_layers}, "
        f"max_iter={etd_config.max_think_iterations}"
    )

    return transformer


# =============================================================================
# Triple-Hybrid Integration
# =============================================================================


@dataclass
class ETDTripleHybridConfig(Config):
    """
    ETD + Triple-Hybrid 통합 설정.

    Triple-Hybrid 백본을 ETD 구조와 결합:
    - Encoder: 주로 DeltaNet (빠른 입력 처리)
    - Think: Mamba-3 + DeltaNet + Attention (깊은 추론)
    - Decoder: Attention 위주 (정밀한 출력)
    """

    # 모델 차원
    hidden_size: int = 2048
    """모델 hidden dimension"""

    # ETD 구조 (레이어 분배)
    n_encoder_layers: int = 8
    """Encoder 레이어 수"""

    n_think_layers: int = 24
    """Think 레이어 수 (핵심 추론)"""

    n_decoder_layers: int = 8
    """Decoder 레이어 수"""

    max_think_iterations: int = 4
    """Think 블록 최대 반복 횟수"""

    # Triple-Hybrid 비율 (각 섹션별로 다르게 설정 가능)
    encoder_layer_pattern: Optional[List[str]] = None
    """Encoder 레이어 패턴 (None이면 자동 생성)"""

    think_layer_pattern: Optional[List[str]] = None
    """Think 레이어 패턴 (None이면 자동 생성)"""

    decoder_layer_pattern: Optional[List[str]] = None
    """Decoder 레이어 패턴 (None이면 자동 생성)"""

    # Encoder: DeltaNet 위주 (빠른 처리)
    encoder_mamba_ratio: float = 0.25
    encoder_deltanet_ratio: float = 0.5
    encoder_attention_ratio: float = 0.25

    # Think: 균형 잡힌 비율 (깊은 추론)
    think_mamba_ratio: float = 0.4
    think_deltanet_ratio: float = 0.4
    think_attention_ratio: float = 0.2

    # Decoder: Attention 위주 (정밀한 출력)
    decoder_mamba_ratio: float = 0.25
    decoder_deltanet_ratio: float = 0.25
    decoder_attention_ratio: float = 0.5

    # Triple-Hybrid 공통 설정
    mamba3_d_state: int = 64
    mamba3_headdim: int = 128
    mamba3_use_complex: bool = True
    deltanet_head_dim: int = 256
    deltanet_num_heads: int = 6
    attention_num_heads: int = 16
    attention_head_dim: int = 128
    hidden_ratio: int = 4
    norm_eps: float = 1e-5

    # Dr.LLM 라우터 (Think 블록용)
    use_layer_router: bool = True
    layer_router: LayerRouterConfig = field(default_factory=LayerRouterConfig)

    # MoDr LoRA 전문가 (Think 블록용)
    use_lora_experts: bool = True
    num_lora_experts: int = 4
    lora_expert: LoRAExpertConfig = field(default_factory=LoRAExpertConfig)

    # 적응형 깊이
    adaptive_depth: bool = True
    confidence_threshold: float = 0.9

    def get_encoder_pattern(self) -> List[str]:
        """Encoder 레이어 패턴 생성"""
        if self.encoder_layer_pattern is not None:
            return self.encoder_layer_pattern

        return self._generate_pattern(
            self.n_encoder_layers,
            self.encoder_mamba_ratio,
            self.encoder_deltanet_ratio,
            self.encoder_attention_ratio,
        )

    def get_think_pattern(self) -> List[str]:
        """Think 레이어 패턴 생성"""
        if self.think_layer_pattern is not None:
            return self.think_layer_pattern

        return self._generate_pattern(
            self.n_think_layers,
            self.think_mamba_ratio,
            self.think_deltanet_ratio,
            self.think_attention_ratio,
        )

    def get_decoder_pattern(self) -> List[str]:
        """Decoder 레이어 패턴 생성"""
        if self.decoder_layer_pattern is not None:
            return self.decoder_layer_pattern

        return self._generate_pattern(
            self.n_decoder_layers,
            self.decoder_mamba_ratio,
            self.decoder_deltanet_ratio,
            self.decoder_attention_ratio,
        )

    def _generate_pattern(
        self,
        n_layers: int,
        mamba_ratio: float,
        deltanet_ratio: float,
        attention_ratio: float,
    ) -> List[str]:
        """비율에 따라 레이어 패턴 생성"""
        n_mamba = int(n_layers * mamba_ratio)
        n_deltanet = int(n_layers * deltanet_ratio)
        n_attention = n_layers - n_mamba - n_deltanet

        # 균일하게 분산 배치
        pattern = []
        mamba_positions = set(range(0, n_layers, max(1, n_layers // max(1, n_mamba)))) if n_mamba > 0 else set()
        attention_positions = set(range(n_layers - 1, -1, -max(1, n_layers // max(1, n_attention)))) if n_attention > 0 else set()

        for i in range(n_layers):
            if i in mamba_positions and len([p for p in pattern if p == "mamba3"]) < n_mamba:
                pattern.append("mamba3")
            elif i in attention_positions and len([p for p in pattern if p == "gated_attention"]) < n_attention:
                pattern.append("gated_attention")
            else:
                pattern.append("gated_deltanet")

        return pattern

    def to_etd_config(self) -> ETDConfig:
        """ETDConfig로 변환"""
        return ETDConfig(
            n_encoder_layers=self.n_encoder_layers,
            n_think_layers=self.n_think_layers,
            n_decoder_layers=self.n_decoder_layers,
            max_think_iterations=self.max_think_iterations,
            adaptive_depth=self.adaptive_depth,
            confidence_threshold=self.confidence_threshold,
            use_layer_router=self.use_layer_router,
            layer_router=self.layer_router,
            use_lora_experts=self.use_lora_experts,
            num_lora_experts=self.num_lora_experts,
            lora_expert=self.lora_expert,
        )


class ETDTripleHybridTransformer(nn.Module):
    """
    ETD + Triple-Hybrid 통합 Transformer.

    구조:
    - Encoder: DeltaNet 위주의 Triple-Hybrid 블록
    - Think: 균형 잡힌 Triple-Hybrid 블록 (반복 실행)
    - Decoder: Attention 위주의 Triple-Hybrid 블록
    """

    def __init__(
        self,
        config: ETDTripleHybridConfig,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.dtype = dtype

        # Import Triple-Hybrid components
        from olmo_core.nn.transformer.triple_hybrid import (
            TripleHybridConfig,
            LayerType,
            TripleHybridTransformer,
        )
        from olmo_core.nn.layer_norm import LayerNormConfig

        layer_norm_config = LayerNormConfig(eps=config.norm_eps)

        # Encoder Triple-Hybrid
        encoder_pattern = config.get_encoder_pattern()
        self.encoder = TripleHybridTransformer(
            config=TripleHybridConfig(
                hidden_size=config.hidden_size,
                num_layers=config.n_encoder_layers,
                layer_pattern=encoder_pattern,
                mamba3_d_state=config.mamba3_d_state,
                mamba3_headdim=config.mamba3_headdim,
                mamba3_use_complex=config.mamba3_use_complex,
                deltanet_head_dim=config.deltanet_head_dim,
                deltanet_num_heads=config.deltanet_num_heads,
                attention_num_heads=config.attention_num_heads,
                attention_head_dim=config.attention_head_dim,
                hidden_ratio=config.hidden_ratio,
                norm_eps=config.norm_eps,
            ),
            layer_norm=layer_norm_config,
            init_device=init_device,
        )

        # Think Triple-Hybrid
        think_pattern = config.get_think_pattern()
        self.think = TripleHybridTransformer(
            config=TripleHybridConfig(
                hidden_size=config.hidden_size,
                num_layers=config.n_think_layers,
                layer_pattern=think_pattern,
                mamba3_d_state=config.mamba3_d_state,
                mamba3_headdim=config.mamba3_headdim,
                mamba3_use_complex=config.mamba3_use_complex,
                deltanet_head_dim=config.deltanet_head_dim,
                deltanet_num_heads=config.deltanet_num_heads,
                attention_num_heads=config.attention_num_heads,
                attention_head_dim=config.attention_head_dim,
                hidden_ratio=config.hidden_ratio,
                norm_eps=config.norm_eps,
            ),
            layer_norm=layer_norm_config,
            init_device=init_device,
        )

        # Decoder Triple-Hybrid
        decoder_pattern = config.get_decoder_pattern()
        self.decoder = TripleHybridTransformer(
            config=TripleHybridConfig(
                hidden_size=config.hidden_size,
                num_layers=config.n_decoder_layers,
                layer_pattern=decoder_pattern,
                mamba3_d_state=config.mamba3_d_state,
                mamba3_headdim=config.mamba3_headdim,
                mamba3_use_complex=config.mamba3_use_complex,
                deltanet_head_dim=config.deltanet_head_dim,
                deltanet_num_heads=config.deltanet_num_heads,
                attention_num_heads=config.attention_num_heads,
                attention_head_dim=config.attention_head_dim,
                hidden_ratio=config.hidden_ratio,
                norm_eps=config.norm_eps,
            ),
            layer_norm=layer_norm_config,
            init_device=init_device,
        )

        # Dr.LLM 스타일 레이어 라우터 (Think 블록용)
        self.layer_routers: Optional[nn.ModuleList] = None
        if config.use_layer_router:
            self.layer_routers = nn.ModuleList([
                config.layer_router.build(
                    d_model=config.hidden_size,
                    dtype=dtype,
                    init_device=init_device,
                )
                for _ in range(config.n_think_layers)
            ])

        # MoDr 스타일 LoRA 전문가 (Think 블록용)
        self.lora_experts: Optional[nn.ModuleList] = None
        self.expert_router: Optional[MoDrExpertRouter] = None
        if config.use_lora_experts:
            self.lora_experts = nn.ModuleList([
                nn.ModuleList([
                    config.lora_expert.build(
                        in_features=config.hidden_size,
                        out_features=config.hidden_size,
                        dtype=dtype,
                        init_device=init_device,
                    )
                    for _ in range(config.num_lora_experts)
                ])
                for _ in range(config.n_think_layers)
            ])
            self.expert_router = MoDrExpertRouter(
                d_model=config.hidden_size,
                num_experts=config.num_lora_experts,
                dtype=dtype,
                init_device=init_device,
            )

        # Think 블록 컨트롤러
        self.think_controller: Optional[ThinkBlockController] = None
        if config.adaptive_depth:
            self.think_controller = ThinkBlockController(
                d_model=config.hidden_size,
                max_iterations=config.max_think_iterations,
                confidence_threshold=config.confidence_threshold,
                dtype=dtype,
                init_device=init_device,
            )

        # 학습 상태
        self._step_count = 0
        self._expert_counts: Optional[torch.Tensor] = None
        if config.use_lora_experts:
            self._expert_counts = torch.zeros(
                config.num_lora_experts, dtype=torch.long
            )

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        ETD + Triple-Hybrid forward pass.

        Args:
            h: Input embeddings (B, T, D)
            attention_mask: Optional attention mask
            **kwargs: Additional arguments

        Returns:
            h: Output hidden states (B, T, D)
            metrics: Training/inference metrics
        """
        metrics: Dict[str, Any] = {
            "think_iterations": 0,
            "router_actions": [],
            "expert_selections": [],
            "early_exit": False,
        }

        # 1. Encoder: 입력 처리
        h, _ = self.encoder(h, attention_mask=attention_mask, **kwargs)

        # 2. Think: 반복적 추론
        max_iterations = self.config.max_think_iterations
        think_iteration = 0

        while think_iteration < max_iterations:
            # Think 블록 통과
            h_think = h
            for layer_idx, layer in enumerate(self.think.layers):
                # Dr.LLM 라우터
                if self.layer_routers is not None and self._step_count >= 1000:
                    router = self.layer_routers[layer_idx]
                    action = router(h_think)
                    action_val = action[0].item()
                    metrics["router_actions"].append(action_val)

                    if action_val == RouterAction.SKIP.value:
                        continue
                    elif action_val == RouterAction.REPEAT.value:
                        h_think = layer(h_think, attention_mask=attention_mask, **kwargs)
                        h_think = layer(h_think, attention_mask=attention_mask, **kwargs)
                    else:
                        h_think = layer(h_think, attention_mask=attention_mask, **kwargs)
                else:
                    h_think = layer(h_think, attention_mask=attention_mask, **kwargs)

                # MoDr LoRA 전문가
                if self.lora_experts is not None and self._step_count >= 500:
                    expert_idx = self.expert_router(h_think)
                    expert_val = expert_idx[0].item()
                    metrics["expert_selections"].append(expert_val)

                    experts = self.lora_experts[layer_idx]
                    lora_delta = experts[expert_val](h_think)
                    h_think = h_think + lora_delta

                    if self._expert_counts is not None:
                        self._expert_counts[expert_val] += 1

            # Final norm from think transformer
            h = self.think.final_norm(h_think)
            think_iteration += 1
            metrics["think_iterations"] = think_iteration

            # 적응형 깊이: 조기 종료
            if self.think_controller is not None and think_iteration < max_iterations:
                should_exit, confidence = self.think_controller.should_exit(h, think_iteration)
                if should_exit:
                    metrics["early_exit"] = True
                    metrics["exit_confidence"] = confidence.mean().item()
                    break

        # 3. Decoder: 출력 생성
        h, _ = self.decoder(h, attention_mask=attention_mask, **kwargs)

        self._step_count += 1
        return h, metrics

    def get_layer_info(self) -> Dict[str, Any]:
        """레이어 구조 정보 반환"""
        return {
            "encoder": {
                "num_layers": self.config.n_encoder_layers,
                "pattern": self.config.get_encoder_pattern(),
            },
            "think": {
                "num_layers": self.config.n_think_layers,
                "pattern": self.config.get_think_pattern(),
                "max_iterations": self.config.max_think_iterations,
            },
            "decoder": {
                "num_layers": self.config.n_decoder_layers,
                "pattern": self.config.get_decoder_pattern(),
            },
            "total_layers": (
                self.config.n_encoder_layers +
                self.config.n_think_layers +
                self.config.n_decoder_layers
            ),
        }


def create_etd_triple_hybrid_model(
    hidden_size: int = 2048,
    n_encoder_layers: int = 8,
    n_think_layers: int = 24,
    n_decoder_layers: int = 8,
    max_think_iterations: int = 4,
    **kwargs,
) -> ETDTripleHybridTransformer:
    """
    ETD + Triple-Hybrid 모델 생성 헬퍼 함수.

    Args:
        hidden_size: 모델 차원
        n_encoder_layers: Encoder 레이어 수
        n_think_layers: Think 레이어 수
        n_decoder_layers: Decoder 레이어 수
        max_think_iterations: Think 최대 반복 횟수
        **kwargs: 추가 설정

    Returns:
        ETDTripleHybridTransformer 인스턴스

    Example:
        ```python
        model = create_etd_triple_hybrid_model(
            hidden_size=2048,
            n_encoder_layers=8,
            n_think_layers=24,
            n_decoder_layers=8,
        )

        # Forward
        output, metrics = model(input_embeds)
        print(f"Think iterations: {metrics['think_iterations']}")
        ```
    """
    config = ETDTripleHybridConfig(
        hidden_size=hidden_size,
        n_encoder_layers=n_encoder_layers,
        n_think_layers=n_think_layers,
        n_decoder_layers=n_decoder_layers,
        max_think_iterations=max_think_iterations,
        **kwargs,
    )
    return ETDTripleHybridTransformer(config)
