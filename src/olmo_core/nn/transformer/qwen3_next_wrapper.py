"""
Qwen3-Next 통합 래퍼

이 모듈은 Qwen3-Next 모델에 다음 기능들을 통합합니다:
- ETD (Encode-Think-Decode): 구조적 레이어 분할
- MoDr (Mixture-of-Depth-Recurrent): LoRA 전문가 분기
- Dr.LLM (Dynamic Layer Routing): Skip/Execute/Repeat 라우터
- MTP (Multi-Token Prediction): 빠른 추론을 위한 멀티토큰 예측

Qwen3-Next 아키텍처:
- Gated DeltaNet (선형 어텐션)
- Gated Attention (표준 어텐션)
- MoE (Mixture of Experts)
- 80B 총 파라미터, 3B 활성화

References:
- Qwen3-Next: https://huggingface.co/Qwen3-Next
- ETD: arXiv:2510.07358
- MoDr: OpenReview
- Dr.LLM: arXiv:2510.12773
- MTP: arXiv:2404.19737
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from olmo_core.config import Config

from .etd import ETDConfig, wrap_transformer_with_etd
from .mtp import MTPConfig, wrap_model_with_mtp

log = logging.getLogger(__name__)

__all__ = [
    "Qwen3NextConfig",
    "Qwen3NextWrapper",
    "create_qwen3_next_model",
]


@dataclass
class Qwen3NextConfig(Config):
    """
    Qwen3-Next + ETD + MTP 통합 설정.

    Qwen3-Next의 Gated DeltaNet + Gated Attention 아키텍처에
    ETD, MoDr, Dr.LLM, MTP를 통합합니다.
    """

    # 기본 모델 설정
    model_name: str = "Qwen/Qwen3-Next-32B-A3B"
    """HuggingFace 모델 이름 또는 경로"""

    vocab_size: int = 151936
    """Qwen3-Next 어휘 크기"""

    d_model: int = 3584
    """모델 hidden dimension"""

    n_layers: int = 48
    """총 레이어 수 (Qwen3-Next-32B-A3B 기준)"""

    # ETD 설정
    use_etd: bool = True
    """ETD 구조 사용 여부"""

    etd_config: ETDConfig = field(default_factory=lambda: ETDConfig(
        n_encoder_layers=12,      # 25%
        n_think_layers=24,        # 50% (반복 추론)
        n_decoder_layers=12,      # 25%
        max_think_iterations=3,   # Qwen3-Next는 더 적은 반복
        adaptive_depth=True,
        confidence_threshold=0.85,
        use_layer_router=True,
        use_lora_experts=True,
        num_lora_experts=4,
        lora_expert=field(default_factory=lambda: {
            "rank": 32,           # 더 높은 rank for 큰 모델
            "alpha": 64.0,
            "dropout": 0.0,
        }),
    ))
    """ETD 설정"""

    # MTP 설정
    use_mtp: bool = True
    """MTP 사용 여부"""

    mtp_config: MTPConfig = field(default_factory=lambda: MTPConfig(
        num_predict_tokens=4,
        mtp_head_type="transformer",
        share_weights_across_steps=True,
        head_num_layers=1,
        head_num_heads=8,
        loss_weight_decay=0.5,
        mtp_loss_weight=0.3,       # Qwen3-Next는 이미 강력하므로 낮은 가중치
        speculation_lookahead=3,
    ))
    """MTP 설정"""

    # 학습 설정
    freeze_backbone: bool = False
    """백본 가중치 동결 여부 (전문가/라우터만 학습)"""

    gradient_checkpointing: bool = True
    """Gradient checkpointing 사용"""

    # 추론 설정
    use_speculative_decoding: bool = True
    """MTP 기반 speculative decoding 사용"""


class Qwen3NextWrapper(nn.Module):
    """
    Qwen3-Next + ETD + MTP 통합 래퍼.

    HuggingFace의 Qwen3-Next 모델을 로드하고
    ETD와 MTP를 적용합니다.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        base_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # 베이스 모델 로드 또는 사용
        if base_model is not None:
            self.model = base_model
        else:
            self.model = self._load_base_model()

        # ETD 적용
        if config.use_etd:
            self._apply_etd()

        # MTP 적용
        if config.use_mtp:
            self._apply_mtp()

        # 백본 동결 (선택적)
        if config.freeze_backbone:
            self._freeze_backbone()

    def _load_base_model(self) -> nn.Module:
        """HuggingFace에서 Qwen3-Next 모델 로드"""
        try:
            from transformers import AutoModelForCausalLM
            log.info(f"Loading Qwen3-Next from: {self.config.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,  # Qwen3-Next는 custom code 필요
            )
            return model
        except ImportError:
            raise ImportError(
                "transformers 라이브러리가 필요합니다. "
                "pip install transformers 로 설치하세요."
            )

    def _apply_etd(self):
        """ETD 래핑 적용"""
        log.info("Applying ETD wrapper to Qwen3-Next...")

        # ETD 설정 검증
        total_layers = (
            self.config.etd_config.n_encoder_layers +
            self.config.etd_config.n_think_layers +
            self.config.etd_config.n_decoder_layers
        )
        if total_layers != self.config.n_layers:
            log.warning(
                f"ETD 레이어 분할({total_layers})이 모델 레이어({self.config.n_layers})와 "
                f"일치하지 않습니다. 자동 조정합니다."
            )
            # 자동 조정: 25% / 50% / 25%
            self.config.etd_config.n_encoder_layers = self.config.n_layers // 4
            self.config.etd_config.n_think_layers = self.config.n_layers // 2
            self.config.etd_config.n_decoder_layers = (
                self.config.n_layers -
                self.config.etd_config.n_encoder_layers -
                self.config.etd_config.n_think_layers
            )

        # HuggingFace 모델의 블록 찾기
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Qwen 스타일
            self._apply_etd_to_hf_model(self.model.model)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT 스타일
            self._apply_etd_to_hf_model(self.model.transformer)
        else:
            log.warning("알 수 없는 모델 구조. ETD 적용을 건너뜁니다.")

    def _apply_etd_to_hf_model(self, backbone: nn.Module):
        """HuggingFace 백본에 ETD 적용"""
        from .etd import ETDTransformer

        # 기존 레이어들을 ModuleDict로 변환
        if hasattr(backbone, "layers"):
            blocks = backbone.layers
            block_type = "layers"
        elif hasattr(backbone, "h"):
            blocks = backbone.h
            block_type = "h"
        else:
            raise ValueError("레이어를 찾을 수 없습니다.")

        # ModuleDict로 변환
        blocks_dict = nn.ModuleDict({
            str(i): block for i, block in enumerate(blocks)
        })

        # d_model 추출
        if hasattr(backbone, "config"):
            d_model = backbone.config.hidden_size
        else:
            d_model = self.config.d_model

        # ETDTransformer 생성
        etd = ETDTransformer(
            base_blocks=blocks_dict,
            etd_config=self.config.etd_config,
            d_model=d_model,
            dtype=torch.bfloat16,
        )

        # 원래 레이어를 ETD로 교체
        backbone.etd = etd
        setattr(backbone, block_type, nn.ModuleList())  # 빈 리스트로 대체

        # forward 패치
        self._patch_forward(backbone, block_type)

    def _patch_forward(self, backbone: nn.Module, block_type: str):
        """HuggingFace 백본의 forward 메서드 패치"""
        original_forward = backbone.forward

        def etd_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple] = None,
            **kwargs,
        ):
            # ETD forward
            h, metrics = backbone.etd(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # 메트릭 저장
            backbone._etd_metrics = metrics

            return h

        # forward 교체 (간단한 버전)
        # 실제 구현에서는 더 복잡한 통합 필요
        backbone._original_forward = original_forward

    def _apply_mtp(self):
        """MTP 래핑 적용"""
        log.info("Applying MTP wrapper to Qwen3-Next...")

        self.model = wrap_model_with_mtp(
            self.model,
            self.config.mtp_config,
            vocab_size=self.config.vocab_size,
        )

    def _freeze_backbone(self):
        """백본 가중치 동결 (ETD/MTP 컴포넌트만 학습)"""
        log.info("Freezing backbone weights...")

        trainable_keywords = [
            "etd", "mtp", "lora", "router", "expert",
        ]

        for name, param in self.model.named_parameters():
            should_train = any(kw in name.lower() for kw in trainable_keywords)
            param.requires_grad = should_train

        # 학습 가능 파라미터 통계
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/total_params:.2f}%)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_speculative: bool = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        텍스트 생성.

        Args:
            input_ids: 입력 토큰 ID
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: Nucleus sampling p
            use_speculative: Speculative decoding 사용 여부

        Returns:
            generated_ids: 생성된 시퀀스
            stats: 생성 통계
        """
        if use_speculative is None:
            use_speculative = self.config.use_speculative_decoding

        if use_speculative and hasattr(self.model, "mtp_decoder"):
            # MTP 기반 speculative decoding
            return self.model.mtp_decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # 기본 생성
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            return outputs, {}

    def get_metrics(self) -> Dict[str, Any]:
        """학습/추론 메트릭 반환"""
        metrics = {}

        # ETD 메트릭
        if hasattr(self.model, "_etd_metrics"):
            metrics["etd"] = self.model._etd_metrics

        # MTP 메트릭
        if hasattr(self.model, "mtp_loss"):
            metrics["mtp_loss"] = self.model.mtp_loss

        return metrics


def create_qwen3_next_model(
    model_name: str = "Qwen/Qwen3-Next-32B-A3B",
    use_etd: bool = True,
    use_mtp: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> Qwen3NextWrapper:
    """
    Qwen3-Next + ETD + MTP 모델 생성 헬퍼.

    Args:
        model_name: HuggingFace 모델 이름
        use_etd: ETD 사용 여부
        use_mtp: MTP 사용 여부
        freeze_backbone: 백본 동결 여부
        **kwargs: 추가 설정

    Returns:
        Qwen3NextWrapper: 통합 모델

    Example:
        ```python
        model = create_qwen3_next_model(
            model_name="Qwen/Qwen3-Next-32B-A3B",
            use_etd=True,
            use_mtp=True,
        )

        # 학습
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # 추론 (speculative decoding)
        generated, stats = model.generate(input_ids, max_new_tokens=100)
        print(f"Speedup: {stats['speedup']:.2f}x")
        ```
    """
    config = Qwen3NextConfig(
        model_name=model_name,
        use_etd=use_etd,
        use_mtp=use_mtp,
        freeze_backbone=freeze_backbone,
    )

    # 추가 설정 적용
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.etd_config, key):
            setattr(config.etd_config, key, value)
        elif hasattr(config.mtp_config, key):
            setattr(config.mtp_config, key, value)

    return Qwen3NextWrapper(config)


# 편의를 위한 프리셋 함수들
def create_qwen3_next_efficient() -> Qwen3NextWrapper:
    """효율성 중시 설정"""
    return create_qwen3_next_model(
        use_etd=True,
        use_mtp=True,
        freeze_backbone=True,  # 빠른 파인튜닝
        max_think_iterations=2,
        num_predict_tokens=3,
    )


def create_qwen3_next_performance() -> Qwen3NextWrapper:
    """성능 중시 설정"""
    return create_qwen3_next_model(
        use_etd=True,
        use_mtp=True,
        freeze_backbone=False,
        max_think_iterations=5,
        num_predict_tokens=4,
        num_lora_experts=8,
    )
