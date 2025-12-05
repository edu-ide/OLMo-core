"""
DNA Transfer: 모델 가중치 전이 기법 모음

이 모듈은 AI 업계에서 사용하는 주요 가중치 전이 기법들을 구현합니다:

1. Upcycling (Dense → MoE): 밀집 모델을 MoE로 변환
2. Net2Net / Model Growth: 뉴런 분열, 레이어 삽입으로 모델 확장
3. Model Slicing (Nemotron style): 큰 모델에서 작은 모델 추출
4. Evolutionary Model Merge: 여러 모델의 레이어를 진화적으로 병합

References:
- Upcycling: arXiv:2212.05055 (Sparse Upcycling)
- Solar DUS: arXiv:2312.15166 (Depth Up-Scaling)
- Net2Net: arXiv:1511.05641
- Model Slicing: NVIDIA Nemotron Elastic
- Evolutionary Merge: Sakana AI (arXiv:2403.13187)
"""

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import Config

log = logging.getLogger(__name__)

__all__ = [
    # Upcycling
    "UpcyclingConfig",
    "upcycle_dense_to_moe",
    "depth_upscale",
    # iDUS (Interlocking Depth-Up-Scaling)
    "IDUSType",
    "get_idus_layer_arrangement",
    "DepthUpscaleConfig",
    # Model Growth
    "ModelGrowthConfig",
    "GrowthMethod",
    "grow_model_width",
    "grow_model_depth",
    # Model Slicing
    "ModelSlicingConfig",
    "slice_model",
    "create_model_family",
    # Evolutionary Merge (with mergekit-style sparsification)
    "EvolutionaryMergeConfig",
    "evolutionary_merge",
    "MergeStrategy",
    "SparsificationMethod",
    "ConsensusMethod",
    "sparsify_delta",
    "get_consensus_mask",
]


# =============================================================================
# 1. UPCYCLING: Dense → MoE 변환
# =============================================================================

@dataclass
class UpcyclingConfig(Config):
    """
    Dense → MoE 업사이클링 설정.

    밀집 모델의 FFN을 복제하여 MoE 전문가로 변환합니다.
    Qwen, DeepSeek, Mistral 등이 사용하는 방식입니다.
    """

    num_experts: int = 8
    """생성할 전문가 수"""

    num_experts_per_token: int = 2
    """토큰당 활성화되는 전문가 수"""

    expert_init_noise: float = 0.01
    """전문가 초기화 시 추가할 노이즈 (다양성 확보)"""

    router_init_std: float = 0.02
    """라우터 초기화 표준편차"""

    copy_all_ffn_weights: bool = True
    """True: 모든 전문가에 FFN 복사, False: 첫 번째만 복사"""

    jitter_noise: float = 0.0
    """추론 시 라우터 지터 노이즈"""

    auxiliary_loss_weight: float = 0.01
    """Load balancing auxiliary loss 가중치"""


class MoERouter(nn.Module):
    """MoE 라우터 (Top-K 게이팅)"""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        jitter_noise: float = 0.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(
            d_model, num_experts, bias=False, dtype=dtype, device=init_device
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)

        Returns:
            router_probs: (B, T, num_experts)
            expert_indices: (B, T, top_k)
            expert_weights: (B, T, top_k)
        """
        # 학습 시 지터 노이즈 추가
        if self.training and self.jitter_noise > 0:
            x = x * (1 + torch.randn_like(x) * self.jitter_noise)

        router_logits = self.gate(x)  # (B, T, E)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K 선택
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return router_probs, expert_indices, expert_weights


class MoEFFN(nn.Module):
    """MoE FFN 레이어 (업사이클된)"""

    def __init__(
        self,
        experts: nn.ModuleList,
        router: MoERouter,
        auxiliary_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.num_experts = len(experts)
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self._aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        router_probs, expert_indices, expert_weights = self.router(x)

        # Auxiliary loss 계산 (load balancing)
        if self.training:
            self._aux_loss = self._compute_aux_loss(router_probs)

        # 전문가 출력 계산
        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            # 이 전문가가 선택된 위치 찾기
            mask = (expert_indices == i).any(dim=-1)  # (B, T)
            if not mask.any():
                continue

            # 선택된 토큰들만 처리
            selected_x = x[mask]  # (N, D)
            expert_out = expert(selected_x)  # (N, D)

            # 가중치 계산
            weight_mask = (expert_indices == i).float()  # (B, T, top_k)
            weights = (expert_weights * weight_mask).sum(dim=-1)  # (B, T)

            # 출력에 기여
            output[mask] += expert_out * weights[mask].unsqueeze(-1)

        return output

    def _compute_aux_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Load balancing auxiliary loss"""
        # 전문가별 평균 라우팅 확률
        expert_probs = router_probs.mean(dim=[0, 1])  # (E,)

        # 균등 분포와의 차이
        uniform = torch.ones_like(expert_probs) / self.num_experts
        aux_loss = F.mse_loss(expert_probs, uniform) * self.auxiliary_loss_weight

        return aux_loss

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss


def upcycle_dense_to_moe(
    model: nn.Module,
    config: UpcyclingConfig,
    ffn_module_names: List[str] = None,
) -> nn.Module:
    """
    Dense 모델을 MoE 모델로 업사이클링.

    Args:
        model: 원본 밀집 모델
        config: 업사이클링 설정
        ffn_module_names: 변환할 FFN 모듈 이름 패턴 (None이면 자동 탐지)

    Returns:
        MoE로 변환된 모델

    Example:
        ```python
        config = UpcyclingConfig(num_experts=8, num_experts_per_token=2)
        moe_model = upcycle_dense_to_moe(dense_model, config)
        ```
    """
    log.info(f"Upcycling dense model to MoE with {config.num_experts} experts")

    # FFN 모듈 탐지
    if ffn_module_names is None:
        ffn_module_names = _detect_ffn_modules(model)

    log.info(f"Found {len(ffn_module_names)} FFN modules to upcycle")

    # 각 FFN을 MoE로 변환
    for name in ffn_module_names:
        _replace_ffn_with_moe(model, name, config)

    # 모델에 auxiliary loss 수집 메서드 추가
    def get_moe_aux_loss(self):
        total_aux_loss = 0
        count = 0
        for module in self.modules():
            if isinstance(module, MoEFFN) and module.aux_loss is not None:
                total_aux_loss += module.aux_loss
                count += 1
        return total_aux_loss / max(count, 1)

    model.get_moe_aux_loss = lambda: get_moe_aux_loss(model)

    return model


def _detect_ffn_modules(model: nn.Module) -> List[str]:
    """FFN 모듈 자동 탐지"""
    ffn_names = []
    ffn_keywords = ["mlp", "ffn", "feed_forward", "ff"]

    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ffn_keywords):
            # Sequential이나 FFN 구조인지 확인
            if isinstance(module, nn.Sequential) or _is_ffn_like(module):
                ffn_names.append(name)

    return ffn_names


def _is_ffn_like(module: nn.Module) -> bool:
    """FFN 구조인지 확인"""
    children = list(module.children())
    if len(children) >= 2:
        # 첫 번째와 마지막이 Linear인지
        first_linear = isinstance(children[0], nn.Linear)
        last_linear = isinstance(children[-1], nn.Linear)
        return first_linear and last_linear
    return False


def _replace_ffn_with_moe(
    model: nn.Module,
    ffn_name: str,
    config: UpcyclingConfig,
):
    """단일 FFN을 MoE로 교체"""
    # 모듈 경로 파싱
    parts = ffn_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    ffn_attr = parts[-1]
    original_ffn = getattr(parent, ffn_attr)

    # d_model 추출
    if hasattr(original_ffn, "in_features"):
        d_model = original_ffn.in_features
    else:
        # 첫 번째 Linear 레이어에서 추출
        for child in original_ffn.modules():
            if isinstance(child, nn.Linear):
                d_model = child.in_features
                break

    # 전문가 생성 (FFN 복제)
    experts = nn.ModuleList()
    for i in range(config.num_experts):
        expert = copy.deepcopy(original_ffn)

        # 다양성을 위한 노이즈 추가
        if config.expert_init_noise > 0 and i > 0:
            with torch.no_grad():
                for param in expert.parameters():
                    param.add_(torch.randn_like(param) * config.expert_init_noise)

        experts.append(expert)

    # 라우터 생성
    router = MoERouter(
        d_model=d_model,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
        jitter_noise=config.jitter_noise,
    )

    # 라우터 초기화
    nn.init.normal_(router.gate.weight, std=config.router_init_std)

    # MoE FFN으로 교체
    moe_ffn = MoEFFN(
        experts=experts,
        router=router,
        auxiliary_loss_weight=config.auxiliary_loss_weight,
    )

    setattr(parent, ffn_attr, moe_ffn)
    log.info(f"Upcycled {ffn_name} to MoE with {config.num_experts} experts")


class IDUSType(Enum):
    """
    iDUS (Interlocking Depth-Up-Scaling) 유형.

    Reference: gauss5930/iDUS (https://github.com/gauss5930/iDUS)

    - ORIGINAL: Solar DUS - 레이어 범위를 단순 연결 (0-24) + (8-32)
    - INTERLOCKED_1: 각 레이어를 교차 삽입 (A0, B0, A1, B1, ...)
    - INTERLOCKED_8: 8개 그룹 단위로 교차 (0-8) + (8-16)*2 + (16-24)*2 + (24-32)
    """
    ORIGINAL = "original"
    INTERLOCKED_1 = "interlocked_1"
    INTERLOCKED_8 = "interlocked_8"


def get_idus_layer_arrangement(
    n_layers: int,
    idus_type: IDUSType = IDUSType.ORIGINAL,
    overlap_start: int = 8,
) -> List[int]:
    """
    iDUS 스타일 레이어 배열 생성.

    Reference: gauss5930/iDUS/src/iDUS.py

    Args:
        n_layers: 원본 모델의 레이어 수 (예: 32)
        idus_type: iDUS 유형
        overlap_start: 중첩 시작 레이어 (기본 8, Solar 방식)

    Returns:
        새로운 레이어 배열 (인덱스 리스트)

    Example:
        ```python
        # Original DUS (Solar 10.7B style): 32 → 48 layers
        arrangement = get_idus_layer_arrangement(32, IDUSType.ORIGINAL)
        # [0, 1, ..., 23, 8, 9, ..., 31]

        # iDUS-1layer: 32 → 48 layers (interleaved)
        arrangement = get_idus_layer_arrangement(32, IDUSType.INTERLOCKED_1)
        # [0, 8, 1, 9, 2, 10, ..., 23, 31]
        ```
    """
    if idus_type == IDUSType.ORIGINAL:
        # Solar DUS: 첫 부분 + 중첩 시작부터 끝까지
        # Reference: iDUS.py line 38
        first_range = list(range(0, n_layers - overlap_start))
        second_range = list(range(overlap_start, n_layers))
        return first_range + second_range

    elif idus_type == IDUSType.INTERLOCKED_1:
        # iDUS-1layer: 각 레이어를 하나씩 교차
        # Reference: iDUS.py lines 39-45
        layer_arrangement = []
        layer_A = list(range(0, n_layers - overlap_start))
        layer_B = list(range(overlap_start, n_layers))
        for i in range(len(layer_A)):
            layer_arrangement.append(layer_A[i])
            layer_arrangement.append(layer_B[i])
        return layer_arrangement

    elif idus_type == IDUSType.INTERLOCKED_8:
        # iDUS-8layer: 8개 그룹 단위로 교차
        # Reference: iDUS.py line 48
        # (0-8) + (8-16)*2 + (16-24)*2 + (24-32)
        group_size = overlap_start  # 8
        n_groups = n_layers // group_size  # 4 for 32 layers

        arrangement = list(range(0, group_size))  # First group
        for g in range(1, n_groups - 1):
            start = g * group_size
            end = start + group_size
            arrangement.extend(list(range(start, end)) * 2)  # Repeat middle groups
        arrangement.extend(list(range(n_layers - group_size, n_layers)))  # Last group

        return arrangement

    else:
        raise ValueError(f"Unknown iDUS type: {idus_type}")


@dataclass
class DepthUpscaleConfig(Config):
    """
    Depth Up-Scaling (DUS) 설정.

    Solar 스타일: 레이어를 복제하여 모델 깊이 증가.

    Reference: gauss5930/iDUS for interlocking patterns.
    """

    scale_factor: int = 2
    """깊이 배율 (2x = 레이어 수 2배)"""

    insertion_method: str = "interleave"
    """삽입 방식: 'interleave' (교차 삽입), 'stack' (위에 쌓기), 'idus_original', 'idus_interlocked_1', 'idus_interlocked_8'"""

    init_noise: float = 0.001
    """복제된 레이어의 초기화 노이즈"""

    skip_embed_and_head: bool = True
    """임베딩과 LM head는 복제하지 않음"""

    idus_overlap_start: int = 8
    """iDUS 모드에서 중첩 시작 레이어 (Solar는 8 사용)"""


def depth_upscale(
    model: nn.Module,
    config: DepthUpscaleConfig,
    layer_container_name: str = "layers",
) -> nn.Module:
    """
    모델 깊이 확장 (Solar DUS 스타일).

    레이어를 복제하여 모델의 깊이를 늘립니다.

    Args:
        model: 원본 모델
        config: DUS 설정
        layer_container_name: 레이어 컨테이너 이름

    Returns:
        깊이가 확장된 모델

    Example:
        ```python
        config = DepthUpscaleConfig(scale_factor=2, insertion_method="interleave")
        scaled_model = depth_upscale(model, config)
        # 16 layers → 32 layers
        ```
    """
    log.info(f"Depth upscaling with factor {config.scale_factor}")

    # 레이어 컨테이너 찾기
    layers = _find_layer_container(model, layer_container_name)
    if layers is None:
        raise ValueError(f"Cannot find layer container: {layer_container_name}")

    original_layers = list(layers)
    n_original = len(original_layers)
    n_target = n_original * config.scale_factor

    log.info(f"Scaling from {n_original} to {n_target} layers")

    # 새 레이어 리스트 생성
    new_layers = []

    if config.insertion_method == "interleave":
        # 교차 삽입: L1, L1', L2, L2', ...
        for layer in original_layers:
            new_layers.append(layer)
            for _ in range(config.scale_factor - 1):
                cloned = copy.deepcopy(layer)
                if config.init_noise > 0:
                    _add_noise_to_module(cloned, config.init_noise)
                new_layers.append(cloned)

    elif config.insertion_method == "stack":
        # 위에 쌓기: L1, L2, ..., Ln, L1', L2', ..., Ln'
        new_layers.extend(original_layers)
        for _ in range(config.scale_factor - 1):
            for layer in original_layers:
                cloned = copy.deepcopy(layer)
                if config.init_noise > 0:
                    _add_noise_to_module(cloned, config.init_noise)
                new_layers.append(cloned)

    elif config.insertion_method.startswith("idus_"):
        # iDUS 스타일 배열
        # Reference: gauss5930/iDUS
        idus_type_str = config.insertion_method.replace("idus_", "")
        idus_type = IDUSType(idus_type_str)

        arrangement = get_idus_layer_arrangement(
            n_original,
            idus_type=idus_type,
            overlap_start=config.idus_overlap_start,
        )

        log.info(f"iDUS arrangement ({idus_type.value}): {n_original} → {len(arrangement)} layers")

        for new_idx, old_idx in enumerate(arrangement):
            if old_idx < len(original_layers):
                if new_idx < n_original:
                    # 원본 레이어 재사용
                    new_layers.append(original_layers[old_idx])
                else:
                    # 복제된 레이어
                    cloned = copy.deepcopy(original_layers[old_idx])
                    if config.init_noise > 0:
                        _add_noise_to_module(cloned, config.init_noise)
                    new_layers.append(cloned)

    # 레이어 교체
    _replace_layers(model, layer_container_name, new_layers)

    log.info(f"Depth upscaling complete: {n_original} → {len(new_layers)} layers")

    return model


def _find_layer_container(
    model: nn.Module, name: str
) -> Optional[nn.ModuleList]:
    """레이어 컨테이너 찾기"""
    for attr_name, module in model.named_modules():
        if name in attr_name and isinstance(module, (nn.ModuleList, nn.Sequential)):
            return module
    return None


def _add_noise_to_module(module: nn.Module, noise_std: float):
    """모듈의 파라미터에 노이즈 추가"""
    with torch.no_grad():
        for param in module.parameters():
            param.add_(torch.randn_like(param) * noise_std)


def _replace_layers(model: nn.Module, container_name: str, new_layers: List[nn.Module]):
    """레이어 컨테이너 교체"""
    parts = container_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    if hasattr(parent, parts[-1]):
        setattr(parent, parts[-1], nn.ModuleList(new_layers))
    else:
        # 중첩된 구조 탐색
        for name, module in parent.named_modules():
            if parts[-1] in name and isinstance(module, (nn.ModuleList, nn.Sequential)):
                # 부모 찾기
                parent_path = name.rsplit(".", 1)[0] if "." in name else ""
                if parent_path:
                    container_parent = parent
                    for p in parent_path.split("."):
                        container_parent = getattr(container_parent, p)
                    setattr(container_parent, name.split(".")[-1], nn.ModuleList(new_layers))
                break


# =============================================================================
# 2. NET2NET / MODEL GROWTH
# =============================================================================

class GrowthMethod(Enum):
    """모델 성장 방식"""
    NET2WIDER = "net2wider"    # 뉴런 분열 (너비 확장)
    NET2DEEPER = "net2deeper"  # 레이어 삽입 (깊이 확장)
    BERT2BERT = "bert2bert"    # BERT 스타일 성장


@dataclass
class ModelGrowthConfig(Config):
    """
    Net2Net / 모델 성장 설정.

    작은 모델에서 큰 모델로 지식을 보존하며 확장합니다.
    """

    method: GrowthMethod = GrowthMethod.NET2WIDER
    """성장 방식"""

    # Net2Wider 설정
    width_multiplier: float = 2.0
    """너비 배율 (hidden size 확장)"""

    # Net2Deeper 설정
    num_new_layers: int = 4
    """추가할 레이어 수"""

    insertion_positions: List[int] = field(default_factory=list)
    """레이어 삽입 위치 (빈 리스트면 균등 분배)"""

    # 공통 설정
    preserve_output: bool = True
    """초기 출력값 보존 (function-preserving)"""

    noise_std: float = 0.01
    """초기화 노이즈"""


def grow_model_width(
    model: nn.Module,
    config: ModelGrowthConfig,
    target_hidden_size: Optional[int] = None,
) -> nn.Module:
    """
    Net2Wider: 모델 너비 확장.

    뉴런을 분열시켜 hidden dimension을 늘립니다.
    수학적으로 초기 출력값을 보존합니다.

    Args:
        model: 원본 모델
        config: 성장 설정
        target_hidden_size: 목표 hidden size (None이면 multiplier 사용)

    Returns:
        너비가 확장된 모델
    """
    log.info(f"Growing model width by {config.width_multiplier}x")

    # 현재 hidden size 탐지
    current_hidden_size = _detect_hidden_size(model)
    if target_hidden_size is None:
        target_hidden_size = int(current_hidden_size * config.width_multiplier)

    log.info(f"Expanding hidden size: {current_hidden_size} → {target_hidden_size}")

    # 모든 Linear 레이어 찾기 및 확장
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _widen_linear(
                model, name, module,
                current_hidden_size, target_hidden_size,
                config.preserve_output, config.noise_std
            )

    # LayerNorm, Embedding 등도 확장
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            _widen_norm(model, name, module, target_hidden_size)
        elif isinstance(module, nn.Embedding):
            _widen_embedding(model, name, module, target_hidden_size)

    return model


def _detect_hidden_size(model: nn.Module) -> int:
    """모델의 hidden size 탐지"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            return module.in_features
        if hasattr(module, "hidden_size"):
            return module.hidden_size
        if hasattr(module, "d_model"):
            return module.d_model
    raise ValueError("Cannot detect hidden size")


def _widen_linear(
    model: nn.Module,
    name: str,
    module: nn.Linear,
    old_size: int,
    new_size: int,
    preserve_output: bool,
    noise_std: float,
):
    """Linear 레이어 너비 확장"""
    in_features = module.in_features
    out_features = module.out_features

    # 입력/출력 중 old_size와 일치하는 것 확장
    new_in = new_size if in_features == old_size else in_features
    new_out = new_size if out_features == old_size else out_features

    if new_in == in_features and new_out == out_features:
        return  # 확장 불필요

    # 새 Linear 생성
    new_linear = nn.Linear(
        new_in, new_out,
        bias=module.bias is not None,
        dtype=module.weight.dtype,
        device=module.weight.device,
    )

    with torch.no_grad():
        # Net2Wider 가중치 복사
        if preserve_output:
            # 뉴런 분열: 기존 뉴런을 복제하고 가중치를 나눔
            if new_in > in_features:
                # 입력 확장: 뉴런 복제
                ratio = new_in // in_features
                expanded_weight = module.weight.repeat(1, ratio)[:, :new_in]
                expanded_weight = expanded_weight / ratio  # 출력 보존
                new_linear.weight[:out_features, :] = expanded_weight

            if new_out > out_features:
                # 출력 확장: 뉴런 분열
                ratio = new_out // out_features
                for i in range(ratio):
                    start = i * out_features
                    end = start + out_features
                    if end <= new_out:
                        new_linear.weight[start:end, :in_features] = module.weight / ratio

            if module.bias is not None:
                if new_out > out_features:
                    ratio = new_out // out_features
                    new_linear.bias[:out_features] = module.bias / ratio
                else:
                    new_linear.bias[:] = module.bias

        # 노이즈 추가
        if noise_std > 0:
            new_linear.weight.add_(torch.randn_like(new_linear.weight) * noise_std)

    # 모듈 교체
    _set_module_by_name(model, name, new_linear)


def _widen_norm(model: nn.Module, name: str, module: nn.Module, new_size: int):
    """Norm 레이어 확장"""
    if isinstance(module, nn.LayerNorm):
        new_norm = nn.LayerNorm(
            new_size,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
            dtype=module.weight.dtype if module.weight is not None else torch.float32,
            device=module.weight.device if module.weight is not None else "cpu",
        )
        if module.weight is not None:
            with torch.no_grad():
                # 기존 값 복사 및 확장
                old_size = module.weight.shape[0]
                new_norm.weight[:old_size] = module.weight
                new_norm.weight[old_size:] = module.weight.mean()
                if module.bias is not None:
                    new_norm.bias[:old_size] = module.bias
                    new_norm.bias[old_size:] = 0
        _set_module_by_name(model, name, new_norm)


def _widen_embedding(model: nn.Module, name: str, module: nn.Embedding, new_size: int):
    """Embedding 레이어 확장"""
    if module.embedding_dim == new_size:
        return

    new_embed = nn.Embedding(
        module.num_embeddings,
        new_size,
        padding_idx=module.padding_idx,
        dtype=module.weight.dtype,
        device=module.weight.device,
    )

    with torch.no_grad():
        old_size = module.embedding_dim
        new_embed.weight[:, :old_size] = module.weight
        # 새 차원은 작은 랜덤 값으로 초기화
        new_embed.weight[:, old_size:] = torch.randn(
            module.num_embeddings, new_size - old_size,
            dtype=module.weight.dtype, device=module.weight.device
        ) * 0.02

    _set_module_by_name(model, name, new_embed)


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """이름으로 모듈 설정"""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def grow_model_depth(
    model: nn.Module,
    config: ModelGrowthConfig,
    layer_container_name: str = "layers",
) -> nn.Module:
    """
    Net2Deeper: 모델 깊이 확장.

    Identity 레이어를 삽입하여 깊이를 늘립니다.
    초기에는 기존 모델과 동일한 출력을 유지합니다.

    Args:
        model: 원본 모델
        config: 성장 설정
        layer_container_name: 레이어 컨테이너 이름

    Returns:
        깊이가 확장된 모델
    """
    log.info(f"Growing model depth by {config.num_new_layers} layers")

    # 레이어 컨테이너 찾기
    layers = _find_layer_container(model, layer_container_name)
    if layers is None:
        raise ValueError(f"Cannot find layer container: {layer_container_name}")

    original_layers = list(layers)
    n_original = len(original_layers)

    # 삽입 위치 결정
    if config.insertion_positions:
        positions = sorted(config.insertion_positions, reverse=True)
    else:
        # 균등 분배
        step = n_original // (config.num_new_layers + 1)
        positions = [step * (i + 1) for i in range(config.num_new_layers)]
        positions = sorted(positions, reverse=True)

    log.info(f"Inserting layers at positions: {positions}")

    # 새 레이어 삽입
    new_layers = list(original_layers)
    for pos in positions:
        if pos < len(new_layers):
            # 해당 위치의 레이어를 복제하여 Identity로 초기화
            template_layer = new_layers[pos]
            identity_layer = _create_identity_layer(template_layer, config.noise_std)
            new_layers.insert(pos + 1, identity_layer)

    # 레이어 교체
    _replace_layers(model, layer_container_name, new_layers)

    log.info(f"Depth growth complete: {n_original} → {len(new_layers)} layers")

    return model


def _create_identity_layer(template: nn.Module, noise_std: float) -> nn.Module:
    """Identity 연산을 수행하는 레이어 생성 (Net2Deeper)"""
    identity_layer = copy.deepcopy(template)

    with torch.no_grad():
        for name, module in identity_layer.named_modules():
            if isinstance(module, nn.Linear):
                # Attention의 O projection이나 FFN의 출력은 Identity로
                if module.in_features == module.out_features:
                    nn.init.eye_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # 입출력 크기가 다르면 작은 값으로 초기화
                    nn.init.normal_(module.weight, std=0.001)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                # 노이즈 추가
                if noise_std > 0:
                    module.weight.add_(torch.randn_like(module.weight) * noise_std)

    return identity_layer


# =============================================================================
# 3. MODEL SLICING (Nemotron Elastic Style)
# =============================================================================

@dataclass
class ModelSlicingConfig(Config):
    """
    모델 슬라이싱 설정 (Nemotron Elastic 스타일).

    큰 부모 모델에서 작은 자식 모델을 추출합니다.
    """

    target_layers: Optional[int] = None
    """목표 레이어 수 (None이면 자동)"""

    target_hidden_size: Optional[int] = None
    """목표 hidden size (None이면 유지)"""

    layer_selection: str = "importance"
    """레이어 선택 방식: 'importance', 'uniform', 'first', 'last'"""

    importance_metric: str = "gradient"
    """중요도 측정 방식: 'gradient', 'activation', 'weight_norm'"""

    preserve_first_last: bool = True
    """첫 번째와 마지막 레이어 보존"""

    calibration_data: Optional[torch.Tensor] = None
    """중요도 계산용 캘리브레이션 데이터"""


def slice_model(
    model: nn.Module,
    config: ModelSlicingConfig,
    layer_container_name: str = "layers",
) -> nn.Module:
    """
    모델 슬라이싱: 큰 모델에서 작은 모델 추출.

    NVIDIA Nemotron Elastic 스타일로 중요한 레이어만 선택합니다.

    Args:
        model: 원본 큰 모델
        config: 슬라이싱 설정
        layer_container_name: 레이어 컨테이너 이름

    Returns:
        슬라이싱된 작은 모델
    """
    log.info("Slicing model to smaller variant")

    # 레이어 컨테이너 찾기
    layers = _find_layer_container(model, layer_container_name)
    if layers is None:
        raise ValueError(f"Cannot find layer container: {layer_container_name}")

    original_layers = list(layers)
    n_original = len(original_layers)

    if config.target_layers is None:
        config.target_layers = n_original // 2

    log.info(f"Slicing from {n_original} to {config.target_layers} layers")

    # 레이어 선택
    if config.layer_selection == "importance":
        selected_indices = _select_layers_by_importance(
            model, original_layers, config
        )
    elif config.layer_selection == "uniform":
        step = n_original / config.target_layers
        selected_indices = [int(i * step) for i in range(config.target_layers)]
    elif config.layer_selection == "first":
        selected_indices = list(range(config.target_layers))
    elif config.layer_selection == "last":
        selected_indices = list(range(n_original - config.target_layers, n_original))
    else:
        raise ValueError(f"Unknown layer selection: {config.layer_selection}")

    # 첫/마지막 레이어 보존
    if config.preserve_first_last:
        if 0 not in selected_indices:
            selected_indices = [0] + selected_indices[:-1]
        if n_original - 1 not in selected_indices:
            selected_indices = selected_indices[:-1] + [n_original - 1]

    selected_indices = sorted(set(selected_indices))[:config.target_layers]

    log.info(f"Selected layer indices: {selected_indices}")

    # 선택된 레이어만 유지
    sliced_layers = [original_layers[i] for i in selected_indices]
    _replace_layers(model, layer_container_name, sliced_layers)

    # Hidden size 슬라이싱 (선택적)
    if config.target_hidden_size is not None:
        model = _slice_hidden_size(model, config.target_hidden_size)

    log.info(f"Model slicing complete: {n_original} → {len(sliced_layers)} layers")

    return model


def _select_layers_by_importance(
    model: nn.Module,
    layers: List[nn.Module],
    config: ModelSlicingConfig,
) -> List[int]:
    """중요도 기반 레이어 선택"""
    n_layers = len(layers)
    importances = []

    if config.importance_metric == "weight_norm":
        # 가중치 norm으로 중요도 측정
        for i, layer in enumerate(layers):
            total_norm = 0
            for param in layer.parameters():
                total_norm += param.norm().item()
            importances.append((i, total_norm))

    elif config.importance_metric == "gradient":
        # Gradient magnitude로 중요도 측정 (캘리브레이션 데이터 필요)
        if config.calibration_data is None:
            log.warning("No calibration data, falling back to weight_norm")
            return _select_layers_by_importance(
                model, layers,
                ModelSlicingConfig(
                    target_layers=config.target_layers,
                    importance_metric="weight_norm"
                )
            )

        # Forward-backward로 gradient 수집
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)

        output = model(config.calibration_data)
        if hasattr(output, "logits"):
            output = output.logits
        loss = output.sum()
        loss.backward()

        for i, layer in enumerate(layers):
            total_grad = 0
            for param in layer.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().sum().item()
            importances.append((i, total_grad))

        model.zero_grad()

    else:
        # 기본: 균등 중요도
        importances = [(i, 1.0) for i in range(n_layers)]

    # 중요도 순으로 정렬하여 선택
    importances.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in importances[:config.target_layers]]

    return selected


def _slice_hidden_size(model: nn.Module, target_size: int) -> nn.Module:
    """Hidden size 슬라이싱 (채널 프루닝)"""
    log.info(f"Slicing hidden size to {target_size}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.in_features > target_size or module.out_features > target_size:
                new_in = min(module.in_features, target_size)
                new_out = min(module.out_features, target_size)

                new_linear = nn.Linear(
                    new_in, new_out,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                    device=module.weight.device,
                )

                with torch.no_grad():
                    # 중요한 채널만 선택 (L2 norm 기준)
                    if module.out_features > target_size:
                        norms = module.weight.norm(dim=1)
                        _, indices = norms.topk(new_out)
                        new_linear.weight[:] = module.weight[indices, :new_in]
                        if module.bias is not None:
                            new_linear.bias[:] = module.bias[indices]
                    else:
                        new_linear.weight[:] = module.weight[:new_out, :new_in]
                        if module.bias is not None:
                            new_linear.bias[:] = module.bias[:new_out]

                _set_module_by_name(model, name, new_linear)

    return model


def create_model_family(
    base_model: nn.Module,
    sizes: List[Tuple[int, int]],  # [(layers, hidden_size), ...]
    layer_container_name: str = "layers",
) -> Dict[str, nn.Module]:
    """
    모델 패밀리 생성 (Nemotron Elastic 스타일).

    하나의 큰 모델에서 여러 크기의 모델 추출.

    Args:
        base_model: 기본 큰 모델
        sizes: 목표 크기 리스트 [(layers, hidden_size), ...]
        layer_container_name: 레이어 컨테이너 이름

    Returns:
        Dict[name, model]: 크기별 모델 딕셔너리

    Example:
        ```python
        family = create_model_family(
            base_12b_model,
            sizes=[(48, 4096), (32, 3072), (24, 2048)],
        )
        model_6b = family["32L_3072D"]
        ```
    """
    log.info(f"Creating model family with {len(sizes)} variants")

    family = {}

    for n_layers, hidden_size in sizes:
        variant_name = f"{n_layers}L_{hidden_size}D"
        log.info(f"Creating variant: {variant_name}")

        # 모델 복제
        variant = copy.deepcopy(base_model)

        # 슬라이싱 설정
        config = ModelSlicingConfig(
            target_layers=n_layers,
            target_hidden_size=hidden_size,
            layer_selection="importance",
            importance_metric="weight_norm",
        )

        # 슬라이싱 적용
        variant = slice_model(variant, config, layer_container_name)

        family[variant_name] = variant

    return family


# =============================================================================
# 4. EVOLUTIONARY MODEL MERGE (with mergekit-style sparsification)
# =============================================================================

class SparsificationMethod(Enum):
    """
    Sparsification methods for model merging.

    Reference: mergekit/sparsify.py, mergekit/merge_methods/generalized_task_arithmetic.py
    """
    NONE = "none"
    MAGNITUDE = "magnitude"          # Keep top-k by absolute value
    RANDOM = "random"                # Random pruning
    MAGNITUDE_OUTLIERS = "magnitude_outliers"  # Outlier-based (uses gamma)
    DELLA_MAGPRUNE = "della_magprune"  # DELLA magnitude pruning (uses epsilon)


class ConsensusMethod(Enum):
    """
    Consensus methods for determining which delta values to keep.

    Reference: mergekit/merge_methods/generalized_task_arithmetic.py
    - COUNT: Majority vote by count of signs
    - SUM: Weighted vote by sum of values
    """
    COUNT = "count"
    SUM = "sum"


def sparsify_delta(
    delta: torch.Tensor,
    density: float = 1.0,
    method: SparsificationMethod = SparsificationMethod.MAGNITUDE,
    rescale: bool = True,
    gamma: float = 0.01,
    epsilon: float = 0.15,
) -> torch.Tensor:
    """
    Sparsify a delta tensor (task vector).

    Reference: mergekit/sparsify.py

    Args:
        delta: The delta tensor to sparsify
        density: Fraction of values to keep (0.0 to 1.0)
        method: Sparsification method
        rescale: Whether to rescale to preserve L1 norm
        gamma: Outlier threshold for magnitude_outliers
        epsilon: Epsilon for della_magprune

    Returns:
        Sparsified delta tensor

    Example:
        ```python
        # Keep top 50% by magnitude
        sparse_delta = sparsify_delta(delta, density=0.5, method=SparsificationMethod.MAGNITUDE)

        # Random pruning with 30% density
        sparse_delta = sparsify_delta(delta, density=0.3, method=SparsificationMethod.RANDOM)
        ```
    """
    if density >= 1.0 or method == SparsificationMethod.NONE:
        return delta

    original_norm = delta.abs().sum() if rescale else None

    if method == SparsificationMethod.MAGNITUDE:
        # Keep top-k by absolute value
        threshold = torch.quantile(delta.abs().flatten(), 1 - density)
        mask = delta.abs() >= threshold
        result = delta * mask

    elif method == SparsificationMethod.RANDOM:
        # Random pruning
        mask = torch.rand_like(delta) < density
        result = delta * mask

    elif method == SparsificationMethod.MAGNITUDE_OUTLIERS:
        # Outlier-based: keep values outside gamma percentile
        # Reference: mergekit gamma parameter
        lower = torch.quantile(delta.flatten(), gamma)
        upper = torch.quantile(delta.flatten(), 1 - gamma)
        mask = (delta < lower) | (delta > upper)
        # Also apply density threshold
        if density < 1.0:
            magnitudes = delta.abs()
            threshold = torch.quantile(magnitudes[mask].flatten(), 1 - density)
            mask = mask & (magnitudes >= threshold)
        result = delta * mask

    elif method == SparsificationMethod.DELLA_MAGPRUNE:
        # DELLA-style magnitude pruning with epsilon
        # Reference: mergekit epsilon parameter
        magnitudes = delta.abs()
        mean_mag = magnitudes.mean()
        threshold = mean_mag * (1 + epsilon)
        # Keep values above threshold, then apply density
        above_threshold = magnitudes >= threshold
        if above_threshold.sum() > delta.numel() * density:
            # Need additional pruning
            values_above = magnitudes[above_threshold]
            top_threshold = torch.quantile(values_above.flatten(), 1 - density)
            mask = above_threshold & (magnitudes >= top_threshold)
        else:
            mask = above_threshold
        result = delta * mask

    else:
        raise ValueError(f"Unknown sparsification method: {method}")

    # Rescale to preserve L1 norm
    if rescale and original_norm is not None:
        current_norm = result.abs().sum()
        if current_norm > 0:
            result = result * (original_norm / current_norm)

    return result


def get_consensus_mask(
    deltas: torch.Tensor,
    method: ConsensusMethod = ConsensusMethod.SUM,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Get consensus mask for merging deltas.

    Reference: mergekit/merge_methods/generalized_task_arithmetic.py get_mask()

    Args:
        deltas: Stacked delta tensors [num_models, ...]
        method: Consensus method (COUNT or SUM)
        dtype: Output mask dtype

    Returns:
        Boolean mask indicating which deltas agree with consensus

    Example:
        ```python
        # Stack deltas from multiple models
        deltas = torch.stack([delta1, delta2, delta3], dim=0)

        # Get consensus mask
        mask = get_consensus_mask(deltas, method=ConsensusMethod.SUM)

        # Apply mask
        merged_delta = (deltas * mask).sum(dim=0)
        ```
    """
    if dtype is None:
        dtype = deltas.dtype

    sign = deltas.sign().to(dtype)

    if method == ConsensusMethod.SUM:
        # TIES paper: weight by magnitude
        sign_weight = deltas.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(dtype) * 2 - 1
    elif method == ConsensusMethod.COUNT:
        # Simple majority vote by count
        majority_sign = (sign.sum(dim=0) >= 0).to(dtype) * 2 - 1
    else:
        raise ValueError(f"Unknown consensus method: {method}")

    # Return mask where each delta agrees with majority
    return sign == majority_sign


class MergeStrategy(Enum):
    """모델 병합 전략"""
    SLERP = "slerp"        # Spherical Linear Interpolation
    TIES = "ties"          # TIES-Merging
    DARE = "dare"          # DARE-TIES
    TASK_ARITHMETIC = "task_arithmetic"  # Task Arithmetic
    EVOLUTIONARY = "evolutionary"  # Sakana AI 스타일


@dataclass
class EvolutionaryMergeConfig(Config):
    """
    진화적 모델 병합 설정 (Sakana AI 스타일).

    여러 모델의 레이어를 유전 알고리즘으로 최적 조합합니다.
    """

    strategy: MergeStrategy = MergeStrategy.EVOLUTIONARY
    """병합 전략"""

    # 진화 설정
    population_size: int = 20
    """한 세대의 개체 수"""

    num_generations: int = 50
    """진화 세대 수"""

    mutation_rate: float = 0.1
    """돌연변이 확률"""

    crossover_rate: float = 0.7
    """교차 확률"""

    elite_ratio: float = 0.1
    """엘리트 보존 비율"""

    # SLERP/Interpolation 설정
    interpolation_weight: float = 0.5
    """보간 가중치 (0=model1, 1=model2)"""

    # TIES 설정
    ties_density: float = 0.5
    """TIES sparse 병합 밀도"""

    # 평가 설정
    eval_fn: Optional[Callable] = None
    """적합도 평가 함수"""

    eval_data: Optional[torch.Tensor] = None
    """평가용 데이터"""


def evolutionary_merge(
    models: List[nn.Module],
    config: EvolutionaryMergeConfig,
    layer_container_name: str = "layers",
) -> nn.Module:
    """
    진화적 모델 병합.

    여러 모델의 레이어를 유전 알고리즘으로 최적 조합합니다.
    Sakana AI의 Evolutionary Model Merge와 유사한 방식입니다.

    Args:
        models: 병합할 모델 리스트
        config: 병합 설정
        layer_container_name: 레이어 컨테이너 이름

    Returns:
        병합된 모델
    """
    log.info(f"Evolutionary merge of {len(models)} models")

    if len(models) < 2:
        raise ValueError("At least 2 models required for merging")

    if config.strategy == MergeStrategy.EVOLUTIONARY:
        return _evolutionary_layer_merge(models, config, layer_container_name)
    elif config.strategy == MergeStrategy.SLERP:
        return _slerp_merge(models[0], models[1], config.interpolation_weight)
    elif config.strategy == MergeStrategy.TIES:
        return _ties_merge(models, config)
    elif config.strategy == MergeStrategy.TASK_ARITHMETIC:
        return _task_arithmetic_merge(models, config)
    else:
        raise ValueError(f"Unknown merge strategy: {config.strategy}")


def _evolutionary_layer_merge(
    models: List[nn.Module],
    config: EvolutionaryMergeConfig,
    layer_container_name: str,
) -> nn.Module:
    """유전 알고리즘 기반 레이어 병합"""
    # 각 모델의 레이어 추출
    all_layers = []
    for model in models:
        layers = _find_layer_container(model, layer_container_name)
        all_layers.append(list(layers))

    n_models = len(models)
    n_layers = len(all_layers[0])

    # 유전자: 각 레이어 위치에서 어떤 모델의 레이어를 사용할지
    # 예: [0, 1, 0, 2, 1, ...] = 첫 레이어는 model0, 두번째는 model1, ...

    def create_individual():
        """랜덤 개체 생성"""
        return [random.randint(0, n_models - 1) for _ in range(n_layers)]

    def crossover(parent1, parent2):
        """교차 (Two-point crossover)"""
        if random.random() > config.crossover_rate:
            return parent1.copy(), parent2.copy()

        point1 = random.randint(0, n_layers - 2)
        point2 = random.randint(point1 + 1, n_layers - 1)

        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return child1, child2

    def mutate(individual):
        """돌연변이"""
        for i in range(n_layers):
            if random.random() < config.mutation_rate:
                individual[i] = random.randint(0, n_models - 1)
        return individual

    def evaluate(individual):
        """적합도 평가"""
        if config.eval_fn is None:
            # 기본: 다양성 점수 (너무 한쪽으로 치우치지 않게)
            counts = [individual.count(i) for i in range(n_models)]
            diversity = -sum((c / n_layers - 1 / n_models) ** 2 for c in counts)
            return diversity

        # 실제 모델 구성 후 평가
        merged = _build_merged_model(models[0], all_layers, individual, layer_container_name)
        return config.eval_fn(merged, config.eval_data)

    # 초기 집단 생성
    population = [create_individual() for _ in range(config.population_size)]

    # 진화 루프
    for gen in range(config.num_generations):
        # 적합도 평가
        fitness_scores = [(ind, evaluate(ind)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        log.info(f"Generation {gen}: Best fitness = {fitness_scores[0][1]:.4f}")

        # 엘리트 선택
        n_elite = max(1, int(config.elite_ratio * config.population_size))
        next_population = [ind for ind, _ in fitness_scores[:n_elite]]

        # 교차 및 돌연변이로 다음 세대 생성
        while len(next_population) < config.population_size:
            # 토너먼트 선택
            parent1 = random.choice([ind for ind, _ in fitness_scores[:config.population_size // 2]])
            parent2 = random.choice([ind for ind, _ in fitness_scores[:config.population_size // 2]])

            child1, child2 = crossover(parent1.copy(), parent2.copy())
            child1 = mutate(child1)
            child2 = mutate(child2)

            next_population.extend([child1, child2])

        population = next_population[:config.population_size]

    # 최고 개체로 모델 구성
    best_individual = fitness_scores[0][0]
    log.info(f"Best individual: {best_individual}")

    merged_model = _build_merged_model(models[0], all_layers, best_individual, layer_container_name)

    return merged_model


def _build_merged_model(
    template: nn.Module,
    all_layers: List[List[nn.Module]],
    individual: List[int],
    layer_container_name: str,
) -> nn.Module:
    """개체 유전자로 병합 모델 구성"""
    merged = copy.deepcopy(template)

    merged_layers = []
    for layer_idx, model_idx in enumerate(individual):
        merged_layers.append(copy.deepcopy(all_layers[model_idx][layer_idx]))

    _replace_layers(merged, layer_container_name, merged_layers)

    return merged


def _slerp_merge(
    model1: nn.Module,
    model2: nn.Module,
    weight: float,
) -> nn.Module:
    """Spherical Linear Interpolation 병합"""
    log.info(f"SLERP merge with weight {weight}")

    merged = copy.deepcopy(model1)

    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(
            merged.named_parameters(), model2.named_parameters()
        ):
            # Flatten for SLERP
            flat1 = param1.flatten()
            flat2 = param2.flatten()

            # Normalize
            norm1 = flat1.norm()
            norm2 = flat2.norm()
            flat1_unit = flat1 / (norm1 + 1e-10)
            flat2_unit = flat2 / (norm2 + 1e-10)

            # SLERP
            dot = (flat1_unit * flat2_unit).sum().clamp(-1, 1)
            theta = torch.acos(dot)

            if theta.abs() < 1e-6:
                # 거의 같은 방향이면 linear interpolation
                merged_flat = (1 - weight) * flat1 + weight * flat2
            else:
                # Spherical interpolation
                sin_theta = torch.sin(theta)
                merged_flat = (
                    torch.sin((1 - weight) * theta) / sin_theta * flat1 +
                    torch.sin(weight * theta) / sin_theta * flat2
                )

            param1.copy_(merged_flat.view_as(param1))

    return merged


def _ties_merge(models: List[nn.Module], config: EvolutionaryMergeConfig) -> nn.Module:
    """TIES-Merging: Trim, Elect Sign, and Merge"""
    log.info("TIES merge")

    base = models[0]
    merged = copy.deepcopy(base)

    with torch.no_grad():
        for name in dict(merged.named_parameters()).keys():
            # 모든 모델의 해당 파라미터 수집
            deltas = []
            base_param = dict(base.named_parameters())[name]

            for model in models[1:]:
                param = dict(model.named_parameters())[name]
                delta = param - base_param
                deltas.append(delta)

            if not deltas:
                continue

            # Trim: 작은 값 제거
            stacked = torch.stack(deltas)
            magnitudes = stacked.abs()
            threshold = torch.quantile(magnitudes, 1 - config.ties_density)
            mask = magnitudes >= threshold

            # Elect Sign: 부호 결정 (다수결)
            signs = torch.sign(stacked)
            elected_sign = torch.sign(signs.sum(dim=0))

            # Merge: 같은 부호만 평균
            same_sign_mask = (signs == elected_sign.unsqueeze(0)) & mask
            merged_delta = (stacked * same_sign_mask).sum(dim=0) / (same_sign_mask.sum(dim=0) + 1e-10)

            # 적용
            merged_param = dict(merged.named_parameters())[name]
            merged_param.add_(merged_delta)

    return merged


def _task_arithmetic_merge(
    models: List[nn.Module],
    config: EvolutionaryMergeConfig,
) -> nn.Module:
    """Task Arithmetic 병합"""
    log.info("Task Arithmetic merge")

    base = models[0]
    merged = copy.deepcopy(base)

    with torch.no_grad():
        for name in dict(merged.named_parameters()).keys():
            base_param = dict(base.named_parameters())[name]
            merged_param = dict(merged.named_parameters())[name]

            # 각 모델의 task vector 합산
            total_delta = torch.zeros_like(base_param)
            for model in models[1:]:
                param = dict(model.named_parameters())[name]
                total_delta += (param - base_param)

            # 평균 적용
            merged_param.add_(total_delta / len(models))

    return merged


# =============================================================================
# 편의 함수
# =============================================================================

def create_dna_transfer_pipeline(
    source_model: nn.Module,
    target_config: Dict[str, Any],
) -> nn.Module:
    """
    DNA 전이 파이프라인.

    소스 모델에서 타겟 설정에 맞는 모델을 자동 생성합니다.

    Args:
        source_model: 소스 모델
        target_config: 타겟 설정
            - "type": "upcycle" | "grow" | "slice" | "merge"
            - 기타 해당 Config 파라미터

    Example:
        ```python
        # Dense 7B → MoE 7Bx8E
        moe_model = create_dna_transfer_pipeline(
            dense_7b,
            {"type": "upcycle", "num_experts": 8}
        )

        # 7B → 13B (width growth)
        larger_model = create_dna_transfer_pipeline(
            model_7b,
            {"type": "grow", "method": "net2wider", "width_multiplier": 1.5}
        )
        ```
    """
    transfer_type = target_config.get("type", "upcycle")

    if transfer_type == "upcycle":
        config = UpcyclingConfig(**{k: v for k, v in target_config.items() if k != "type"})
        return upcycle_dense_to_moe(source_model, config)

    elif transfer_type == "grow":
        config = ModelGrowthConfig(**{k: v for k, v in target_config.items() if k != "type"})
        if config.method == GrowthMethod.NET2WIDER:
            return grow_model_width(source_model, config)
        else:
            return grow_model_depth(source_model, config)

    elif transfer_type == "slice":
        config = ModelSlicingConfig(**{k: v for k, v in target_config.items() if k != "type"})
        return slice_model(source_model, config)

    elif transfer_type == "depth_upscale":
        config = DepthUpscaleConfig(**{k: v for k, v in target_config.items() if k != "type"})
        return depth_upscale(source_model, config)

    else:
        raise ValueError(f"Unknown transfer type: {transfer_type}")
