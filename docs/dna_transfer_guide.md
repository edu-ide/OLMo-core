# DNA Transfer: 모델 가중치 전이 가이드

이 문서는 AI 업계에서 사용하는 주요 가중치 전이 기법들을 설명합니다.

## 개요

DNA Transfer는 기존 모델의 "지식(가중치)"을 새 모델로 전달하여 학습 비용을 절감하고 안정성을 높이는 기법들입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DNA Transfer Methods                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. UPCYCLING                    2. MODEL GROWTH                            │
│  ┌────────────┐                  ┌────────────┐                             │
│  │  Dense 7B  │                  │  Small 7B  │                             │
│  │    FFN     │                  │   Width    │                             │
│  └─────┬──────┘                  └─────┬──────┘                             │
│        │ 복제                          │ 분열                               │
│        ▼                              ▼                                     │
│  ┌────────────┐                  ┌────────────┐                             │
│  │  MoE 7Bx8E │                  │  Large 13B │                             │
│  │ E0 E1 E2.. │                  │   Width    │                             │
│  └────────────┘                  └────────────┘                             │
│                                                                              │
│  3. MODEL SLICING                4. EVOLUTIONARY MERGE                      │
│  ┌────────────┐                  ┌────────────┐ ┌────────────┐             │
│  │  Large 12B │                  │  Model A   │ │  Model B   │             │
│  │  48 layers │                  │   Math     │ │   Code     │             │
│  └─────┬──────┘                  └─────┬──────┘ └─────┬──────┘             │
│        │ 추출                          │ 교배         │                    │
│        ▼                              └──────┬───────┘                     │
│  ┌────────────┐                  ┌────────────┐                             │
│  │  Small 6B  │                  │  Merged C  │                             │
│  │  24 layers │                  │ Math+Code  │                             │
│  └────────────┘                  └────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Upcycling: Dense → MoE 변환

밀집 모델의 FFN을 복제하여 MoE(Mixture of Experts)로 변환합니다.

### 사용 사례
- Qwen, DeepSeek, Mistral의 MoE 모델
- 적은 추가 학습으로 용량 확장

### 사용법

```python
from olmo_core.nn.transformer import UpcyclingConfig, upcycle_dense_to_moe

# 설정
config = UpcyclingConfig(
    num_experts=8,              # 전문가 수
    num_experts_per_token=2,    # 토큰당 활성화 전문가
    expert_init_noise=0.01,     # 다양성을 위한 노이즈
    auxiliary_loss_weight=0.01, # Load balancing loss
)

# Dense → MoE 변환
moe_model = upcycle_dense_to_moe(dense_model, config)

# 학습 시 auxiliary loss 사용
outputs = moe_model(input_ids, labels=labels)
total_loss = outputs.loss + moe_model.get_moe_aux_loss()
```

### Depth Up-Scaling (Solar 스타일)

레이어를 복제하여 모델 깊이를 확장합니다.

```python
from olmo_core.nn.transformer import DepthUpscaleConfig, depth_upscale

config = DepthUpscaleConfig(
    scale_factor=2,              # 2배 깊이
    insertion_method="interleave",  # 교차 삽입
    init_noise=0.001,
)

# 16 layers → 32 layers
scaled_model = depth_upscale(model, config)
```

## 2. Net2Net / Model Growth

모델의 크기를 수학적으로 확장하면서 지식을 보존합니다.

### Net2Wider: 너비 확장

뉴런을 분열시켜 hidden dimension을 늘립니다.

```python
from olmo_core.nn.transformer import ModelGrowthConfig, GrowthMethod, grow_model_width

config = ModelGrowthConfig(
    method=GrowthMethod.NET2WIDER,
    width_multiplier=2.0,       # 2배 너비
    preserve_output=True,       # 초기 출력값 보존
    noise_std=0.01,
)

# d_model: 4096 → 8192
wider_model = grow_model_width(model, config)
```

### Net2Deeper: 깊이 확장

Identity 레이어를 삽입하여 깊이를 늘립니다.

```python
from olmo_core.nn.transformer import grow_model_depth

config = ModelGrowthConfig(
    method=GrowthMethod.NET2DEEPER,
    num_new_layers=8,           # 8개 레이어 추가
    preserve_output=True,
)

# 24 layers → 32 layers
deeper_model = grow_model_depth(model, config)
```

### Function-Preserving 원리

```
Net2Wider:
- 뉴런 A를 A'와 A''로 분열
- 가중치를 절반으로 나눔: W/2, W/2
- 초기 출력: (W/2 * x) + (W/2 * x) = W * x (동일)

Net2Deeper:
- Identity 레이어 삽입: y = x (변화 없음)
- 추가 학습으로 점진적 학습
```

## 3. Model Slicing (Nemotron Elastic)

큰 부모 모델에서 작은 자식 모델을 추출합니다.

### 단일 모델 슬라이싱

```python
from olmo_core.nn.transformer import ModelSlicingConfig, slice_model

config = ModelSlicingConfig(
    target_layers=24,           # 목표 레이어 수
    target_hidden_size=2048,    # 목표 hidden size
    layer_selection="importance",  # 중요도 기반 선택
    importance_metric="weight_norm",
    preserve_first_last=True,
)

# 12B (48L, 4096D) → 6B (24L, 2048D)
small_model = slice_model(large_model, config)
```

### 모델 패밀리 생성

하나의 큰 모델에서 여러 크기의 모델을 추출합니다.

```python
from olmo_core.nn.transformer import create_model_family

# 12B 모델에서 패밀리 생성
family = create_model_family(
    base_12b_model,
    sizes=[
        (48, 4096),  # 12B (원본)
        (36, 3584),  # 9B
        (24, 3072),  # 6B
        (16, 2048),  # 3B
    ],
)

# 사용
model_9b = family["36L_3584D"]
model_6b = family["24L_3072D"]
```

### 레이어 선택 방식

| 방식 | 설명 | 추천 |
|-----|------|------|
| `importance` | 중요도 기반 선택 | 성능 최적 |
| `uniform` | 균등 간격 선택 | 빠른 실험 |
| `first` | 앞쪽 레이어 | 인코더 중시 |
| `last` | 뒤쪽 레이어 | 디코더 중시 |

## 4. Evolutionary Model Merge

여러 모델의 레이어를 유전 알고리즘으로 최적 조합합니다.

### Sakana AI 스타일 병합

```python
from olmo_core.nn.transformer import (
    EvolutionaryMergeConfig,
    MergeStrategy,
    evolutionary_merge
)

config = EvolutionaryMergeConfig(
    strategy=MergeStrategy.EVOLUTIONARY,
    population_size=20,         # 세대당 개체 수
    num_generations=50,         # 진화 세대 수
    mutation_rate=0.1,
    crossover_rate=0.7,
    elite_ratio=0.1,
)

# 수학 모델 + 코딩 모델 → 통합 모델
merged_model = evolutionary_merge(
    [math_model, code_model],
    config,
)
```

### 기타 병합 전략

```python
# SLERP (Spherical Linear Interpolation)
config = EvolutionaryMergeConfig(
    strategy=MergeStrategy.SLERP,
    interpolation_weight=0.5,  # 50:50 병합
)

# TIES-Merging (Sparse merge)
config = EvolutionaryMergeConfig(
    strategy=MergeStrategy.TIES,
    ties_density=0.5,  # 50% 밀도
)

# Task Arithmetic
config = EvolutionaryMergeConfig(
    strategy=MergeStrategy.TASK_ARITHMETIC,
)
```

### 병합 전략 비교

| 전략 | 설명 | 장점 | 단점 |
|-----|------|-----|------|
| **SLERP** | 구면 선형 보간 | 부드러운 병합 | 2개 모델만 |
| **TIES** | 희소 병합 | 충돌 감소 | 복잡 |
| **Task Arithmetic** | 작업 벡터 합 | 간단 | 간섭 가능 |
| **Evolutionary** | 유전 알고리즘 | 최적 조합 | 평가 함수 필요 |

## 통합 파이프라인

간단한 설정으로 DNA 전이를 수행합니다.

```python
from olmo_core.nn.transformer import create_dna_transfer_pipeline

# Dense → MoE
moe_model = create_dna_transfer_pipeline(
    dense_model,
    {"type": "upcycle", "num_experts": 8}
)

# 너비 확장
wider_model = create_dna_transfer_pipeline(
    small_model,
    {"type": "grow", "method": "net2wider", "width_multiplier": 1.5}
)

# 깊이 확장
deeper_model = create_dna_transfer_pipeline(
    small_model,
    {"type": "depth_upscale", "scale_factor": 2}
)

# 슬라이싱
smaller_model = create_dna_transfer_pipeline(
    large_model,
    {"type": "slice", "target_layers": 24}
)
```

## 실전 예제: 모델 개발 파이프라인

### 예제 1: 7B Dense → 7Bx8E MoE

```python
# 1. 기존 7B 모델 로드
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 2. MoE로 업사이클
from olmo_core.nn.transformer import UpcyclingConfig, upcycle_dense_to_moe

config = UpcyclingConfig(
    num_experts=8,
    num_experts_per_token=2,
    expert_init_noise=0.01,
)
moe_model = upcycle_dense_to_moe(base_model, config)

# 3. 추가 학습 (짧은 기간)
# 기존 지식이 보존되어 있어 빠르게 수렴
```

### 예제 2: 13B → 7B/3B 패밀리

```python
# 1. 큰 모델 학습 (또는 로드)
large_model = train_13b_model()

# 2. 패밀리 생성
from olmo_core.nn.transformer import create_model_family

family = create_model_family(
    large_model,
    sizes=[
        (40, 5120),   # 13B 원본
        (32, 4096),   # 7B
        (24, 2560),   # 3B
    ],
)

# 3. 각 모델 추가 파인튜닝 (선택적)
for name, model in family.items():
    finetune(model, f"data/finetune_{name}")
```

### 예제 3: 전문 모델 병합

```python
# 1. 전문 모델들 준비
math_model = load_model("math-specialist")
code_model = load_model("code-specialist")
writing_model = load_model("writing-specialist")

# 2. 진화적 병합
from olmo_core.nn.transformer import EvolutionaryMergeConfig, evolutionary_merge

def evaluate_model(model, data):
    """다양한 작업에서 성능 평가"""
    math_score = eval_math(model, data["math"])
    code_score = eval_code(model, data["code"])
    writing_score = eval_writing(model, data["writing"])
    return (math_score + code_score + writing_score) / 3

config = EvolutionaryMergeConfig(
    strategy=MergeStrategy.EVOLUTIONARY,
    population_size=30,
    num_generations=100,
    eval_fn=evaluate_model,
    eval_data={"math": ..., "code": ..., "writing": ...},
)

merged_model = evolutionary_merge(
    [math_model, code_model, writing_model],
    config,
)
```

## 하이퍼파라미터 가이드

### Upcycling

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| num_experts | 8 | 전문가 수 |
| num_experts_per_token | 2 | 활성화 전문가 |
| expert_init_noise | 0.01 | 다양성 노이즈 |
| auxiliary_loss_weight | 0.01 | Load balancing |

### Model Growth

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| width_multiplier | 1.5-2.0 | 너비 배율 |
| preserve_output | True | 출력 보존 |
| noise_std | 0.01 | 초기화 노이즈 |

### Model Slicing

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| layer_selection | importance | 선택 방식 |
| preserve_first_last | True | 경계 레이어 보존 |

### Evolutionary Merge

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| population_size | 20-50 | 집단 크기 |
| num_generations | 50-100 | 세대 수 |
| mutation_rate | 0.1 | 돌연변이율 |
| elite_ratio | 0.1 | 엘리트 보존 |

## 참고 문헌

1. **Sparse Upcycling**: arXiv:2212.05055
2. **Solar DUS**: arXiv:2312.15166
3. **Net2Net**: arXiv:1511.05641
4. **NVIDIA Nemotron Elastic**: Technical Report (2024)
5. **Sakana Evolutionary Merge**: arXiv:2403.13187
6. **TIES-Merging**: arXiv:2306.01708
7. **Task Arithmetic**: arXiv:2212.04089

## 관련 파일

- `src/olmo_core/nn/transformer/dna_transfer.py`: DNA Transfer 모듈
- `src/olmo_core/nn/transformer/etd.py`: ETD 통합 (함께 사용 가능)
- `src/olmo_core/nn/transformer/mtp.py`: MTP (함께 사용 가능)
