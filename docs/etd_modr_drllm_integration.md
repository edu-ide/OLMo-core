# ETD + MoDr + Dr.LLM 통합 아키텍처

이 문서는 OLMo-core에 통합된 세 가지 최신 아키텍처의 설계와 사용법을 설명합니다.

## 개요

세 가지 아키텍처를 결합하여 적응적이고 효율적인 추론을 가능하게 합니다:

| 아키텍처 | 역할 | 논문 |
|---------|------|------|
| **ETD** | 어디서 생각할지 (구조적 레이어 분할) | arXiv:2510.07358 |
| **MoDr** | 어떻게 생각할지 (LoRA 전문가 다양성) | OpenReview |
| **Dr.LLM** | 얼마나 생각할지 (동적 깊이 라우팅) | arXiv:2510.12773 |

```
┌─────────────────────────────────────────────────────────────┐
│                        입력 토큰                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ENCODER (7 layers)                        │
│                   입력 → Latent Space                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   THINK (4 layers × N iterations)           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Layer Router (Dr.LLM)                               │   │
│  │  ┌─────────┬─────────┬─────────┐                    │   │
│  │  │  SKIP   │ EXECUTE │ REPEAT  │                    │   │
│  │  │  (0x)   │  (1x)   │  (2x)   │                    │   │
│  │  └─────────┴─────────┴─────────┘                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LoRA Experts (MoDr)                                 │   │
│  │  ┌────────┬────────┬────────┬────────┐              │   │
│  │  │ Expert │ Expert │ Expert │ Expert │              │   │
│  │  │   0    │   1    │   2    │   3    │              │   │
│  │  └────────┴────────┴────────┴────────┘              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Adaptive Depth Controller                           │   │
│  │  신뢰도 ≥ threshold → Early Exit                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DECODER (5 layers)                        │
│                   Latent → 출력 토큰                        │
└─────────────────────────────────────────────────────────────┘
```

## 빠른 시작

### 1. Config 프리셋 사용

```python
from olmo_core.nn.transformer import TransformerConfig

# ETD 통합 설정 생성
config = TransformerConfig.olmo3_7B_etd(
    vocab_size=50280,
    # ETD 구조
    n_encoder_layers=7,
    n_think_layers=4,
    n_decoder_layers=5,
    # Think 블록 설정
    max_think_iterations=5,
    adaptive_depth=True,
    confidence_threshold=0.9,
    # Dr.LLM 라우터
    use_layer_router=True,
    # MoDr 전문가
    use_lora_experts=True,
    num_lora_experts=4,
    lora_rank=16,
    lora_alpha=32.0,
)

# 모델 빌드
model = config.build()
```

### 2. 기존 모델에 ETD 래핑

```python
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.etd import ETDConfig, wrap_transformer_with_etd

# 기존 모델 로드
config = TransformerConfig.olmo3_7B(vocab_size=50280)
model = config.build()

# ETD 설정 생성
etd_config = ETDConfig(
    n_encoder_layers=7,
    n_think_layers=4,
    n_decoder_layers=5,
    max_think_iterations=5,
    adaptive_depth=True,
    use_layer_router=True,
    use_lora_experts=True,
    num_lora_experts=4,
)

# ETD 래핑 적용
model = wrap_transformer_with_etd(model, etd_config)
```

## 구성 요소 상세

### ETD (Encode-Think-Decode)

레이어를 세 블록으로 분할하여 구조적 추론을 가능하게 합니다:

- **Encoder (7 layers)**: 입력 토큰을 latent space로 매핑
- **Think (4 layers)**: 반복 실행되는 추론 블록 (최대 5회)
- **Decoder (5 layers)**: Latent representation을 출력으로 변환

```python
from olmo_core.nn.transformer.etd import ETDConfig

etd = ETDConfig(
    n_encoder_layers=7,      # 입력 처리
    n_think_layers=4,        # 반복 추론
    n_decoder_layers=5,      # 출력 생성
    max_think_iterations=5,  # 최대 반복 횟수
)
```

### Dr.LLM (Dynamic Layer Routing)

각 Think 레이어 앞에서 Skip/Execute/Repeat 결정을 내립니다:

| 액션 | 설명 | 효과 |
|-----|------|-----|
| **SKIP** | 레이어 건너뛰기 | 연산 절약 |
| **EXECUTE** | 1회 실행 | 기본 동작 |
| **REPEAT** | 2회 실행 | 더 깊은 추론 |

```python
from olmo_core.nn.transformer.etd import LayerRouterConfig

router = LayerRouterConfig(
    hidden_size=256,         # 라우터 MLP 크기
    window_size=8,           # Windowed mean pooling
    num_actions=3,           # Skip/Execute/Repeat
    temperature=1.0,         # Softmax 온도
    use_gumbel=False,        # Gumbel-Softmax (학습 시)
)
```

### MoDr (LoRA Experts)

Think 블록에서 다양한 사고 패턴을 위한 LoRA 전문가 분기:

```python
from olmo_core.nn.transformer.etd import LoRAExpertConfig

expert = LoRAExpertConfig(
    rank=16,                 # LoRA rank
    alpha=32.0,              # Scaling factor
    dropout=0.0,             # LoRA dropout
)
```

**수식**: `W' = W + (alpha/rank) × B @ A`

### Adaptive Depth (적응형 깊이)

Think 반복 중 신뢰도 기반 조기 종료:

```python
etd = ETDConfig(
    adaptive_depth=True,
    confidence_threshold=0.9,  # 이 신뢰도 초과 시 조기 종료
)
```

## 학습 가이드

### 1. 웜업 전략

라우터와 전문가는 학습 초기에 불안정할 수 있으므로 웜업 기간 동안 비활성화됩니다:

```python
etd = ETDConfig(
    router_warmup_steps=1000,   # 라우터 웜업 (초기에는 항상 EXECUTE)
    expert_warmup_steps=500,    # 전문가 웜업 (초기에는 균등 선택)
)
```

### 2. Load Balancing (부하 분산)

전문가 사용 불균형을 방지하기 위해 auxiliary-loss-free 방식 사용:

```python
# 배치 후 전문가 bias 업데이트
model.etd.update_expert_bias(lr=0.01)
```

### 3. 메트릭 모니터링

```python
# Forward 후 메트릭 확인
output = model(input_ids, labels=labels)
metrics = model._etd_metrics

print(f"Think iterations: {metrics['think_iterations']}")
print(f"Router actions: {metrics['router_actions']}")
print(f"Expert selections: {metrics['expert_selections']}")
print(f"Early exit: {metrics['early_exit']}")

# 라우터 통계
stats = model.etd.get_router_stats()
print(f"Skip count: {stats['skip_count']}")
print(f"Execute count: {stats['execute_count']}")
print(f"Repeat count: {stats['repeat_count']}")
```

## 하이퍼파라미터 권장값

### 기본 설정 (7B 모델)

| 파라미터 | 값 | 설명 |
|---------|---|------|
| n_encoder_layers | 7 | 전체의 ~44% |
| n_think_layers | 4 | 전체의 ~25% |
| n_decoder_layers | 5 | 전체의 ~31% |
| max_think_iterations | 5 | 최대 반복 |
| num_lora_experts | 4 | 적당한 다양성 |
| lora_rank | 16 | 효율/성능 균형 |

### 효율성 중시 설정

```python
config = TransformerConfig.olmo3_7B_etd(
    vocab_size=50280,
    max_think_iterations=3,    # 반복 줄임
    adaptive_depth=True,       # 조기 종료 활성화
    confidence_threshold=0.8,  # 낮은 임계값 → 빠른 종료
    num_lora_experts=2,        # 전문가 수 줄임
    lora_rank=8,               # 낮은 rank
)
```

### 성능 중시 설정

```python
config = TransformerConfig.olmo3_7B_etd(
    vocab_size=50280,
    max_think_iterations=8,    # 더 많은 반복
    adaptive_depth=True,
    confidence_threshold=0.95, # 높은 임계값 → 충분한 사고
    num_lora_experts=8,        # 더 많은 전문가
    lora_rank=32,              # 높은 rank
)
```

## 추론 (Inference)

### KV Cache 고려사항

ETD 구조에서는 Think 블록이 반복되므로 KV cache 관리가 다릅니다:

```python
# Think 반복 시에도 Encoder/Decoder KV cache는 재사용
# Think 블록의 KV cache만 반복마다 갱신
```

### 적응형 깊이와 지연시간

- `adaptive_depth=True`: 쉬운 토큰은 적은 반복으로 빠르게 처리
- 평균 반복 횟수를 모니터링하여 `confidence_threshold` 조정

## 참고 문헌

1. **ETD (Encode, Think, Decode)**: arXiv:2510.07358
   - 레이어 분할을 통한 구조적 추론
   - Angular distance 기반 레이어 경계 탐지

2. **MoDr (Mixture-of-Depth-Recurrent)**: OpenReview
   - LoRA 기반 전문가 분기
   - 사고 다양성 확보

3. **Dr.LLM (Dynamic Layer Routing)**: arXiv:2510.12773
   - MCTS 기반 라우터 학습
   - Skip/Execute/Repeat 동적 결정

## 관련 파일

- `src/olmo_core/nn/transformer/etd.py`: ETD 모듈 구현
- `src/olmo_core/nn/transformer/config.py`: TransformerConfig.olmo3_7B_etd()
- `docs/hybrid_attention_moe.md`: 하이브리드 어텐션/MoE 설명
