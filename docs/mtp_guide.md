# MTP (Multi-Token Prediction) 가이드

이 문서는 OLMo-core에 통합된 MTP (Multi-Token Prediction) 모듈의 사용법을 설명합니다.

## 개요

MTP는 LLM의 학습과 추론을 동시에 개선하는 기술입니다:

| 단계 | 효과 | 구현 방식 |
|-----|------|----------|
| **학습** | 더 나은 표현 학습 | n개 미래 토큰 동시 예측 |
| **추론** | 2-3x 속도 향상 | Self-speculative decoding |

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Trunk (Backbone)                   │
│                     ┌──────────────────┐                     │
│                     │   Hidden States   │                    │
│                     │      h_1:i        │                    │
│                     └────────┬─────────┘                     │
└──────────────────────────────┼───────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Main LM Head   │  │    MTP Head      │  │    MTP Head      │
│   (k=0)          │  │    (k=1)         │  │    (k=2,3,...)   │
│                  │  │                  │  │                  │
│  h_i → t_{i+1}   │  │  h_i + e_{i+1}   │  │  h^k + e_{i+k}   │
│                  │  │  → t_{i+2}       │  │  → t_{i+k+1}     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## 참고 논문

1. **Meta MTP**: arXiv:2404.19737
   - "Better & Faster Large Language Models via Multi-token Prediction"
   - n개의 독립적 출력 헤드로 미래 토큰 예측

2. **FastMTP**: arXiv:2509.18362
   - DeepSeek-V3 스타일 단일 MTP 헤드
   - 공유 가중치로 효율적 구현

3. **Efficient MTP**: arXiv:2502.09419
   - 언어별 어휘 압축
   - 추론 최적화 기법

## 빠른 시작

### 1. 기존 모델에 MTP 추가

```python
from olmo_core.nn.transformer import MTPConfig, wrap_model_with_mtp

# MTP 설정
mtp_config = MTPConfig(
    num_predict_tokens=4,          # 4개 미래 토큰 예측 (최적)
    mtp_head_type="transformer",   # transformer 또는 mlp
    share_weights_across_steps=True,  # FastMTP 스타일
    loss_weight_decay=0.5,         # 지수 감쇠 계수
    mtp_loss_weight=0.5,           # MTP loss 가중치
)

# 모델에 MTP 추가
model = wrap_model_with_mtp(model, mtp_config)

# 학습
outputs = model(input_ids, labels=labels)
total_loss = outputs.loss  # main_loss + mtp_loss
mtp_loss = outputs.mtp_loss
```

### 2. Speculative Decoding으로 추론

```python
# MTP가 추가된 모델로 빠른 생성
generated, stats = model.mtp_decoder.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

print(f"Speedup: {stats['speedup']:.2f}x")
print(f"Acceptance rate: {stats['avg_acceptance_rate']:.2%}")
```

## 구성 요소 상세

### MTPConfig

```python
@dataclass
class MTPConfig(Config):
    num_predict_tokens: int = 4
    """예측할 미래 토큰 수. 논문에서 4가 최적"""

    mtp_head_type: str = "transformer"
    """MTP 헤드 타입: 'transformer', 'mlp', 'shared_lm_head'"""

    share_weights_across_steps: bool = True
    """FastMTP 스타일: 모든 예측 스텝에서 동일한 헤드 재사용"""

    loss_weight_decay: float = 0.5
    """지수 감쇠 계수 (β). 먼 미래 토큰일수록 낮은 가중치"""

    mtp_loss_weight: float = 0.5
    """MTP 보조 loss 총 가중치"""

    speculation_lookahead: int = 3
    """추론 시 드래프트 토큰 수"""
```

### MTP Loss

지수 감쇠 가중치를 사용하여 여러 예측 스텝의 loss를 결합합니다:

```
L_mtp = Σ(k=1 to K) α_k * CE(logits_k, targets_k)
α_k = β^(k-1) / Σβ^(j-1)
```

예시 (K=4, β=0.5):
- α_1 = 0.533 (k=1: t_{i+2} 예측)
- α_2 = 0.267 (k=2: t_{i+3} 예측)
- α_3 = 0.133 (k=3: t_{i+4} 예측)
- α_4 = 0.067 (k=4: t_{i+5} 예측)

### MTP Head 타입

| 타입 | 설명 | 파라미터 | 추천 |
|-----|------|---------|------|
| `transformer` | Transformer 레이어 | ~3% 추가 | 성능 최적 |
| `mlp` | 간단한 MLP | ~1% 추가 | 효율성 중시 |
| `shared_lm_head` | LM head 공유 | 0% 추가 | 최소 오버헤드 |

## Self-Speculative Decoding

MTP 헤드를 드래프트 모델로 사용하여 추론을 가속화합니다:

```
1. Trunk forward → hidden states
2. MTP 헤드로 k개 드래프트 토큰 생성
3. 드래프트 + 1개 토큰을 한 번에 검증
4. 수락된 토큰들 + 새 토큰 추가
5. 반복
```

### 기대 성능

| 설정 | Acceptance Rate | Speedup |
|-----|-----------------|---------|
| k=1 (기본) | ~81% | ~1.5x |
| k=2 | ~56% | ~1.8x |
| k=3 | ~36% | ~2.0x |

## 학습 가이드

### 1. 웜업 전략

초기에는 MTP loss를 낮게 유지:

```python
def get_mtp_weight(step, warmup_steps=1000, target_weight=0.5):
    if step < warmup_steps:
        return target_weight * (step / warmup_steps)
    return target_weight
```

### 2. 메모리 효율적 학습

순차 forward/backward로 메모리 사용량 최적화:

```python
mtp_config = MTPConfig(
    num_predict_tokens=4,
)

training_config = MTPTrainingConfig(
    gradient_checkpointing=True,
    sequential_forward=True,  # O(V+d) vs O(nV+d)
)
```

### 3. 메트릭 모니터링

```python
outputs = model(input_ids, labels=labels)

# 각 스텝별 loss 확인
for k, loss in outputs.mtp_step_losses.items():
    print(f"{k}: {loss:.4f}")

# 전체 MTP loss
print(f"MTP loss: {outputs.mtp_loss:.4f}")
print(f"Total loss: {outputs.loss:.4f}")
```

## ETD + MTP 통합

ETD와 MTP를 함께 사용하여 최대 효과를 얻을 수 있습니다:

```python
from olmo_core.nn.transformer import (
    ETDConfig, wrap_transformer_with_etd,
    MTPConfig, wrap_model_with_mtp,
)

# 1. ETD 래핑
etd_config = ETDConfig(
    n_encoder_layers=7,
    n_think_layers=4,
    n_decoder_layers=5,
    max_think_iterations=5,
    use_layer_router=True,
    use_lora_experts=True,
)
model = wrap_transformer_with_etd(model, etd_config)

# 2. MTP 추가
mtp_config = MTPConfig(
    num_predict_tokens=4,
    mtp_head_type="transformer",
)
model = wrap_model_with_mtp(model, mtp_config)
```

## 하이퍼파라미터 권장값

### 학습 설정

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| num_predict_tokens | 4 | 논문 실험 결과 최적 |
| mtp_head_type | transformer | 성능 vs 효율 균형 |
| loss_weight_decay | 0.5 | 먼 미래 토큰 감쇠 |
| mtp_loss_weight | 0.5 | 메인:MTP = 1:0.5 |

### 추론 설정

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| speculation_lookahead | 3 | 수락률과 속도의 균형 |
| speculation_temperature | 0.0 | Greedy 드래프트 |
| acceptance_threshold | 0.9 | 품질 유지 |

## 관련 파일

- `src/olmo_core/nn/transformer/mtp.py`: MTP 모듈 구현
- `src/olmo_core/nn/transformer/etd.py`: ETD 모듈 (함께 사용)
- `docs/etd_modr_drllm_integration.md`: ETD 통합 문서
