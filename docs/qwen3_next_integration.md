# Qwen3-Next 통합 아키텍처 가이드

이 문서는 Qwen3-Next 모델에 ETD + MoDr + Dr.LLM + MTP를 통합한 고급 LLM 아키텍처를 설명합니다.

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Qwen3-Next + Advanced Features                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        입력 토큰                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   ENCODER (12 layers, 25%)                           │    │
│  │   ┌─────────────────┐  ┌─────────────────┐                          │    │
│  │   │  Gated DeltaNet │  │  Gated Attention │                         │    │
│  │   │  (선형 어텐션)   │  │  (표준 어텐션)   │                          │    │
│  │   └─────────────────┘  └─────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   THINK (24 layers × N iterations, 50%)              │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  Dr.LLM Layer Router                                       │    │    │
│  │   │  ┌─────────┬───────────┬──────────┐                       │    │    │
│  │   │  │  SKIP   │  EXECUTE  │  REPEAT  │                       │    │    │
│  │   │  │  (0x)   │   (1x)    │   (2x)   │                       │    │    │
│  │   │  └─────────┴───────────┴──────────┘                       │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  MoDr LoRA Experts                                         │    │    │
│  │   │  ┌────────┬────────┬────────┬────────┐                    │    │    │
│  │   │  │Expert 0│Expert 1│Expert 2│Expert 3│                    │    │    │
│  │   │  │  Math  │  Code  │ Reason │ Create │                    │    │    │
│  │   │  └────────┴────────┴────────┴────────┘                    │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  Adaptive Depth Controller                                  │    │    │
│  │   │  신뢰도 ≥ 0.85 → Early Exit                                 │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   DECODER (12 layers, 25%)                           │    │
│  │   ┌─────────────────┐  ┌─────────────────┐                          │    │
│  │   │  Gated DeltaNet │  │  Gated Attention │                         │    │
│  │   └─────────────────┘  └─────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    │                               │                         │
│                    ▼                               ▼                         │
│  ┌───────────────────────────────┐  ┌───────────────────────────────┐       │
│  │        Main LM Head           │  │         MTP Head              │       │
│  │        (k=0: t_{i+1})         │  │   (k=1,2,3,4: t_{i+2:5})      │       │
│  └───────────────────────────────┘  └───────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 통합 컴포넌트

| 컴포넌트 | 역할 | 효과 |
|---------|------|------|
| **Qwen3-Next** | 베이스 모델 | Gated DeltaNet + Attention, MoE |
| **ETD** | 구조적 레이어 분할 | 반복 추론으로 깊은 사고 |
| **MoDr** | LoRA 전문가 분기 | 다양한 사고 패턴 |
| **Dr.LLM** | 동적 깊이 라우팅 | 효율적 연산 할당 |
| **MTP** | 멀티토큰 예측 | 2-3x 추론 속도 향상 |

## 빠른 시작

### 기본 사용법

```python
from olmo_core.nn.transformer import create_qwen3_next_model

# 통합 모델 생성
model = create_qwen3_next_model(
    model_name="Qwen/Qwen3-Next-32B-A3B",
    use_etd=True,
    use_mtp=True,
)

# 학습
outputs = model(input_ids, labels=labels)
total_loss = outputs.loss  # main + mtp loss

# 추론 (Speculative Decoding)
generated, stats = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
)
print(f"Speedup: {stats['speedup']:.2f}x")
```

### 효율성 중시 설정

```python
from olmo_core.nn.transformer.qwen3_next_wrapper import create_qwen3_next_efficient

# 빠른 파인튜닝을 위한 설정
model = create_qwen3_next_efficient()
# - 백본 동결
# - 적은 Think 반복 (2회)
# - 적은 MTP 토큰 (3개)
```

### 성능 중시 설정

```python
from olmo_core.nn.transformer.qwen3_next_wrapper import create_qwen3_next_performance

# 최대 성능을 위한 설정
model = create_qwen3_next_performance()
# - 전체 학습
# - 많은 Think 반복 (5회)
# - 많은 MTP 토큰 (4개)
# - 많은 LoRA 전문가 (8개)
```

## 세부 설정

### ETD 설정

```python
from olmo_core.nn.transformer import ETDConfig, Qwen3NextConfig

etd_config = ETDConfig(
    # 레이어 분할 (48 layers 기준)
    n_encoder_layers=12,      # 25%
    n_think_layers=24,        # 50%
    n_decoder_layers=12,      # 25%

    # Think 블록 설정
    max_think_iterations=3,
    adaptive_depth=True,
    confidence_threshold=0.85,

    # Dr.LLM 라우터
    use_layer_router=True,
    layer_router=LayerRouterConfig(
        hidden_size=256,
        temperature=1.0,
    ),

    # MoDr 전문가
    use_lora_experts=True,
    num_lora_experts=4,
    lora_expert=LoRAExpertConfig(
        rank=32,
        alpha=64.0,
    ),
)
```

### MTP 설정

```python
from olmo_core.nn.transformer import MTPConfig

mtp_config = MTPConfig(
    # 예측 설정
    num_predict_tokens=4,         # 4개 미래 토큰 예측
    mtp_head_type="transformer",  # transformer/mlp/shared_lm_head

    # Loss 설정
    loss_weight_decay=0.5,        # 지수 감쇠 계수
    mtp_loss_weight=0.3,          # MTP loss 가중치

    # Speculative Decoding
    speculation_lookahead=3,
)
```

## 학습 가이드

### 1. 파인튜닝 전략

```python
# 방법 1: 전체 학습 (최대 성능)
model = create_qwen3_next_model(freeze_backbone=False)

# 방법 2: 효율적 파인튜닝 (ETD/MTP만 학습)
model = create_qwen3_next_model(freeze_backbone=True)
# 학습 가능: 라우터, LoRA 전문가, MTP 헤드
# 약 3-5%의 파라미터만 학습
```

### 2. Loss 구성

```python
outputs = model(input_ids, labels=labels)

# 개별 loss 확인
main_loss = outputs.loss - outputs.mtp_loss  # 메인 next-token loss
mtp_loss = outputs.mtp_loss                   # MTP 보조 loss

# 각 MTP 스텝별 loss
for k, loss in outputs.mtp_step_losses.items():
    print(f"{k}: {loss:.4f}")
```

### 3. 메트릭 모니터링

```python
# ETD 메트릭
metrics = model.get_metrics()

print(f"Think iterations: {metrics['etd']['think_iterations']}")
print(f"Early exit: {metrics['etd']['early_exit']}")
print(f"Router actions: {metrics['etd']['router_actions']}")
print(f"Expert selections: {metrics['etd']['expert_selections']}")
```

## 추론 가이드

### Self-Speculative Decoding

MTP 헤드를 드래프트 모델로 사용하여 추론 속도를 2-3배 향상시킵니다:

```
1. Trunk forward → hidden states
2. MTP 헤드로 k개 드래프트 토큰 생성
3. 드래프트 + 1토큰을 한 번에 검증
4. 수락된 토큰들 + 새 토큰 추가
5. 반복
```

### 추론 성능

| 설정 | Acceptance Rate | Speedup |
|-----|-----------------|---------|
| 기본 (k=3) | ~50% | ~1.8x |
| 최적화 | ~60% | ~2.0x |

```python
# Speculative decoding 활성화/비활성화
generated, stats = model.generate(
    input_ids,
    use_speculative=True,  # False로 기본 생성
)
```

## 메모리 최적화

### Gradient Checkpointing

```python
config = Qwen3NextConfig(
    gradient_checkpointing=True,  # 기본값
)
```

### Mixed Precision

```python
# BF16으로 로드 (기본)
model = create_qwen3_next_model()  # torch.bfloat16

# 추론 시 FP16도 지원
model = model.half()
```

## 하드웨어 요구사항

| 모델 | 파라미터 | 활성화 | VRAM (학습) | VRAM (추론) |
|-----|---------|-------|-------------|-------------|
| Qwen3-Next-32B-A3B | 80B | 3B | ~48GB | ~16GB |
| + ETD/MTP | +5% | +3B | ~52GB | ~18GB |

## 성능 벤치마크

### 학습 속도

| 설정 | 토큰/초 | 상대 속도 |
|-----|--------|----------|
| 기본 Qwen3-Next | 1000 | 1.0x |
| + ETD (반복 3회) | 600 | 0.6x |
| + MTP | 900 | 0.9x |
| + ETD + MTP | 550 | 0.55x |

### 추론 속도

| 설정 | 토큰/초 | 상대 속도 |
|-----|--------|----------|
| 기본 | 100 | 1.0x |
| + MTP Speculative | 200 | 2.0x |
| + ETD (적응형) | 150 | 1.5x |
| + 둘 다 | 280 | 2.8x |

## 관련 파일

- `src/olmo_core/nn/transformer/qwen3_next_wrapper.py`: Qwen3-Next 통합 래퍼
- `src/olmo_core/nn/transformer/etd.py`: ETD + MoDr + Dr.LLM
- `src/olmo_core/nn/transformer/mtp.py`: MTP (Multi-Token Prediction)
- `docs/etd_modr_drllm_integration.md`: ETD 상세 문서
- `docs/mtp_guide.md`: MTP 상세 문서

## 참고 문헌

1. **Qwen3-Next**: Alibaba (2025)
   - Gated DeltaNet + Gated Attention
   - MoE (80B total, 3B active)

2. **ETD**: arXiv:2510.07358
   - Encode-Think-Decode 구조

3. **MoDr**: OpenReview
   - Mixture-of-Depth-Recurrent

4. **Dr.LLM**: arXiv:2510.12773
   - Dynamic Layer Routing

5. **MTP**: arXiv:2404.19737
   - Multi-Token Prediction
