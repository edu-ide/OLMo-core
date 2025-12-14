# Cross-Architecture Transfer: Qwen3 → Hybrid 가이드

이 문서는 Qwen3-8B에서 Hybrid (GatedDeltaNet + GatedAttention) 모델로의 가중치 전이를 설명합니다.

## 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Cross-Architecture Transfer Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Qwen3-8B (Source)                    Hybrid Model (Target)                 │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │ Standard Attention  │    직접      │ GatedOutputAttention│               │
│  │ Q, K, V, O          │ ──────────▶  │ Q, K, V, O + Gate   │               │
│  └─────────────────────┘    복사      └─────────────────────┘               │
│                                                                              │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │ Standard Attention  │    변환      │ GatedDeltaNetAttn   │               │
│  │ Q, K, V, O          │ ──────────▶  │ Q, K, V, O + Gate   │               │
│  └─────────────────────┘    + 초기화  │ + State recurrence  │               │
│                                       └─────────────────────┘               │
│                                                                              │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │ SwiGLU FFN          │    직접      │ SwiGLU FFN          │               │
│  │ gate, up, down      │ ──────────▶  │ gate, up, down      │               │
│  └─────────────────────┘    복사      └─────────────────────┘               │
│                                                                              │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │ Embedding, Norms    │    직접      │ Embedding, Norms    │               │
│  │ LM Head             │ ──────────▶  │ LM Head             │               │
│  └─────────────────────┘    복사      └─────────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 빠른 시작

### 1. 기본 사용법

```python
from olmo_core.nn.transformer import create_hybrid_from_qwen3

# Qwen3-8B → Hybrid 모델 생성
model = create_hybrid_from_qwen3(
    "Qwen/Qwen3-8B-Base",
    device="cuda",
)

# 레이어 통계 확인
print(model.get_layer_stats())
# {'deltanet_layers': 24, 'gated_attention_layers': 8, 'total_layers': 32}

# 학습
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]
```

### 2. ETD + MTP 포함 완전 통합

```python
from olmo_core.nn.transformer import create_full_hybrid_model

# Qwen3 → Hybrid + ETD + MTP
model = create_full_hybrid_model(
    "Qwen/Qwen3-8B-Base",
    use_etd=True,   # Encode-Think-Decode
    use_mtp=True,   # Multi-Token Prediction
    device="cuda",
)

# 학습
outputs = model(input_ids, labels=labels)
total_loss = outputs["loss"]  # task + mtp loss

# 추론 (MTP speculative decoding으로 ~2x 가속)
generated = model.mtp_decoder.generate(input_ids, max_new_tokens=100)
```

## Hybrid Attention 구조

### 레이어 패턴

기본적으로 4개 레이어마다 3개는 GatedDeltaNet, 1개는 GatedAttention:

```
Layer 0: GatedDeltaNet (O(n) 선형)
Layer 1: GatedDeltaNet (O(n) 선형)
Layer 2: GatedDeltaNet (O(n) 선형)
Layer 3: GatedAttention (O(n²) 표준)
Layer 4: GatedDeltaNet ...
...
```

### 패턴 커스터마이징

```python
from olmo_core.nn.transformer import HybridAttentionConfig, CrossArchTransferConfig

# 2개 DeltaNet + 2개 GatedAttention
hybrid_config = HybridAttentionConfig(
    deltanet_layer_pattern=[0, 1],  # 0, 1번 위치에 DeltaNet
    pattern_period=4,                # 4개마다 반복
)

config = CrossArchTransferConfig(
    source_model="Qwen/Qwen3-8B-Base",
    hybrid_attention=hybrid_config,
)
```

### GatedDeltaNet vs GatedAttention

| 특성 | GatedDeltaNet | GatedAttention |
|-----|---------------|----------------|
| 복잡도 | O(n) 선형 | O(n²) 제곱 |
| 장점 | 긴 시퀀스 효율적 | 정밀한 어텐션 |
| 사용처 | 대부분 레이어 | 중요 레이어 |
| 메모리 | 낮음 | 높음 |

## 가중치 전이 상세

### 직접 복사 (100% 호환)

| 컴포넌트 | 소스 | 타겟 |
|---------|------|------|
| Embedding | embed_tokens | embed_tokens |
| Q proj | self_attn.q_proj | w_q |
| K proj | self_attn.k_proj | w_k |
| V proj | self_attn.v_proj | w_v |
| O proj | self_attn.o_proj | w_out |
| FFN gate | mlp.gate_proj | gate_proj |
| FFN up | mlp.up_proj | up_proj |
| FFN down | mlp.down_proj | down_proj |
| Norms | *_layernorm | *_norm |
| LM Head | lm_head | lm_head |

### Gate 초기화 (새로 추가)

```python
config = CrossArchTransferConfig(
    gate_init_strategy="zero",  # sigmoid(0) = 0.5
    # 또는
    gate_init_strategy="small",  # 약간 열림
    gate_init_strategy="ones",   # 완전 열림 (원본과 유사)
)
```

## Knowledge Distillation (선택)

가중치 전이 후 추가로 Knowledge Distillation을 사용할 수 있습니다:

```python
from olmo_core.nn.transformer.cross_arch_transfer import DistillationTrainer

# Teacher (원본 Qwen3)와 Student (Hybrid) 준비
trainer = DistillationTrainer(
    teacher=qwen3_model,
    student=hybrid_model,
    config=CrossArchTransferConfig(
        use_distillation=True,
        distillation_temperature=2.0,
        distillation_alpha=0.5,  # KD:Task = 0.5:0.5
    ),
)

# 학습 루프
for batch in dataloader:
    losses = trainer.compute_loss(batch["input_ids"], batch["labels"])
    loss = losses["loss"]
    loss.backward()
    optimizer.step()
```

## VRAM 요구량

### Qwen3-8B → Hybrid

| 단계 | VRAM 사용량 |
|-----|------------|
| 소스 로드 (CPU) | ~0GB GPU |
| Hybrid 생성 | ~16GB |
| 전이 완료 | ~16GB |
| 학습 (BF16) | ~50-60GB |
| 추론 | ~16GB |

### 128GB VRAM에서 가능한 설정

```python
# 전체 파라미터 학습 가능!
model = create_full_hybrid_model(
    "Qwen/Qwen3-8B-Base",
    use_etd=True,
    use_mtp=True,
    device="cuda",
    dtype=torch.bfloat16,
)

# Gradient checkpointing으로 메모리 절약
model.gradient_checkpointing_enable()
```

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Full Hybrid Model Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  입력 토큰                                                                   │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Embedding                                    │    │
│  │                    (Qwen3-8B에서 복사)                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    ENCODER (8 layers, 25%)                           │    │
│  │   Layer 0-7: GatedDeltaNet + GatedAttention (3:1 비율)              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    THINK (16 layers × N회, 50%)                      │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  Dr.LLM Router: SKIP / EXECUTE / REPEAT                    │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │  MoDr LoRA Experts (4개)                                   │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   Layer 8-23: GatedDeltaNet + GatedAttention (3:1 비율)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DECODER (8 layers, 25%)                           │    │
│  │   Layer 24-31: GatedDeltaNet + GatedAttention (3:1 비율)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ├──────────────────────────────────────────────────────────────┐       │
│      ▼                                                              ▼       │
│  ┌─────────────────────────┐                    ┌─────────────────────────┐ │
│  │      Main LM Head       │                    │       MTP Head          │ │
│  │      (t_{i+1})          │                    │   (t_{i+2,3,4,5})       │ │
│  └─────────────────────────┘                    └─────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 관련 파일

- `src/olmo_core/nn/transformer/cross_arch_transfer.py`: Cross-Architecture Transfer
- `src/olmo_core/nn/attention/__init__.py`: GatedDeltaNet, GatedAttention 구현
- `src/olmo_core/nn/transformer/etd.py`: ETD + MoDr + Dr.LLM
- `src/olmo_core/nn/transformer/mtp.py`: MTP
- `src/olmo_core/nn/transformer/dna_transfer.py`: DNA Transfer 기법들

## 참고 문헌

1. **TransMamba**: arXiv:2408.09233 (Transformer → Mamba 전이)
2. **Gated DeltaNet**: arXiv:2412.06464
3. **Gated Attention**: arXiv:2505.06708
4. **Qwen3**: Alibaba (2025)
