# Continuous Latent Reasoning 가이드

CoCoNut 및 후속 연구들을 통합한 Latent Reasoning 모듈입니다.

## 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Latent Reasoning Evolution                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CoCoNut (2024.12)          개선 연구들 (2025)                              │
│  ┌─────────────────┐        ┌─────────────────────────────────────────┐    │
│  │ Multi-stage     │   →    │ CODI: Single-stage self-distillation   │    │
│  │ Curriculum      │        │       20%+ 정확도 향상                  │    │
│  └─────────────────┘        └─────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────┐        ┌─────────────────────────────────────────┐    │
│  │ Sequential      │   →    │ PCCoT: Jacobi iteration 병렬화          │    │
│  │ Forward Pass    │        │        Training/Inference 모두 가속     │    │
│  └─────────────────┘        └─────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────┐        ┌─────────────────────────────────────────┐    │
│  │ Hidden State    │   →    │ KaVa: KV-cache distillation             │    │
│  │ Alignment Only  │        │       Stepwise internal guidance        │    │
│  └─────────────────┘        └─────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────┐        ┌─────────────────────────────────────────┐    │
│  │ Full Model      │   →    │ SoftCoT: Frozen LLM + projection        │    │
│  │ Training        │        │          Catastrophic forgetting 방지   │    │
│  └─────────────────┘        └─────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 핵심 아이디어: Superposition Reasoning

**Reasoning by Superposition** (ICML 2025 Workshop):
- Continuous thought vector는 여러 reasoning path를 **superposition**으로 인코딩
- Discrete CoT가 O(n²) steps 필요한 문제를 O(n)으로 해결
- BFS-like 병렬 탐색이 latent space에서 자연스럽게 발생

```
Discrete CoT:                    Continuous Thought:
┌───┐ → ┌───┐ → ┌───┐           ┌─────────────────────┐
│ A │   │ B │   │ C │           │ A + B + C (동시에)   │
└───┘   └───┘   └───┘           │ superposition state │
Sequential (3 steps)             └─────────────────────┘
                                 Parallel (1 step)
```

## 빠른 시작

### 1. 기본 사용법

```python
from olmo_core.nn.transformer import (
    LatentReasoningConfig,
    LatentReasoningMode,
    wrap_model_with_latent_reasoning,
)

# CODI 모드 (추천)
config = LatentReasoningConfig(
    mode=LatentReasoningMode.CODI,
    num_latent_thoughts=4,
    compression_ratio=3.0,
)

model = wrap_model_with_latent_reasoning(base_model, config)

# Forward
outputs = model(input_ids, num_reasoning_steps=4)
logits = outputs["logits"]
latent_info = outputs["latent_info"]
```

### 2. Full Pipeline 통합

```python
from olmo_core.nn.transformer import create_dr_tulu_hybrid

# Dr.Tulu + Hybrid + MoE + ETD + Latent Reasoning + MTP
model = create_dr_tulu_hybrid(
    use_moe=True,
    use_etd=True,
    use_latent_reasoning=True,  # CODI 기본
    latent_mode="codi",         # or "pccot", "kava", "softcot"
    use_mtp=True,
)
```

### 3. 간편 파이프라인

```python
from olmo_core.nn.transformer import create_latent_reasoning_pipeline

# 간편하게 latent reasoning 추가
model = create_latent_reasoning_pipeline(
    base_model,
    mode="codi",
    num_latent_thoughts=4,
)
```

## Latent Reasoning 모드

### CODI (추천)

**Self-Distillation for Continuous CoT** (EMNLP 2025)

- Same model이 teacher (explicit CoT)와 student (implicit) 역할
- Answer token의 hidden state alignment
- Single-stage training (no curriculum)
- **GSM8K에서 CoCoNut 대비 20%+ 정확도 향상**

```python
config = LatentReasoningConfig(
    mode=LatentReasoningMode.CODI,
    num_latent_thoughts=4,
    compression_ratio=3.0,  # CoT → Latent 압축 비율
    distillation=DistillationConfig(
        temperature=2.0,
        alpha=0.5,
        hidden_alignment_weight=1.0,
    ),
)
```

### PCCoT

**Parallel Continuous CoT with Jacobi Iteration** (EMNLP 2025)

- Jacobi iteration으로 latent thoughts 병렬 업데이트
- Training/Inference 모두 병렬화
- 분산 감소로 안정적 학습

```python
config = LatentReasoningConfig(
    mode=LatentReasoningMode.PCCOT,
    num_latent_thoughts=4,
    jacobi_iterations=3,
    jacobi_tolerance=1e-4,
    parallel_thoughts=True,
)
```

### KaVa

**KV-Cache Distillation** (October 2025)

- Teacher의 reasoning trace를 KV-cache로 distill
- Stepwise internal guidance
- CODI, PCCoT 대비 GSM8K 성능 향상

```python
config = LatentReasoningConfig(
    mode=LatentReasoningMode.KAVA,
    distillation=DistillationConfig(
        kv_cache_weight=0.5,
    ),
)
```

### SoftCoT

**Soft Token Projection** (ACL 2025)

- 경량 assistant model이 soft thought tokens 생성
- Frozen LLM + projection layer만 학습
- Catastrophic forgetting 방지
- **SoftCoT++**: Test-time scaling via contrastive learning

```python
config = LatentReasoningConfig(
    mode=LatentReasoningMode.SOFTCOT,
    freeze_base_model=True,
    projection_dim=256,
    use_contrastive=True,  # SoftCoT++
)
```

## 모드 비교

| 모드 | 장점 | 단점 | 추천 사용 |
|-----|-----|-----|---------|
| **CODI** | 가장 높은 정확도, 간단 | Teacher 필요 | 일반 추천 |
| **PCCoT** | 병렬화, 빠른 학습 | 수렴 불안정 | 대규모 배치 |
| **KaVa** | 풍부한 supervision | KV-cache 메모리 | 정밀 추론 |
| **SoftCoT** | Forgetting 방지 | 별도 모델 필요 | SOTA LLM 적용 |

## Interpretability

### Latent Decoding

```python
from olmo_core.nn.transformer import decode_latent_to_tokens

# Latent thoughts를 tokens으로 decoding
thoughts = outputs["latent_info"]["thought_history"][-1]
embed_matrix = model.get_input_embeddings().weight

top_k_tokens = decode_latent_to_tokens(
    thoughts,
    embed_matrix,
    top_k=5,
)

print(tokenizer.decode(top_k_tokens[0][0]))  # Top-1 token for first thought
```

**주의**: 이 decoding이 실제 reasoning을 faithfully represent하지 않을 수 있습니다.

### Superposition Entropy

```python
from olmo_core.nn.transformer import compute_superposition_entropy

entropy = compute_superposition_entropy(thoughts)
print(f"Superposition entropy: {entropy:.4f}")
# 높을수록 더 많은 reasoning paths가 인코딩됨
```

## Training Tips

### 1. CODI Self-Distillation

```python
# Training loop
for batch in dataloader:
    # Teacher mode: explicit CoT
    teacher_outputs = model(
        batch["input_ids"],
        batch["cot_ids"],
        mode="teacher",
    )

    # Student mode: implicit latent
    student_outputs = model(
        batch["input_ids"],
        mode="student",
    )

    # Distillation loss
    loss = model.latent_think.distillation.compute_distillation_loss(
        teacher_outputs["hidden_states"],
        student_outputs["hidden_states"],
    )

    loss["total_loss"].backward()
```

### 2. PCCoT Jacobi 안정화

```python
# Jacobi 수렴 모니터링
outputs = model(input_ids, return_thoughts=True)
jacobi_info = outputs["latent_info"]

if not jacobi_info["converged"]:
    # 수렴 실패시 iteration 증가
    model.latent_think.jacobi_solver.max_iterations += 1
```

### 3. Curriculum (선택적)

CODI는 curriculum이 필요 없지만, 점진적 학습도 가능:

```python
# Stage 1: 1 latent thought
config.num_latent_thoughts = 1

# Stage 2: 2 latent thoughts
config.num_latent_thoughts = 2

# Stage 3: 4 latent thoughts
config.num_latent_thoughts = 4
```

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Full Model with Latent Reasoning                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  입력 토큰                                                                   │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Hybrid Transformer (GatedDeltaNet + GatedAttention)                │    │
│  │  + MoE (8 experts)                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ETD THINK Block                                                    │    │
│  │  ┌───────────────────────────────────────────────────────────┐     │    │
│  │  │  Latent Reasoning (CODI/PCCoT/KaVa)                       │     │    │
│  │  │                                                            │     │    │
│  │  │  ┌────────────────┐                                       │     │    │
│  │  │  │ Continuous     │ ← Hidden state feedback               │     │    │
│  │  │  │ Thoughts (4)   │ ← Superposition encoding              │     │    │
│  │  │  └────────────────┘                                       │     │    │
│  │  │         │                                                  │     │    │
│  │  │         ▼ (PCCoT: Jacobi iteration)                       │     │    │
│  │  │  ┌────────────────┐                                       │     │    │
│  │  │  │ Parallel       │                                       │     │    │
│  │  │  │ Update         │                                       │     │    │
│  │  │  └────────────────┘                                       │     │    │
│  │  └───────────────────────────────────────────────────────────┘     │    │
│  │                                                                     │    │
│  │  + MoDr LoRA Experts                                               │    │
│  │  + Dr.LLM Router                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MTP Head (4 future tokens)                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 참고 문헌

1. **CoCoNut**: arXiv:2412.06769 (Meta FAIR, 2024.12)
2. **CODI**: EMNLP 2025 (Self-distillation)
3. **PCCoT**: EMNLP 2025 (Jacobi iteration)
4. **KaVa**: October 2025 (KV-cache distillation)
5. **SoftCoT/SoftCoT++**: ACL 2025
6. **Reasoning by Superposition**: MOSS@ICML 2025
7. **Huginn**: ICML 2025 Spotlight (Recurrent depth)
8. **Ouro/LoopLM**: 2025 (Pretraining-phase latent reasoning)

## 관련 파일

- `src/olmo_core/nn/transformer/latent_reasoning.py`: Latent Reasoning 모듈
- `src/olmo_core/nn/transformer/etd.py`: ETD Think block (통합)
- `src/olmo_core/nn/transformer/cross_arch_transfer.py`: Full pipeline
