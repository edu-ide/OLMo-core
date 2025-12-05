# Full Architecture: Triple-Hybrid + ETD + LaDiR + Block Diffusion + MTP

**차세대 LLM을 위한 통합 아키텍처**

---

## 1. 전체 아키텍처 개요

### 핵심 개념: 수평 데이터 흐름 + 수직 레이어 스택

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FULL ARCHITECTURE (데이터 흐름)                          │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        INPUT SEQUENCE                                 │  │
│  │  [Question Tokens] ──────────────────────────────────────────────────│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  BACKBONE + ETD (Triple-Hybrid layers organized by ETD)              │  │
│  │  ════════════════════════════════════════════════════════════════════│  │
│  │                                                                      │  │
│  │   ENCODER (8)        THINK (24, N회 반복)        DECODER (8)         │  │
│  │  ┌──────────┐      ┌──────────────────┐      ┌──────────┐           │  │
│  │  │ M3│GD│GD │  →   │ M3│M3│GD│GD│GA│GD│  →   │ GA│GA│GD │           │  │
│  │  │ GD│GA│GD │      │    ... x24       │      │ GA│M3│GA │           │  │
│  │  │ M3│GA│   │      │ +Router +Expert  │      │ GD│M3│   │           │  │
│  │  └──────────┘      └──────────────────┘      └──────────┘           │  │
│  │  DeltaNet 위주       균형 (4:4:2)            Attention 위주          │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  LaDiR: LATENT REASONING (시퀀스 끝에 Memory Slots 추가)              │  │
│  │  ════════════════════════════════════════════════════════════════════│  │
│  │                                                                      │  │
│  │   [Hidden States] + [MEM₁] [MEM₂] [MEM₃]  ← 3개 Memory Slots         │  │
│  │                          │                                           │  │
│  │                          ▼                                           │  │
│  │   ┌────────────────────────────────────────────────────────────┐    │  │
│  │   │  VAE Encoder: Memory Slots → Latent z (압축)               │    │  │
│  │   │  Flow Matching: Noise → Latent z (생성)                    │    │  │
│  │   │  Prophet: Confidence gap 기반 조기 종료                    │    │  │
│  │   │  VAE Decoder: Latent z → Memory Slots (복원)               │    │  │
│  │   └────────────────────────────────────────────────────────────┘    │  │
│  │                          │                                           │  │
│  │   [Hidden States with Reasoning] ← Memory Slots가 추론 정보 담음     │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  OUTPUT ACCELERATION (MTP / Block Diffusion)                         │  │
│  │  ════════════════════════════════════════════════════════════════════│  │
│  │                                                                      │  │
│  │   ┌─────────────────────┐    ┌─────────────────────┐                │  │
│  │   │   MTP Head          │ OR │   Block Diffusion   │                │  │
│  │   │   n개 토큰 동시 예측 │    │   AR+Diffusion 혼합 │                │  │
│  │   └─────────────────────┘    └─────────────────────┘                │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        OUTPUT TOKENS                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

M3 = Mamba-3, GD = Gated DeltaNet, GA = Gated Attention
```

### CoCoNut vs LaDiR: 위치는 같고, 방식만 다름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CoCoNut vs LaDiR 비교                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CoCoNut (Original)                                                 │   │
│  │  ───────────────────                                                │   │
│  │                                                                     │   │
│  │  [Question] + [<BOT> T₁ T₂ T₃ ... Tₙ <EOT>]                        │   │
│  │                       └─────────────────┘                           │   │
│  │                       N개 Thought Tokens                            │   │
│  │                       (시퀀스 END에 위치)                           │   │
│  │                                                                     │   │
│  │  생성: Autoregressive (한 토큰씩 순차 생성)                         │   │
│  │  학습: Curriculum Learning 필요                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LaDiR (Latent Diffusion Reasoning)                                 │   │
│  │  ──────────────────────────────────                                 │   │
│  │                                                                     │   │
│  │  [Question] + [MEM₁] [MEM₂] [MEM₃]                                 │   │
│  │                └────────────────┘                                   │   │
│  │                3개 Memory Slots                                     │   │
│  │                (시퀀스 END에 위치) ← 위치는 CoCoNut과 동일!         │   │
│  │                                                                     │   │
│  │  생성: Diffusion (병렬 생성, Prophet 조기 종료)                     │   │
│  │  학습: VAE → Diffusion 2단계 학습                                   │   │
│  │  압축률: ~512x (긴 CoT를 3개 토큰으로 압축)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ⚠️ 핵심 차이:                                                              │
│  • 위치: 둘 다 시퀀스 END (동일)                                           │
│  • 토큰 수: CoCoNut N개 vs LaDiR 3개 (압축)                                │
│  • 생성 방식: AR vs Diffusion                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ETD와 LaDiR의 관계 (수정됨)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          관계 다이어그램 (수정)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Triple-Hybrid: "레이어 타입" 제공 (Mamba-3 / DeltaNet / Attention)         │
│       │                                                                     │
│       ▼                                                                     │
│  ETD: "레이어 조직화" (Encoder / Think / Decoder)                           │
│       │                                                                     │
│       │   ← ETD 내부에서는 레이어만 반복, LaDiR은 여기 없음                  │
│       │                                                                     │
│       ▼                                                                     │
│  LaDiR: "시퀀스 끝에서" Latent Reasoning 수행                               │
│       │                                                                     │
│       │   ← ETD 처리 후, 시퀀스 끝에 Memory Slots 추가                      │
│       │                                                                     │
│       ▼                                                                     │
│  Output: MTP 또는 Block Diffusion으로 출력 가속                             │
│                                                                             │
│  ⚠️ 수정된 핵심 포인트:                                                      │
│  • ETD: 레이어 반복 (Think 블록 N회 실행)                                   │
│  • LaDiR: 시퀀스 끝에서 작동 (ETD 내부 아님!)                               │
│  • 둘은 순차적 관계 (ETD → LaDiR → Output)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 각 레이어 상세

### Layer 1: Backbone (Triple-Hybrid) - "어떤 레이어?"

| 컴포넌트 | 파일 | 역할 | 메모리 타입 | 비율 |
|---------|------|------|------------|------|
| **Mamba-3** | `mamba_memory.py` | 심층 장기 기억 | 복소수 상태 공간 | 40% |
| **Gated DeltaNet** | `gated_deltanet.py` | 고속 작업 기억 | 연관 행렬 (d×d) | 40% |
| **Gated Attention** | `attention/` | 집중 주의 | KV Cache | 20% |

### Layer 2: ETD Structure - "어떻게 구조화?"

| 컴포넌트 | 역할 | 특징 |
|---------|------|------|
| **Encoder** | 입력 처리 | 1회 실행 |
| **Think** | 사고/추론 | N회 반복 (핵심) |
| **Decoder** | 출력 생성 | 1회 실행 |
| **Router** | 동적 깊이 | Skip/Execute/Repeat |
| **MoDr** | 전문가 분기 | LoRA 기반 |

### Layer 3: LaDiR Reasoning - "어떻게 추론?"

| 컴포넌트 | 역할 | 특징 |
|---------|------|------|
| **LaDiRVAE** | 압축 | Reasoning → Latent |
| **LaDiRDiffusion** | 생성 | Flow Matching |
| **ProphetEarlyExit** | 가속 | Confidence gap 기반 조기 종료 |

### Output Acceleration - "어떻게 출력?"

| 컴포넌트 | 파일 | 역할 | 속도 향상 |
|---------|------|------|----------|
| **MTP** | `mtp.py` | n개 토큰 동시 예측 | 2-3x |
| **Block Diffusion** | `block_diffusion.py` | AR+Diffusion 하이브리드 | 2.5x (vs AR) |

---

## 3. 데이터 흐름

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ENCODER (ETD)                              │
│  Triple-Hybrid layers: mostly DeltaNet for local patterns       │
│  [DeltaNet] → [DeltaNet] → [Mamba] → [DeltaNet] → ...          │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       THINK (ETD)                               │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Iteration 1-N (N회 반복)                                  ││
│  │  ┌────────────────────────────────────────────────────┐   ││
│  │  │ Triple-Hybrid Block                                │   ││
│  │  │ [Mamba-3] → [DeltaNet] → [Attention] → ...         │   ││
│  │  │     +Router (Skip/Execute/Repeat)                  │   ││
│  │  │     +LoRA Experts (MoDr)                           │   ││
│  │  └────────────────────────────────────────────────────┘   ││
│  │                                                            ││
│  │  ⚠️ LaDiR은 여기 없음! (ETD 내부 아님)                      ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DECODER (ETD)                              │
│  Triple-Hybrid layers: Attention + DeltaNet for output          │
│  [Attention] → [DeltaNet] → [Attention] → ...                  │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LaDiR (시퀀스 END에 위치)                      │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  [Hidden States] + [MEM₁] [MEM₂] [MEM₃]  ← Memory Slots    ││
│  │                                                            ││
│  │  Training: VAE로 CoT 압축 학습                              ││
│  │  Inference: Diffusion + Prophet으로 Memory 생성             ││
│  │                                                            ││
│  │  ⚠️ ETD 이후에 적용! (ETD 내부 아님)                         ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT GENERATION                             │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │   MTP Head          │  │   Block Diffusion    │              │
│  │   (4 tokens/step)   │  │   (parallel blocks)  │              │
│  └─────────────────────┘  └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
Output Tokens
```

---

## 4. 파일 구조

```
OLMo-core/src/olmo_core/nn/transformer/
│
├── [BACKBONE - Layer 1: Triple-Hybrid]
│   │
│   ├── mamba_memory.py          # 🆕 Mamba-3 (Deep Long-term Memory)
│   │   ├── MambaMemoryConfig
│   │   ├── MambaMemory
│   │   └── MambaMemoryBlock
│   │
│   ├── gated_deltanet.py        # ✅ 기존 (Working Memory)
│   │   ├── GatedDeltaNetConfig
│   │   ├── GatedDeltaNet
│   │   └── GatedDeltaNetBlock
│   │
│   ├── attention/               # ✅ 기존 (Focused Attention)
│   │   ├── __init__.py
│   │   ├── flash_attn_api.py
│   │   ├── kv_cache.py
│   │   └── ...
│   │
│   └── triple_hybrid.py         # 🆕 통합 (Layer 조합)
│       ├── TripleHybridConfig
│       ├── TripleHybridBlock
│       └── LayerScheduler
│
├── [ETD STRUCTURE - Layer Organization]
│   │
│   └── etd.py                   # ✅ 업데이트 (ETD + Triple-Hybrid 통합)
│       ├── ETDConfig            (기존 ETD 설정)
│       ├── ETDTransformer       (기존 블록 래핑)
│       ├── ThinkBlockController (적응형 깊이)
│       ├── LayerRouter          (Skip/Execute/Repeat)
│       ├── MoDrExpertRouter     (LoRA experts)
│       │
│       ├── [ETD + Triple-Hybrid 통합] 🆕
│       ├── BackboneType         (STANDARD / TRIPLE_HYBRID)
│       ├── ETDTripleHybridConfig (통합 설정)
│       ├── ETDTripleHybridTransformer (E-T-D + Triple-Hybrid)
│       └── create_etd_triple_hybrid_model() (헬퍼 함수)
│
├── [REASONING - Layer 3: LaDiR] (시퀀스 END에서 작동!)
│   │
│   └── latent_reasoning.py      # ✅ 수정 (LaDiR만 유지)
│       ├── LaDiRConfig          (VAE + Diffusion 설정)
│       ├── ProphetConfig        (조기 종료 설정)
│       ├── LaDiRVAE             (Text → Latent 압축)
│       ├── LaDiRDiffusion       (Flow Matching 생성)
│       ├── FlowMatchingScheduler (Noise scheduler)
│       ├── ProphetEarlyExit     (Confidence gap 조기 종료)
│       ├── LaDiRModule          (🆕 시퀀스 END에서 작동)
│       └── LatentReasoningWrapper (🆕 ETD 출력 후 LaDiR 적용)
│
├── [OUTPUT ACCELERATION]
│   │
│   ├── mtp.py                   # ✅ 기존 (Multi-Token Prediction)
│   │   ├── MTPConfig
│   │   ├── MTPHead
│   │   ├── MTPLoss
│   │   └── MTPSpeculativeDecoder
│   │
│   └── block_diffusion.py       # ✅ 기존 (Block Discrete Diffusion)
│       ├── BlockDiffusionConfig
│       ├── NoiseSchedule
│       └── BlockDiffusionLayer
│
├── [CORE]
│   ├── block.py                 # 기본 Transformer block
│   ├── model.py                 # Transformer 모델
│   └── config.py                # 설정
│
├── [DEPRECATED - 제거 예정]
│   │
│   └── latent_reasoning.py 에서 제거:
│       ├── ContinuousThought
│       ├── CODIDistillation
│       ├── PCCoTJacobiSolver
│       ├── KaVaDistillation
│       ├── SoftCoTProjection
│       └── LatentReasoningMode (COCONUT, CODI, PCCOT, KAVA, SOFTCOT)
│
└── [DOCS]
    └── TRIPLE_HYBRID_ARCHITECTURE.md  # 이 문서
```

---

## 5. 통합 Configuration

```python
from dataclasses import dataclass, field
from olmo_core.config import Config

@dataclass
class FullArchitectureConfig(Config):
    """전체 아키텍처 통합 설정"""

    # === Layer 1: Triple-Hybrid Backbone ===
    backbone: TripleHybridConfig = field(default_factory=lambda: TripleHybridConfig(
        d_model=2048,
        n_layers=40,
        mamba_ratio=0.4,      # 40% Mamba-3 (장기 기억)
        deltanet_ratio=0.4,   # 40% DeltaNet (작업 기억)
        attention_ratio=0.2,  # 20% Attention (정밀 검색)
    ))

    # === Layer 2: ETD Structure ===
    etd: ETDConfig = field(default_factory=lambda: ETDConfig(
        num_encoder_layers=8,
        num_think_layers=24,   # Think 블록이 대부분
        num_decoder_layers=8,
        think_iterations=4,    # Think 반복 횟수
        use_router=True,       # 동적 라우팅
    ))

    # === Layer 3: LaDiR Reasoning ===
    ladir: LaDiRConfig = field(default_factory=lambda: LaDiRConfig(
        latent_dim=128,
        num_memory_slots=3,
        num_inference_steps=50,
        prophet=ProphetConfig(
            enabled=True,
            threshold_early=7.5,
            threshold_mid=5.0,
            threshold_late=2.5,
        ),
    ))

    # === Output Acceleration ===
    mtp: MTPConfig = field(default_factory=lambda: MTPConfig(
        num_predict_tokens=4,  # 4 토큰 동시 예측
        use_speculative=True,
    ))

    block_diffusion: BlockDiffusionConfig = field(default_factory=lambda: BlockDiffusionConfig(
        block_size=16,
        noise_schedule="log_linear",
    ))
```

---

## 6. Implementation Roadmap

### Phase 1: Backbone (Triple-Hybrid) ✅ COMPLETED
- [x] `mamba_memory.py` 생성 - Mamba-3 (Complex SSM) 구현
- [x] `triple_hybrid.py` 생성 - 4:4:2 비율 레이어 조합
- [x] 기존 `gated_deltanet.py` 연동 확인

### Phase 2: ETD Integration ✅ COMPLETED
- [x] `etd.py`와 Triple-Hybrid 통합
  - [x] `BackboneType` enum 추가
  - [x] `ETDTripleHybridConfig` 생성
  - [x] `ETDTripleHybridTransformer` 구현
  - [x] `create_etd_triple_hybrid_model()` 헬퍼 함수
- [x] Think Block에서 Triple-Hybrid 레이어 사용하도록 수정
  - [x] 섹션별 다른 레이어 비율 지원 (Encoder/Think/Decoder)
  - [x] LayerRouter + MoDrExpert 통합

### Phase 3: LaDiR Cleanup ✅ COMPLETED
- [x] `latent_reasoning.py` deprecated 코드 제거
  - [x] `ContinuousThought` 제거
  - [x] `CODIDistillation` 제거
  - [x] `PCCoTJacobiSolver` 제거
  - [x] `KaVaDistillation` 제거
  - [x] `SoftCoTProjection` 제거
  - [x] 관련 Mode enum 정리 (LADIR만 유지)
- [x] LaDiR 전용으로 정리

### Phase 4: Output Layer Integration
- [ ] `mtp.py` 연동 확인
- [ ] `block_diffusion.py` 연동 확인

### Phase 5: Full Integration & Testing
- [ ] 전체 파이프라인 통합
- [ ] Configuration 통합 클래스 생성
- [ ] 단위/통합 테스트
- [ ] 성능 벤치마크

---

## 7. Summary

```
┌────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE SUMMARY                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Q: 어떤 레이어?     → Triple-Hybrid (Mamba + DeltaNet + Attn) │
│  Q: 어떻게 구조화?   → ETD (Encode-Think-Decode)               │
│  Q: 어떻게 추론?     → LaDiR (VAE + Flow + Prophet)            │
│  Q: 어떻게 출력?     → MTP + Block Diffusion                   │
│                                                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                │
│  Files:                                                        │
│  ├─ Backbone                                                   │
│  │  • mamba_memory.py      (✅ 완료 - Mamba-3 Complex SSM)     │
│  │  • gated_deltanet.py    (✅ 기존)                           │
│  │  • triple_hybrid.py     (✅ 완료 - 4:4:2 조합)              │
│  │                                                             │
│  ├─ Structure                                                  │
│  │  • etd.py               (✅ 기존)                           │
│  │                                                             │
│  ├─ Reasoning                                                  │
│  │  • latent_reasoning.py  (✅ 완료 - LaDiR only)              │
│  │                                                             │
│  └─ Output                                                     │
│     • mtp.py               (✅ 기존)                           │
│     • block_diffusion.py   (✅ 기존)                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. References

1. **Mamba-3**: "Improved Sequence Modeling using State Space Principles" (2025)
2. **Gated DeltaNet**: "Improving Mamba2 with Delta Rule" (arXiv:2412.06464)
3. **ETD**: "Encode, Think, Decode" (arXiv:2510.07358)
4. **LaDiR**: "Latent Diffusion for Reasoning" (arXiv:2510.08558)
5. **Prophet**: "Early Commit Decoding" (arXiv:2508.19982)
6. **BD3-LM**: "Block Discrete Denoising Diffusion" (arXiv:2503.09573, ICLR 2025)
7. **MTP**: "Multi-token Prediction" (arXiv:2404.19737, Meta)
8. **Triple-Hybrid Report**: 내부 아키텍처 분석 문서

---

*문서 작성일: 2025-12-05*
*작성자: AI Architect*
