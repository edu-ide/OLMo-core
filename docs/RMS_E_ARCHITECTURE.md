# RMS-E: Recursive Memory-Sparse Experts Architecture

**Next-Generation Small Model Architecture Design**

---

## 1. 개요 (Overview)

**RMS-E (Recursive Memory-Sparse Experts)**는 "작은 모델로 거대 모델의 성능을 낸다"는 목표 아래 설계된 차세대 소형 언어 모델 아키텍처입니다. 기존의 깊은 레이어 적층(Stacking Layers) 방식에서 벗어나, **재귀적 루프(Recursive Loop)**와 **희소 전문가(Sparse Experts)**, 그리고 **무한 메모리(Infinite Memory)**를 결합하여 파라미터 효율성을 극대화했습니다.

### 핵심 철학: "Small Loop, Infinite Context"
*   **Layerless**: 수십 개의 레이어를 쌓는 대신, 하나의 강력한 루프 블록을 재사용합니다.
*   **Memory-Centric**: 모델의 크기가 아닌, 메모리의 크기를 키워 지능을 확장합니다.
*   **Adaptive**: 문제는 난이도에 따라 필요한 만큼만 생각하고(Loop), 스스로 멈춥니다(SBT).

---

## 2. 전체 아키텍처 (Full Architecture)

![RMS-E Architecture](https://mermaid.ink/img/JSV7aW5pdDogeyd0aGVtZSc6ICduZXV0cmFsJywgJ3RoZW1lVmFyaWFibGVzJzogeyAnZm9udEZhbWlseSc6ICdUaW1lcyBOZXcgUm9tYW4nLCAnZm9udFNpemUnOiAnMTRweCcsICdwcmltYXJ5Q29sb3InOiAnI2ZmZicsICdlZGdlTGFiZWxCYWNrZ3JvdW5kJzonI2ZmZicsICdjbHVzdGVyQmtnJzogJyNmOWY5ZjknLCAnY2x1c3RlckJvcmRlcic6ICcjNjY2JyB9fX0lJQpncmFwaCBTRQogICAgSW5wdXRbSW5wdXQgU2VxdWVuY2VdIC0tPiBBdGxhc1tBdGxhcyBSZXRyaWV2YWw8YnIvPihTcGFyc2UgVHJpZ2dlclddXQogICAgQXRsYXMgLS0+IFtitGFuc1tUaXRhbnMgTWVtb3J5PGJyLz4oTmV1cmFsIE1lbW9yeSldCiAgICAKICAgIHN1YmdyYXBoIFJtc0RCIFsiUk1TLUUgQ29yZSAoUmVjdXJzaXZlIEJsb2NrKSJdCiAgICAgICAgZGlyZWN0aW9uIFRCCiAgICAgICAgU3RhdGVbSHlkZGVuIFN0YXRlIGhcX3RdCiAgICAgICAgU2hhcmVkW1NoYXJlZCBFeHBlcnRzPGJyLz4oU3RhYmlsaXR5KV0KICAgICAgICBSb3V0ZWRbUm91dGVkIEV4cGVydHM8YnIvPihkaXZlcnNpdHkgMTAvNTEyKV0KICAgICAgICBTQlRbU0JUIEdhdGU8YnIvPihDb252ZXJnZW5jZSBDaGVjayldCiAgICAgICAgCiAgICAgICAgU3RhdGUgLS0+IFNoYXJlZCAmIFJvdXRlZAogICAgICAgIFNoYXJlZCAmIFJvdXRlZCAtLT4gU3RhdGUKICAgICAgICBTdGF0ZSAtLi0+IFNCVAogICAgZW5kCiAgICAKICAgIFRpdGFucyAtLT4gU3RhdGUKICAgIFNCVCAtLT58Q29udmVyZ2VkfCBPdXRwdXRbT3V0cHV0IFRva2VuXQogICAgU0JUIC0tPnxCcmFrZV9GYWlsfCBTdGF0ZQ==)

<!-- Mermaid Code
```mermaid
graph TD
    Input[Input Sequence] --> Atlas[Atlas Retrieval<br/>(Surprise Trigger)]
    Atlas --> Titans[Titans Memory<br/>(Neural Memory)]
    
    subgraph RmsBlock ["RMS-E Core (Recursive Block)"]
        direction TB
        State[Hidden State h_t]
        Shared[Shared Experts<br/>(Stability)]
        Routed[Routed Experts<br/>(Diversity 10/512)]
        SBT[SBT Gate<br/>(Convergence Check)]
        
        State --> Shared & Routed
        Shared & Routed --> State
        State -.- SBT
    end
    
    Titans --> State
    SBT -->|Converged| Output[Output Token]
    SBT -->|Continue| State
```
-->

---

## 3. 핵심 컴포넌트 상세 (Core Components)

### 3.1 LoopLM (Ouro 기반 재귀 구조)
*   **개념**: 기존 40층 모델을 **단일 블록의 40회 반복**으로 대체합니다.
*   **Latent Reasoning**: 텍스트(Token)를 생성하며 생각하는 것이 아니라, **잠재 공간(Latent Space)** 내에서 순환하며 추론합니다. 이는 `<think>` 토큰을 생성하는 비용을 0으로 만듭니다.
*   **이점**: 파라미터 공유로 인해 모델 사이즈가 1/10로 줄어듭니다 (10B 성능을 1B로 구현).

### 3.2 Memory-Sparse Experts (Qwen3-Next 기반 MoE)
*   **구조**: 재귀 루프 내부의 FFN(Feed-Forward Network)을 **Ultra-Sparse MoE**로 대체합니다.
*   **전문가 구성**:
    *   **Shared Experts**: 모든 루프에서 항상 활성화 (문맥 유지, 안정성).
    *   **Routed Experts**: 루프 단계($k$)마다 다르게 호출되는 전문가 (512개 중 상위 10개).
        *   *$k=1$: "이해" 전문가*
        *   *$k=5$: "추론" 전문가*
        *   *$k=10$: "정리" 전문가*
*   **효과**: 단일 블록이지만 루프 횟수마다 다른 전문가를 사용하여, 마치 서로 다른 레이어를 통과하는 듯한 **Function Diversity**를 확보합니다.

### 3.3 Titans (Neural Long-term Memory)
*   **역할**: Transformer의 Attention 윈도우 한계를 극복하는 **반영구적 기억 장치**입니다.
*   **작동**:
    *   입력 토큰을 **Neural Parameters ($M$)**로 압축하여 저장합니다.
    *   루프가 돌 때마다 이 메모리($M$)를 참조(Query)하여 과거의 문맥을 불러옵니다.
*   **성능**: **10M+ 토큰**의 컨텍스트를 처리하며, KV Cache 없이도 긴 대화를 기억합니다.

### 3.4 Atlas (Dynamic Retrieval Trigger)
*   **트리거**: 입력의 **놀라움(Surprise Score)**이 높을 때(즉, 모델이 모르는 내용일 때)만 작동합니다.
*   **기능**: 외부 벡터 DB에서 관련 지식을 검색하여 Loop의 초기 상태($h_0$)에 주입합니다.
*   **효과**: 모델 파라미터에 모든 지식을 저장할 필요가 없어, 모델 크기를 **2B 이하**로 유지하면서도 **100B급 지식**을 활용합니다.

---

## 4. 최적화 전략 (Optimization Strategy)

### 4.1 SBT (Self-Braking Tuning) - 오버띵킹 방지
기존 Ouro나 CoT 모델의 문제점인 "무한 루프"나 "과도한 생각(Overthinking)"을 방지하는 핵심 기술입니다.
*   **Braking Gate**: 루프가 반복될 때마다 은닉 상태의 변화량($\Delta h$)을 감시합니다.
*   **메커니즘**:
    *   `if` $\Delta h < \epsilon$ (수렴) or `Brake_Score` > Threshold:
    *   `then` **EXIT Loop** -> Generate Token.
*   **성능**: 불필요한 연산을 줄여 평균 추론 속도를 **30% 이상 향상**시킵니다.

### 4.2 Entropy Regularization (학습 단계)
*   모델이 항상 최대 횟수(예: 40회)만 루프를 돌려고 하는 경향을 막기 위해, 루프 종료 확률의 엔트로피를 정규화 항으로 Loss에 추가합니다. 이는 모델이 문제 난이도에 따라 **가변적인 깊이**를 갖도록 훈련시킵니다.

---

## 5. 성능 비교 (Performance Comparison)

| 특징 | Ouro (LoopLM) | Qwen3-Next | Titans | **RMS-E (제안)** |
| :--- | :--- | :--- | :--- | :--- |
| **파라미터 효율** | 높음 | 중간 (MoE) | 중간 | **최상** (재귀 + 희소성) |
| **추론 방식** | Latent Loop | Token Generation | Standard | **SBT Latent Loop** |
| **기억 용량** | 제한적 | 제한적 | 무한 | **무한 (Neural Memory)** |
| **지식 확장성** | 낮음 | 낮음 | 낮음 | **높음 (Atlas RAG)** |
| **오버띵킹 제어** | 엔트로피 | Budget 토큰 | N/A | **SBT (자율 제동)** |
| **목표 사이즈** | < 7B | 7B+ | 7B+ | **< 2B (On-Device)** |

---

## 6. 결론 (Conclusion)

RMS-E 아키텍처는 **Ouro의 재귀성**, **Qwen3의 효율성**, **Titans의 기억력**, **Atlas의 지식**, **SBT의 제어력**을 하나로 통합한 결정체입니다. 이는 단순한 경량화가 아니라, 지능의 밀도를 높이는 방향으로의 진화이며, **온디바이스 AI 및 실시간 엣지 컴퓨팅**을 위한 최적의 솔루션입니다.
