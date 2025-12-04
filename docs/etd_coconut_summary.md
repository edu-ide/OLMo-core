# ETD vs Coconut: 요약 및 작업 맥락

이 문서는 ETD(Encode-Think-Decode)와 Coconut(Chain of Continuous Thought) 관련 논의, 장단점, 요구사항, 그리고 현재 OLMo-core에 적용된 변경 사항을 정리한 메모입니다. 다른 에이전트/작업자와 협업 시 참고용입니다.

## 핵심 차이
- **ETD**: 표준 토큰 기반, 생각 블록을 반복해 깊은 추론. 아키텍처 변경 없이 프롬프트/토큰 패턴(예: `<think>` 섹션)만 추가해도 됨. 정밀 추론(수학/코딩)에 유리하지만 연산/시간은 증가.
- **Coconut**: 중간 추론을 latent(hidden)로 돌림. 토큰 비용/속도는 유리하지만 generate 루프/입력 경로를 수정해야 하고, 보조 로스/커리큘럼 없이 불안정할 수 있음. 디버깅도 어려움.

## 데이터 재사용/전처리
- **ETD**: 기존 텍스트 데이터셋 그대로 사용 가능. 생각 구간을 명시하면 효과↑. CoT가 없어도 동작.
- **Coconut**: 데이터는 재사용 가능하지만 추가가 필요:
  - latent_generate 루프(토큰 대신 hidden 피드백) 구현
  - 커리큘럼: 명시적 CoT → 부분 latent → latent-only
  - 보조 로스(선택): SIM-CoT(보조 디코더로 latent→텍스트), Latent-SFT(soft embedding), 엔트로피/노름 정규화
  - BFS/beam 하이퍼 설정이 필요할 수 있음

## 혼합/라우팅 전략
- 가장 안전: **모드 전환**(제어 토큰/옵션으로 ETD 또는 Coconut만 실행).
- MoE식 라우팅으로 자동 선택은 가능하지만 구현/튜닝 부담↑.
- 하이브리드 실험은 작은 스케줄(예: 일부 스텝만 latent)부터 시작 권장.

## 최신 연구에서 나온 완화책(요약)
- **Gated DeltaNet**: 하이브리드(일부 softmax 유지), zero-centered norm+WD, 청크 병렬 DeltaNet.
- **Gated Attention Output**: 헤드별 스칼라 게이팅으로 sink 완화/양자화 친화.
- **MoD**: 정적 capacity/top-k, A-MoD(어텐션 맵 재활용), GateSkip(soft→hard 스킵), 보조 로스.
- **MoR**: KV 공유, 라우터 e2e 학습, retrofit 라우팅.
- **Coconut 안정화**: SIM-CoT, Latent-SFT, CoT→latent 커리큘럼.
- **동적 연산**: shared expert, aux-loss-free routing bias, 제어 토큰(ThinkLess/MixReasoning).

## 현재 OLMo-core에 적용된 변경(브랜치: feat/gated-deltanet)
- 새 프리셋:
  - `TransformerConfig.olmo3_7B_hybrid_gated`: 일부 레이어 softmax, 나머지 `gated_output` 어텐션.
  - `TransformerConfig.olmo3_7B_hybrid_moe`: 주기적으로 MoE FFN(top_k, num_experts, shared expert).
- 새 어텐션 유형:
  - `gated_output` (헤드별 시그모이드 게이트)
  - `gated_deltanet` (실험적 선형 어텐션, TP/CP 미구현)
- 문서:
  - `docs/hybrid_attention_moe.md` (하이브리드 어텐션/MoE 의도·사용법)
  - `docs/coconut_latent_usage.md` (Coconut 데이터 재활용/커리큘럼 가이드)
  - 본 메모 `docs/etd_coconut_summary.md`

## 제안된 실험 순서
1) ETD 전용: 생각 토큰 프롬프트만으로 baseline 확보.
2) Coconut 전용: latent_generate 루프 구현 후, 커리큘럼+보조 로스 적용.
3) 모드 전환: 제어 토큰/옵션으로 요청별 ETD vs Coconut 선택.
4) 필요 시 소규모 혼합 스케줄(일부 스텝만 latent) 실험.
