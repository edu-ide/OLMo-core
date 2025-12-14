# Coconut-style latent reasoning: 재사용 가능한 데이터 포맷 가이드

이 문서는 기존 텍스트 데이터셋(질문/답변/CoT)을 활용해 Coconut(Chain of Continuous Thought) 방식으로 학습할 때 필요한 포맷과 커리큘럼 아이디어를 정리합니다. 핵심은 **명시적 텍스트 CoT → 점진적 latent CoT 전환**을 통해 안정적으로 학습·디버깅하는 것입니다.

## 전환 단계 개요
1) **명시적 CoT 단계**: 기존 데이터셋에 포함된 CoT(중간 추론 텍스트)를 그대로 사용. 표준 next-token 로스로 학습.
2) **하이브리드 단계**: 중간 일부 스텝을 latent 스텝으로 대체. 예) 3스텝 CoT 중 앞의 2스텝만 텍스트, 마지막 1스텝은 latent hidden 피드백으로.
3) **latent 우선 단계**: 대부분의 중간 스텝을 latent로 수행하고, 최종 답만 텍스트로 디코딩.

## 데이터 포맷 예시
### 1) 명시적 CoT (기존)
```
<user>문제</user>
<assistant>생각 1 …\n생각 2 …\n최종 답: …</assistant>
```

### 2) 하이브리드(부분 latent) 예시
- 앞부분 CoT 텍스트 → 모델 입력/감독
- 뒷부분 CoT → latent 스텝으로 돌리고, 필요 시 보조 디코더로 텍스트 복원
```
<user>문제</user>
<assistant>생각 1 …\n생각 2 …</assistant>
# 이후 스텝은 latent_generate 로 진행, 마지막에 답만 텍스트로 디코드
```

### 3) latent-only 중간, 답만 텍스트
- CoT 텍스트를 제거하고, 정답만 남김. 중간 추론은 latent 루프에서 수행.
```
<user>문제</user>
<assistant>최종 답: …</assistant>
```

## 안정화/디버깅 옵션
- **SIM-CoT**: 학습 시에만 보조 디코더를 붙여 latent hidden → 텍스트로 번역하도록 보조 로스 부여(잠재 추론 내용 확인 용도).
- **Latent-SFT**: latent 벡터를 임의로 두지 않고, 소프트 어휘 분포(soft embedding) 위에 투사해 학습 안정성 확보.
- **커리큘럼 스케줄**: epoch/step에 따라 텍스트 CoT 비중을 줄이고 latent 스텝 비중을 늘리는 스케줄을 명시.

## 기존 데이터셋 재활용 체크리스트
- CoT 필드가 있는 데이터: 바로 1→2→3 단계로 전환 가능.
- CoT가 없는 데이터: 정답만 있는 형식이면 3단계로 바로 사용 가능하지만, SIM-CoT/Latent-SFT 없이 latent-only를 돌리면 불안정할 수 있음.
- 멀티턴/챗 포맷: 시스템/유저/어시스턴트 역할 구분을 유지하고, latent 스텝은 모델 내부 루프로 처리하되 외부 텍스트에는 노출하지 않음.

## 요약
- 데이터 자체는 텍스트 기반 그대로 사용 가능하되, **중간 추론을 얼마나 텍스트로 남길지 vs. latent로 돌릴지**를 커리큘럼으로 조정.
- 보조 디코더나 soft-embedding 제약을 활용하면 latent-only 단계의 불안정성과 디버깅 어려움을 완화할 수 있습니다.
