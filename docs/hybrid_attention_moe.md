# Hybrid Attention & MoE presets (OLMo-core)

이 문서는 OLMo-3 7B용 하이브리드 어텐션/게이팅/MoE 프리셋을 추가한 이유와 기본 하이퍼파라미터를 정리합니다.  
핵심 목표는 **효율**(선형·희소 경로 활용)과 **정확도**(일부 소프트맥스 레이어 유지)를 동시에 확보하는 것입니다.

## 무엇이 달라졌나
- `TransformerConfig.olmo3_7B_hybrid_gated(...)`
  - `hybrid_period` 주기로 일부 레이어는 기존 소프트맥스 어텐션을 유지하고, 나머지는 `gated_output` 어텐션(헤드별 시그모이드 게이트)을 사용.
  - 게이트는 head-wise scalar 게이팅으로 오버헤드를 최소화하면서 attention sink 완화·안정성 향상을 노림.
- `TransformerConfig.olmo3_7B_hybrid_moe(...)`
  - `hybrid_period` 주기로 MoE FFN을 적용하고, 나머지는 dense FFN을 유지하는 하이브리드 MoE.
  - 기본값: `num_experts=64`, `top_k=8`, `capacity_factor=1.25`, `shared_expert=True`, `lb_loss_weight=0.01`, `z_loss_weight=0.001`.
  - MoE 블록은 `moe_reordered_norm` 타입으로, 로드밸런스/안정성에 초점.

## 왜 이렇게 설계했나
- **하이브리드 어텐션**: Qwen3-Next 등 최신 모델처럼 일부 레이어를 소프트맥스로 남겨 복잡한 회상/복사 작업을 보존하고, 나머지에 게이팅을 넣어 효율·안정성을 개선.
- **게이팅(헤드별 시그모이드)**: 파라미터 증가를 최소화하면서 출력 스케일 폭주를 억제하고 양자화 친화적 특성을 기대.
- **하이브리드 MoE**: 모든 레이어에 MoE를 넣는 대신 일정 주기만 활성화해 FLOPs/메모리를 제어하면서 희소 전문가의 이득을 취함. shared expert는 라우터 실패 시 성능 방어용 안전망 역할.

## 사용 예시
```python
from olmo_core.nn.transformer import TransformerConfig

# 하이브리드 게이팅 어텐션
cfg = TransformerConfig.olmo3_7B_hybrid_gated(
    vocab_size=tokenizer.vocab_size,
    hybrid_period=4,              # 3/4 레이어는 gated_output, 1/4은 softmax
)

# 하이브리드 MoE
cfg = TransformerConfig.olmo3_7B_hybrid_moe(
    vocab_size=tokenizer.vocab_size,
    hybrid_period=2,              # 절반 MoE, 절반 dense
    num_experts=64,
    top_k=8,
    shared_expert=True,
)
model = cfg.build(init_device="meta")
```

## 튜닝 가이드 (권장 시작점)
- `hybrid_period`: 작게(2~4) 잡을수록 소프트맥스/모든 토큰 처리 비율이 높아져 품질↑, 효율↓.
- `num_experts`/`top_k`: 64/8 → 32/4로 줄이면 속도/메모리↓, 품질 다소↓.
- `capacity_factor`: 1.0~1.25 범위에서 하드웨어/로드밸런스에 맞춰 조정.
- `shared_expert`: 안정성을 위해 기본적으로 `True` 유지 권장.

## 알림
- `gated_output`/`gated_deltanet` 어텐션은 실험 기능입니다. TP/CP 최적 커널이 없는 환경에서는 성능 이득이 제한될 수 있습니다.
- MoE는 라우터 로스와 capacity 설정에 민감하므로, 학습 초기에 모니터링(로드밸런스, 토큰 드랍률)이 필요합니다.
