# Block Diffusion 가이드

BD3-LM (Block Discrete Denoising Diffusion Language Model) + Fast-dLLM 패턴을 통합한 AR-Diffusion 하이브리드입니다.

## 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Block Diffusion Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Block 1          Block 2          Block 3          Block 4                 │
│  ┌────────┐       ┌────────┐       ┌────────┐       ┌────────┐             │
│  │Diffusion│  AR  │Diffusion│  AR  │Diffusion│  AR  │Diffusion│            │
│  │(병렬)   │ ───► │(병렬)   │ ───► │(병렬)   │ ───► │(병렬)   │            │
│  └────────┘  KV   └────────┘  KV   └────────┘  KV   └────────┘             │
│              Cache            Cache            Cache                        │
│                                                                              │
│  Block 내부: Bidirectional + Diffusion (병렬 생성)                          │
│  Block 간: Causal AR + KV Cache (효율적 연결)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 핵심 아이디어

1. **Block-wise Semi-AR Generation** (BD3-LM)
   - Block 단위로 AR 진행
   - Block 내부는 Diffusion으로 병렬 생성
   - KV Cache로 이전 블록 정보 재사용

2. **Replace Position KV Cache** (Fast-dLLM)
   - 전체 KV 재계산 대신 특정 위치만 업데이트
   - 효율적인 block-wise generation

3. **Block Diffusion Attention Mask** (BD3-LM)
   - M_BD: Block Diagonal (블록 내 양방향)
   - M_OBC: Offset Block Causal (이전 블록의 x0 참조)
   - M_BC: Block Causal (x0 간 causal)

## 성능

- **2.5x faster** than AR (LLaMA3, Qwen2.5) on GSM8K
- **50x faster** than vanilla dLLMs (LLaDA, Dream)
- Global coherence (bidirectional within block)

## 빠른 시작

### 1. 기본 사용법

```python
from olmo_core.nn.transformer import (
    BlockDiffusionConfig,
    NoiseScheduleType,
    wrap_model_with_block_diffusion,
)

# Block Diffusion 설정
config = BlockDiffusionConfig(
    block_size=64,
    num_diffusion_steps=8,
    noise_schedule=NoiseScheduleType.LOGLINEAR,  # BD3-LM default
    use_kv_cache=True,
    use_replace_position=True,  # Fast-dLLM pattern
)

# 모델 래핑
model = wrap_model_with_block_diffusion(base_model, config)

# 생성
output = model.generate(input_ids, max_new_tokens=256)
```

### 2. 간편 생성

```python
from olmo_core.nn.transformer import create_block_diffusion_model

model = create_block_diffusion_model(
    base_model,
    block_size=64,
    num_diffusion_steps=8,
)

output = model.generate(input_ids, max_new_tokens=256)
```

### 3. Training

```python
# Training forward pass
outputs = model(
    input_ids=input_ids,
    labels=labels,
)

loss = outputs["loss"]
loss.backward()
```

## 노이즈 스케줄

BD3-LM의 노이즈 스케줄을 그대로 구현했습니다:

### LogLinear (기본값, 권장)

```python
config = BlockDiffusionConfig(
    noise_schedule=NoiseScheduleType.LOGLINEAR,
)
```

- `move_chance = t`
- `loss_scaling = -1/t`
- BD3-LM의 기본 스케줄

### Cosine

```python
config = BlockDiffusionConfig(
    noise_schedule=NoiseScheduleType.COSINE,
)
```

- `move_chance = 1 - cos(t * π/2)`
- 부드러운 전환

### Square / Sqrt

```python
config = BlockDiffusionConfig(
    noise_schedule=NoiseScheduleType.SQUARE,  # exp=2
    # or
    noise_schedule=NoiseScheduleType.SQRT,    # exp=0.5
)
```

## Block Diffusion Attention Mask

BD3-LM의 specialized attention mask를 구현합니다:

```python
from olmo_core.nn.transformer import BlockDiffAttentionMask

# SDPA mask
mask_module = BlockDiffAttentionMask(
    seq_len=1024,
    block_size=64,
    attn_backend="sdpa",
)
attn_mask = mask_module.get_mask(device=device)

# FlexAttention mask function (for training with x0)
from functools import partial
flex_mask_fn = partial(
    BlockDiffAttentionMask.flex_mask_fn,
    block_size=64,
    n=512,  # length of xt
)
```

### Mask 구성 (Training with x0)

시퀀스가 `[xt, x0]` 형태일 때:

1. **M_BD (Block Diagonal)**: xt 내 같은 블록은 양방향 attention
2. **M_OBC (Offset Block Causal)**: xt가 x0의 이전 블록들 참조
3. **M_BC (Block Causal)**: x0 간 block-causal attention

## Replace Position KV Cache

Fast-dLLM의 효율적 KV cache 업데이트 패턴:

```python
from olmo_core.nn.transformer import ReplacePositionKVCache

cache = ReplacePositionKVCache(
    num_layers=32,
    batch_size=4,
    max_seq_len=2048,
    num_kv_heads=8,
    head_dim=128,
)

# 초기화
cache.initialize(model_outputs.past_key_values)

# 특정 위치만 업데이트
replace_position = torch.zeros(B, seq_len, dtype=torch.bool)
replace_position[:, block_start:block_end] = True

new_k, new_v = cache.update(
    layer_idx=0,
    new_k=new_keys,
    new_v=new_values,
    replace_position=replace_position,
)
```

## 설정 옵션

```python
@dataclass
class BlockDiffusionConfig:
    # Block 설정
    block_size: int = 64          # 블록당 토큰 수 (권장: 32-128)
    max_blocks: int = 32          # 최대 블록 수

    # Diffusion 설정
    num_diffusion_steps: int = 8  # 블록당 diffusion steps (권장: 4-16)
    noise_schedule: NoiseScheduleType = NoiseScheduleType.LOGLINEAR
    mask_token_id: int = -1       # -1이면 vocab_size 사용
    noise_eps: float = 1e-3

    # Sampling 설정
    sampling_eps: float = 1e-3
    nucleus_p: float = 1.0        # 1.0 = no nucleus sampling
    use_first_hitting: bool = False  # BD3-LM first-hitting sampler

    # KV Cache 설정
    use_kv_cache: bool = True
    use_replace_position: bool = True  # Fast-dLLM pattern

    # Attention 설정
    attn_backend: str = "sdpa"    # "sdpa", "flex", "flash"

    # Training
    antithetic_sampling: bool = True  # variance reduction
    training_resample: bool = True
    use_cross_attn: bool = False      # x0 conditioning (BD3-LM)
```

## Full Pipeline 통합

```python
from olmo_core.nn.transformer import create_full_hybrid_model

# Dr.Tulu + Hybrid + MoE + ETD + Latent Reasoning + Block Diffusion + MTP
model = create_full_hybrid_model(
    # 기본 설정
    use_moe=True,
    use_etd=True,
    use_latent_reasoning=True,
    use_block_diffusion=True,  # Block Diffusion 활성화
    use_mtp=True,

    # Block Diffusion 설정
    block_size=64,
    num_diffusion_steps=8,
    noise_schedule="loglinear",
)
```

## 참고 문헌

1. **BD3-LM**: arXiv:2503.09573 (ICLR 2025 Oral)
   - Block Discrete Denoising Diffusion Language Model
   - Semi-AR + Diffusion 하이브리드

2. **Fast-dLLM**: arXiv:2505.22618 (NVIDIA)
   - Block-wise Approximate KV Cache
   - Confidence-aware Parallel Decoding

3. **D2F**: arXiv:2508.09192 (Discrete Diffusion Forcing)
   - 2.5x faster than AR
   - 50x faster than vanilla dLLMs

4. **LLaDA**: arXiv:2502.09992
   - Bidirectional attention
   - Masking diffusion objective

## 관련 파일

- `src/olmo_core/nn/transformer/block_diffusion.py`: Block Diffusion 모듈
- `src/olmo_core/nn/transformer/cross_arch_transfer.py`: Full pipeline 통합
- `references/repos/bd3lms/`: BD3-LM 참조 구현
- `references/repos/Fast-dLLM/`: Fast-dLLM 참조 구현
- `references/repos/Discrete-Diffusion-Forcing/`: D2F 참조 구현
