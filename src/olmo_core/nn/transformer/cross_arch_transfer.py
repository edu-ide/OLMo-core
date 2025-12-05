"""
Cross-Architecture Transfer: Qwen3 → Hybrid (GatedDeltaNet + GatedAttention)

이 모듈은 표준 Transformer (Qwen3-8B)에서 Hybrid 아키텍처로
가중치를 전이합니다.

전이 전략:
1. 호환 레이어 직접 복사: Embedding, LayerNorm, FFN
2. Attention → GatedAttention: Q,K,V,O 복사 + Gate 초기화
3. Attention → GatedDeltaNet: Q,K,V,O 복사 + Gate 초기화 + State 초기화

References:
- TransMamba: arXiv:2408.09233 (Transformer → Mamba)
- CAB: Cross-Architecture Bridge
- Qwen3-Next: Gated DeltaNet + Gated Attention Hybrid
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import Config
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    GatedDeltaNetAttention,
    GatedOutputAttention,
)

log = logging.getLogger(__name__)

__all__ = [
    "CrossArchTransferConfig",
    "HybridAttentionConfig",
    "HybridTransformerBlock",
    "HybridTransformer",
    "transfer_qwen3_to_hybrid",
    "create_hybrid_from_qwen3",
]


@dataclass
class HybridAttentionConfig(Config):
    """
    Hybrid Attention 설정 (Qwen3-Next 스타일).

    일부 레이어는 GatedDeltaNet (O(n) 선형),
    나머지는 GatedAttention (O(n²) 표준)을 사용합니다.
    """

    # 레이어 패턴
    deltanet_layer_pattern: List[int] = field(default_factory=lambda: [0, 1, 2])
    """GatedDeltaNet을 사용할 레이어 패턴 (반복됨). 예: [0,1,2]면 4개마다 3개가 DeltaNet"""

    pattern_period: int = 4
    """패턴 주기. 예: 4면 [0,1,2,3] 중 deltanet_layer_pattern에 해당하는 것만 DeltaNet"""

    # Attention 공통 설정
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    bias: bool = False
    qk_norm: bool = True

    # Gate 설정
    gate_bias: bool = False
    gate_init_value: float = 0.0
    """Gate 초기값 (0이면 sigmoid(0)=0.5로 시작)"""

    def should_use_deltanet(self, layer_idx: int) -> bool:
        """해당 레이어가 DeltaNet을 사용해야 하는지"""
        position_in_period = layer_idx % self.pattern_period
        return position_in_period in self.deltanet_layer_pattern


@dataclass
class CrossArchTransferConfig(Config):
    """
    Cross-Architecture Transfer 설정.

    Qwen3-8B → Hybrid 변환에 필요한 설정입니다.
    """

    source_model: str = "Qwen/Qwen3-8B-Base"
    """소스 모델 (HuggingFace 경로)"""

    hybrid_attention: HybridAttentionConfig = field(default_factory=HybridAttentionConfig)
    """Hybrid attention 설정"""

    # 전이 설정
    copy_attention_weights: bool = True
    """Attention Q,K,V,O 가중치 복사"""

    copy_ffn_weights: bool = True
    """FFN 가중치 복사"""

    copy_norm_weights: bool = True
    """LayerNorm 가중치 복사"""

    copy_embedding_weights: bool = True
    """Embedding 가중치 복사"""

    # Gate 초기화
    gate_init_strategy: str = "zero"
    """Gate 초기화: 'zero' (0.5 시작), 'small' (약간 열림), 'ones' (완전 열림)"""

    # Knowledge Distillation 설정
    use_distillation: bool = True
    """가중치 전이 후 Knowledge Distillation 사용"""

    distillation_temperature: float = 2.0
    """KD 온도"""

    distillation_alpha: float = 0.5
    """KD loss 가중치 (vs task loss)"""


class HybridTransformerBlock(nn.Module):
    """
    Hybrid Transformer Block (GatedDeltaNet 또는 GatedAttention).

    Qwen3-Next 스타일의 블록으로, 레이어 인덱스에 따라
    다른 어텐션 타입을 사용합니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        layer_idx: int,
        config: HybridAttentionConfig,
        ffn_hidden_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_deltanet = config.should_use_deltanet(layer_idx)

        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads or n_heads
        head_dim = config.head_dim or (d_model // n_heads)
        ffn_hidden = ffn_hidden_size or (d_model * 4)

        # Pre-norm
        self.input_norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)
        self.post_attn_norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)

        # Attention (Hybrid: DeltaNet or Gated)
        if self.use_deltanet:
            # GatedDeltaNet은 GQA를 지원하지 않음 → n_kv_heads = n_heads (MHA) 사용
            self.attention = GatedDeltaNetAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,  # DeltaNet은 MHA만 지원
                bias=config.bias,
                gate_bias=config.gate_bias,
                dtype=dtype,
                init_device=init_device,
            )
        else:
            self.attention = GatedOutputAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                bias=config.bias,
                gate_bias=config.gate_bias,
                dtype=dtype,
                init_device=init_device,
            )

        # FFN (SwiGLU style)
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            hidden_size=ffn_hidden,
            bias=config.bias,
            dtype=dtype,
            init_device=init_device,
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Pre-norm + Attention + Residual
        h = self.input_norm(x)
        h = self.attention(h, **kwargs)
        x = x + h

        # Pre-norm + FFN + Residual
        h = self.post_attn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (Qwen/Llama 스타일)"""

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.up_proj = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridTransformer(nn.Module):
    """
    Hybrid Transformer (Qwen3-Next 스타일).

    GatedDeltaNet + GatedAttention을 조합한 모델입니다.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        config: HybridAttentionConfig,
        ffn_hidden_size: Optional[int] = None,
        max_seq_len: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(
            vocab_size, d_model, dtype=dtype, device=init_device
        )

        # Transformer Blocks
        self.layers = nn.ModuleList([
            HybridTransformerBlock(
                d_model=d_model,
                layer_idx=i,
                config=config,
                ffn_hidden_size=ffn_hidden_size,
                dtype=dtype,
                init_device=init_device,
            )
            for i in range(n_layers)
        ])

        # Final norm
        self.norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)

        # LM Head (weight tying with embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype, device=init_device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Embedding
        h = self.embed_tokens(input_ids)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, **kwargs)

        # Final norm
        h = self.norm(h)

        # LM Head
        logits = self.lm_head(h)

        outputs = {"logits": logits, "hidden_states": h}

        # Loss 계산
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss

        return outputs

    def get_layer_stats(self) -> Dict[str, int]:
        """레이어 타입 통계"""
        deltanet_count = sum(1 for l in self.layers if l.use_deltanet)
        gated_count = self.n_layers - deltanet_count
        return {
            "deltanet_layers": deltanet_count,
            "gated_attention_layers": gated_count,
            "total_layers": self.n_layers,
        }


# ==============================================================================
# Weight Transfer Functions
# ==============================================================================

def transfer_qwen3_to_hybrid(
    source_model: nn.Module,
    target_model: HybridTransformer,
    config: CrossArchTransferConfig,
) -> HybridTransformer:
    """
    Qwen3 가중치를 Hybrid 모델로 전이.

    Args:
        source_model: Qwen3 소스 모델 (HuggingFace)
        target_model: Hybrid 타겟 모델
        config: 전이 설정

    Returns:
        가중치가 전이된 Hybrid 모델
    """
    log.info("Starting cross-architecture weight transfer...")

    # 소스 모델 구조 파악
    source_state = source_model.state_dict()

    # 1. Embedding 전이
    if config.copy_embedding_weights:
        _transfer_embeddings(source_model, target_model, source_state)

    # 2. Layer별 전이
    n_layers = target_model.n_layers
    for layer_idx in range(n_layers):
        _transfer_layer(
            source_model, target_model,
            layer_idx, config, source_state
        )

    # 3. Final Norm 전이
    if config.copy_norm_weights:
        _transfer_final_norm(source_model, target_model, source_state)

    # 4. LM Head 전이
    _transfer_lm_head(source_model, target_model, source_state)

    # 5. Gate 초기화
    _initialize_gates(target_model, config)

    log.info("Cross-architecture transfer complete!")

    return target_model


def _transfer_embeddings(
    source: nn.Module,
    target: HybridTransformer,
    state_dict: Dict[str, torch.Tensor],
):
    """Embedding 가중치 전이"""
    log.info("Transferring embeddings...")

    # Qwen3 스타일 키 탐색
    embed_keys = [k for k in state_dict.keys() if "embed" in k.lower()]

    for key in embed_keys:
        if "weight" in key and state_dict[key].shape == target.embed_tokens.weight.shape:
            target.embed_tokens.weight.data.copy_(state_dict[key])
            log.info(f"  Copied {key} → embed_tokens.weight")
            break


def _transfer_layer(
    source: nn.Module,
    target: HybridTransformer,
    layer_idx: int,
    config: CrossArchTransferConfig,
    state_dict: Dict[str, torch.Tensor],
):
    """단일 레이어 가중치 전이"""
    target_layer = target.layers[layer_idx]
    use_deltanet = target_layer.use_deltanet

    log.info(f"Transferring layer {layer_idx} ({'DeltaNet' if use_deltanet else 'GatedAttn'})...")

    # Qwen3 레이어 키 패턴
    # model.layers.{idx}.self_attn.{q,k,v,o}_proj.weight
    # model.layers.{idx}.mlp.{gate,up,down}_proj.weight
    # model.layers.{idx}.input_layernorm.weight
    # model.layers.{idx}.post_attention_layernorm.weight

    prefix = f"model.layers.{layer_idx}"

    # Attention 전이
    if config.copy_attention_weights:
        attn = target_layer.attention

        # Q, K, V, O projections
        _copy_if_exists(state_dict, f"{prefix}.self_attn.q_proj.weight", attn.w_q.weight)
        _copy_if_exists(state_dict, f"{prefix}.self_attn.k_proj.weight", attn.w_k.weight)
        _copy_if_exists(state_dict, f"{prefix}.self_attn.v_proj.weight", attn.w_v.weight)
        _copy_if_exists(state_dict, f"{prefix}.self_attn.o_proj.weight", attn.w_out.weight)

        # Bias (있으면)
        _copy_if_exists(state_dict, f"{prefix}.self_attn.q_proj.bias", getattr(attn.w_q, 'bias', None))
        _copy_if_exists(state_dict, f"{prefix}.self_attn.k_proj.bias", getattr(attn.w_k, 'bias', None))
        _copy_if_exists(state_dict, f"{prefix}.self_attn.v_proj.bias", getattr(attn.w_v, 'bias', None))
        _copy_if_exists(state_dict, f"{prefix}.self_attn.o_proj.bias", getattr(attn.w_out, 'bias', None))

    # FFN 전이
    if config.copy_ffn_weights:
        ffn = target_layer.ffn

        _copy_if_exists(state_dict, f"{prefix}.mlp.gate_proj.weight", ffn.gate_proj.weight)
        _copy_if_exists(state_dict, f"{prefix}.mlp.up_proj.weight", ffn.up_proj.weight)
        _copy_if_exists(state_dict, f"{prefix}.mlp.down_proj.weight", ffn.down_proj.weight)

    # Norm 전이
    if config.copy_norm_weights:
        _copy_if_exists(state_dict, f"{prefix}.input_layernorm.weight", target_layer.input_norm.weight)
        _copy_if_exists(state_dict, f"{prefix}.post_attention_layernorm.weight", target_layer.post_attn_norm.weight)


def _transfer_final_norm(
    source: nn.Module,
    target: HybridTransformer,
    state_dict: Dict[str, torch.Tensor],
):
    """Final LayerNorm 전이"""
    _copy_if_exists(state_dict, "model.norm.weight", target.norm.weight)


def _transfer_lm_head(
    source: nn.Module,
    target: HybridTransformer,
    state_dict: Dict[str, torch.Tensor],
):
    """LM Head 전이"""
    _copy_if_exists(state_dict, "lm_head.weight", target.lm_head.weight)


def _copy_if_exists(
    state_dict: Dict[str, torch.Tensor],
    key: str,
    target_param: Optional[torch.Tensor],
):
    """키가 있으면 복사"""
    if target_param is None:
        return

    if key in state_dict:
        source_tensor = state_dict[key]
        if source_tensor.shape == target_param.shape:
            target_param.data.copy_(source_tensor)
        else:
            log.warning(f"Shape mismatch for {key}: {source_tensor.shape} vs {target_param.shape}")
    else:
        # 다른 키 패턴 시도
        alt_keys = [
            key.replace("model.", ""),
            key.replace("self_attn", "attention"),
            key.replace("mlp", "feed_forward"),
        ]
        for alt_key in alt_keys:
            if alt_key in state_dict:
                source_tensor = state_dict[alt_key]
                if source_tensor.shape == target_param.shape:
                    target_param.data.copy_(source_tensor)
                    return


def _initialize_gates(
    model: HybridTransformer,
    config: CrossArchTransferConfig,
):
    """Gate 파라미터 초기화"""
    log.info(f"Initializing gates with strategy: {config.gate_init_strategy}")

    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, 'w_gate'):
            with torch.no_grad():
                if config.gate_init_strategy == "zero":
                    # sigmoid(0) = 0.5 → 절반 열림
                    nn.init.zeros_(attn.w_gate.weight)
                    if attn.w_gate.bias is not None:
                        nn.init.zeros_(attn.w_gate.bias)

                elif config.gate_init_strategy == "small":
                    # 약간 양수 → 절반보다 조금 더 열림
                    nn.init.normal_(attn.w_gate.weight, mean=0.0, std=0.01)
                    if attn.w_gate.bias is not None:
                        nn.init.constant_(attn.w_gate.bias, 0.5)

                elif config.gate_init_strategy == "ones":
                    # 크게 양수 → 거의 완전 열림 (원본과 유사)
                    nn.init.normal_(attn.w_gate.weight, mean=0.0, std=0.01)
                    if attn.w_gate.bias is not None:
                        nn.init.constant_(attn.w_gate.bias, 2.0)


# ==============================================================================
# High-Level API
# ==============================================================================

def create_hybrid_from_qwen3(
    source_model_name: str = "Qwen/Qwen3-8B-Base",
    config: Optional[CrossArchTransferConfig] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HybridTransformer:
    """
    Qwen3 모델에서 Hybrid 모델 생성.

    Args:
        source_model_name: HuggingFace 모델 이름
        config: 전이 설정 (None이면 기본값)
        device: 디바이스
        dtype: 데이터 타입

    Returns:
        가중치가 전이된 Hybrid 모델

    Example:
        ```python
        # Qwen3-8B → Hybrid (GatedDeltaNet + GatedAttention)
        hybrid_model = create_hybrid_from_qwen3(
            "Qwen/Qwen3-8B-Base",
            device="cuda",
        )

        # 레이어 통계 확인
        print(hybrid_model.get_layer_stats())
        # {'deltanet_layers': 24, 'gated_attention_layers': 8, 'total_layers': 32}
        ```
    """
    if config is None:
        config = CrossArchTransferConfig(source_model=source_model_name)

    log.info(f"Loading source model: {source_model_name}")

    # HuggingFace 모델 로드
    try:
        from transformers import AutoModelForCausalLM, AutoConfig

        hf_config = AutoConfig.from_pretrained(source_model_name, trust_remote_code=True)
        source_model = AutoModelForCausalLM.from_pretrained(
            source_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="cpu",  # CPU에서 로드 후 전이
        )
    except ImportError:
        raise ImportError("transformers 라이브러리가 필요합니다.")

    # 소스 모델 정보 추출
    vocab_size = hf_config.vocab_size
    d_model = hf_config.hidden_size
    n_layers = hf_config.num_hidden_layers
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, 'num_key_value_heads', n_heads)
    ffn_hidden = hf_config.intermediate_size

    log.info(f"Source model: vocab={vocab_size}, d_model={d_model}, layers={n_layers}, heads={n_heads}")

    # Hybrid Attention 설정 업데이트
    config.hybrid_attention.n_heads = n_heads
    config.hybrid_attention.n_kv_heads = n_kv_heads

    # Hybrid 모델 생성
    log.info("Creating hybrid model...")
    hybrid_model = HybridTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        config=config.hybrid_attention,
        ffn_hidden_size=ffn_hidden,
        dtype=dtype,
        init_device="cpu",
    )

    # 가중치 전이
    hybrid_model = transfer_qwen3_to_hybrid(source_model, hybrid_model, config)

    # 디바이스로 이동
    hybrid_model = hybrid_model.to(device)

    # 소스 모델 메모리 해제
    del source_model
    torch.cuda.empty_cache()

    log.info(f"Hybrid model created: {hybrid_model.get_layer_stats()}")

    return hybrid_model


# ==============================================================================
# Knowledge Distillation (Optional)
# ==============================================================================

class DistillationTrainer:
    """
    Knowledge Distillation 트레이너.

    Teacher (Qwen3) → Student (Hybrid)로 지식 증류.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: HybridTransformer,
        config: CrossArchTransferConfig,
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.temperature = config.distillation_temperature
        self.alpha = config.distillation_alpha

        # Teacher는 학습 안함
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Distillation loss 계산.

        L = α * L_kd + (1-α) * L_task
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            if hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits
            else:
                teacher_logits = teacher_outputs

        # Student forward
        student_outputs = self.student(input_ids, labels=labels)
        student_logits = student_outputs["logits"]
        task_loss = student_outputs.get("loss", torch.tensor(0.0))

        # KL Divergence loss (soft targets)
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss

        return {
            "loss": total_loss,
            "kd_loss": kd_loss,
            "task_loss": task_loss,
        }


# ==============================================================================
# Integration with ETD + MTP
# ==============================================================================

def create_full_hybrid_model(
    source_model_name: str = "rl-research/DR-Tulu-8B",
    use_moe: bool = True,
    use_etd: bool = True,
    use_mtp: bool = True,
    use_latent_reasoning: bool = True,
    use_block_diffusion: bool = False,
    latent_mode: str = "codi",
    num_experts: int = 8,
    num_experts_per_token: int = 2,
    num_latent_thoughts: int = 4,
    block_size: int = 64,
    num_diffusion_steps: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Qwen3/Dr.Tulu → Hybrid + MoE + ETD + Latent Reasoning + Block Diffusion + MTP.

    기본값은 Dr.Tulu-8B (Qwen3 + SFT + RLER)로,
    이미 추론 능력이 강화된 모델에서 시작합니다.

    Args:
        source_model_name: 소스 모델 이름
        use_moe: MoE 업사이클링 적용 여부
        use_etd: ETD 적용 여부
        use_mtp: MTP 적용 여부
        use_latent_reasoning: Latent Reasoning (CODI/PCCoT) 적용 여부
        use_block_diffusion: Block Diffusion (AR-Diffusion hybrid) 적용 여부
        latent_mode: "codi", "pccot", "kava", "softcot" 중 선택
        num_experts: MoE 전문가 수
        num_experts_per_token: 토큰당 활성화 전문가 수
        num_latent_thoughts: Continuous thought 토큰 수
        block_size: Block Diffusion block size
        num_diffusion_steps: Diffusion steps per block
        device: 디바이스
        dtype: 데이터 타입

    Returns:
        완전 통합된 모델

    Example:
        ```python
        # Dr.Tulu-8B 기반 (기본값, 추천)
        model = create_full_hybrid_model(
            use_moe=True,
            use_etd=True,
            use_latent_reasoning=True,  # CODI 기반 latent reasoning
            use_mtp=True,
        )

        # PCCoT 병렬 latent reasoning
        model = create_full_hybrid_model(
            latent_mode="pccot",
            use_latent_reasoning=True,
        )

        # 파라미터 확인
        print(model.get_param_count())
        # {'total_B': 37.0, 'active_B': 13.0}
        ```

    Recommended source models:
        - rl-research/DR-Tulu-8B: 추천! (Qwen3 + SFT + RLER, 추론 강화)
        - rl-research/DR-Tulu-SFT-8B: SFT만 적용된 버전
        - Qwen/Qwen3-8B-Base: 기본 Qwen3

    Latent Reasoning modes:
        - codi: Self-distillation (EMNLP 2025), CoCoNut 대비 20%+ 정확도
        - pccot: Jacobi iteration으로 병렬화 (EMNLP 2025)
        - kava: KV-cache distillation (October 2025)
        - softcot: Frozen LLM + projection (ACL 2025)
    """
    # 1. Hybrid 모델 생성
    log.info("Step 1: Creating hybrid model from Qwen3...")
    hybrid_model = create_hybrid_from_qwen3(
        source_model_name,
        device="cpu",  # CPU에서 생성 후 MoE 변환
        dtype=dtype,
    )

    # 2. MoE 업사이클링
    if use_moe:
        log.info(f"Step 2: Upcycling to MoE ({num_experts} experts, {num_experts_per_token} active)...")
        from .dna_transfer import UpcyclingConfig, upcycle_dense_to_moe

        moe_config = UpcyclingConfig(
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_init_noise=0.01,
            auxiliary_loss_weight=0.01,
        )

        hybrid_model = upcycle_dense_to_moe(hybrid_model, moe_config)

    # 3. ETD 적용
    if use_etd:
        log.info("Step 3: Applying ETD wrapper...")
        from .etd import ETDConfig, wrap_transformer_with_etd

        etd_config = ETDConfig(
            n_encoder_layers=hybrid_model.n_layers // 4,
            n_think_layers=hybrid_model.n_layers // 2,
            n_decoder_layers=hybrid_model.n_layers // 4,
            max_think_iterations=3,
            adaptive_depth=True,
            confidence_threshold=0.85,
            use_layer_router=True,
            use_lora_experts=True,
            num_lora_experts=4,
        )

        # ETD는 blocks를 기대하므로 layers를 blocks로 래핑
        hybrid_model.blocks = hybrid_model.layers
        hybrid_model = wrap_transformer_with_etd(hybrid_model, etd_config)

    # 4. Latent Reasoning 적용
    if use_latent_reasoning:
        log.info(f"Step 4: Applying Latent Reasoning ({latent_mode})...")
        from .latent_reasoning import (
            LatentReasoningConfig,
            LatentReasoningMode,
            wrap_model_with_latent_reasoning,
        )

        mode_map = {
            "coconut": LatentReasoningMode.COCONUT,
            "codi": LatentReasoningMode.CODI,
            "pccot": LatentReasoningMode.PCCOT,
            "kava": LatentReasoningMode.KAVA,
            "softcot": LatentReasoningMode.SOFTCOT,
        }

        latent_config = LatentReasoningConfig(
            mode=mode_map.get(latent_mode.lower(), LatentReasoningMode.CODI),
            num_latent_thoughts=num_latent_thoughts,
            compression_ratio=3.0,  # CODI default
            jacobi_iterations=3,    # PCCoT default
            parallel_thoughts=True,
            track_superposition=False,
        )

        hybrid_model = wrap_model_with_latent_reasoning(hybrid_model, latent_config)

    # 5. MTP 적용
    if use_mtp:
        log.info("Step 5: Applying MTP wrapper...")
        from .mtp import MTPConfig, wrap_model_with_mtp

        mtp_config = MTPConfig(
            num_predict_tokens=4,
            mtp_head_type="transformer",
            share_weights_across_steps=True,
            loss_weight_decay=0.5,
            mtp_loss_weight=0.3,
            speculation_lookahead=3,
        )

        hybrid_model = wrap_model_with_mtp(
            hybrid_model,
            mtp_config,
            vocab_size=hybrid_model.vocab_size,
        )

    # 6. Block Diffusion 적용 (AR-Diffusion Hybrid)
    if use_block_diffusion:
        log.info(f"Step 6: Applying Block Diffusion (block_size={block_size}, steps={num_diffusion_steps})...")
        from .block_diffusion import (
            BlockDiffusionConfig,
            wrap_model_with_block_diffusion,
        )

        bd_config = BlockDiffusionConfig(
            block_size=block_size,
            num_diffusion_steps=num_diffusion_steps,
            confidence_threshold=0.9,
            use_confidence_aware=True,
            use_kv_cache=True,
        )

        hybrid_model = wrap_model_with_block_diffusion(
            hybrid_model,
            bd_config,
            vocab_size=hybrid_model.vocab_size,
        )

    # 디바이스로 이동
    hybrid_model = hybrid_model.to(device)

    # 파라미터 카운트 함수 추가
    def get_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())

        # 활성 파라미터 계산 (MoE 고려)
        if use_moe:
            # FFN은 전문가 중 일부만 활성화
            # 대략 (attention + embed) + (ffn / num_experts * num_active)
            non_ffn_params = total_params * 0.4  # attention, embed 등
            ffn_params = total_params * 0.6      # FFN (MoE)
            active_ffn = ffn_params / num_experts * num_experts_per_token
            active_params = non_ffn_params + active_ffn
        else:
            active_params = total_params

        return {
            "total_params": total_params,
            "total_B": total_params / 1e9,
            "active_params": int(active_params),
            "active_B": active_params / 1e9,
        }

    hybrid_model.get_param_count = lambda: get_param_count(hybrid_model)

    # 최종 로그
    param_info = hybrid_model.get_param_count()
    log.info("=" * 60)
    log.info("Full hybrid model creation complete!")
    log.info(f"  - Hybrid attention: ✓ (GatedDeltaNet + GatedAttention)")
    log.info(f"  - MoE: {'✓ ' + str(num_experts) + ' experts' if use_moe else '✗'}")
    log.info(f"  - ETD: {'✓' if use_etd else '✗'}")
    log.info(f"  - Latent Reasoning: {'✓ ' + latent_mode.upper() if use_latent_reasoning else '✗'}")
    log.info(f"  - Block Diffusion: {'✓ block=' + str(block_size) if use_block_diffusion else '✗'}")
    log.info(f"  - MTP: {'✓' if use_mtp else '✗'}")
    log.info(f"  - Total params: {param_info['total_B']:.1f}B")
    log.info(f"  - Active params: {param_info['active_B']:.1f}B")
    log.info("=" * 60)

    return hybrid_model


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_dr_tulu_hybrid(
    use_moe: bool = True,
    use_etd: bool = True,
    use_mtp: bool = True,
    use_latent_reasoning: bool = True,
    use_block_diffusion: bool = False,
    latent_mode: str = "codi",
    num_experts: int = 8,
    device: str = "cuda",
) -> nn.Module:
    """
    Dr.Tulu-8B 기반 Hybrid 모델 생성 (추천).

    Dr.Tulu는 Qwen3-8B에 SFT + RLER이 적용된 모델로,
    추론 능력이 강화되어 있습니다.

    Example:
        ```python
        # AR 모드 (기본)
        model = create_dr_tulu_hybrid()

        # AR-Diffusion 하이브리드 모드
        model = create_dr_tulu_hybrid(use_block_diffusion=True)

        print(model.get_param_count())
        # {'total_B': 37.0, 'active_B': 13.0}
        ```
    """
    return create_full_hybrid_model(
        source_model_name="rl-research/DR-Tulu-8B",
        use_moe=use_moe,
        use_etd=use_etd,
        use_mtp=use_mtp,
        use_latent_reasoning=use_latent_reasoning,
        use_block_diffusion=use_block_diffusion,
        latent_mode=latent_mode,
        num_experts=num_experts,
        device=device,
    )


def create_qwen3_hybrid(
    use_moe: bool = True,
    use_etd: bool = True,
    use_mtp: bool = True,
    use_latent_reasoning: bool = True,
    use_block_diffusion: bool = False,
    latent_mode: str = "codi",
    num_experts: int = 8,
    device: str = "cuda",
) -> nn.Module:
    """
    Qwen3-8B-Base 기반 Hybrid 모델 생성.

    기본 Qwen3 모델에서 시작합니다.
    추가 학습이 더 필요할 수 있습니다.
    """
    return create_full_hybrid_model(
        source_model_name="Qwen/Qwen3-8B-Base",
        use_moe=use_moe,
        use_etd=use_etd,
        use_mtp=use_mtp,
        use_latent_reasoning=use_latent_reasoning,
        use_block_diffusion=use_block_diffusion,
        latent_mode=latent_mode,
        num_experts=num_experts,
        device=device,
    )
