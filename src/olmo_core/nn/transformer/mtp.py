"""
MTP (Multi-Token Prediction) 모듈

이 모듈은 LLM의 학습과 추론을 가속화하는 Multi-Token Prediction을 구현합니다.
- 학습 시: n개의 미래 토큰을 동시에 예측하여 더 나은 표현 학습
- 추론 시: Self-speculative decoding으로 2-3x 속도 향상

References:
- Meta MTP: arXiv:2404.19737 (Better & Faster Large Language Models via Multi-token Prediction)
- FastMTP: arXiv:2509.18362 (DeepSeek-V3 style single MTP head)
- Efficient MTP: arXiv:2502.09419 (On multi-token prediction for efficient LLM inference)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from olmo_core.config import Config

log = logging.getLogger(__name__)

__all__ = [
    "MTPConfig",
    "MTPHead",
    "MTPLoss",
    "MTPSpeculativeDecoder",
    "MedusaResBlock",
    "MedusaMTPHead",
    "wrap_model_with_mtp",
]


@dataclass
class MTPConfig(Config):
    """
    MTP (Multi-Token Prediction) 설정.

    Meta의 MTP 논문과 FastMTP를 결합한 구현입니다:
    - 학습 시: n개의 미래 토큰을 예측하는 보조 헤드 사용
    - 추론 시: Self-speculative decoding으로 병렬 검증
    """

    num_predict_tokens: int = 4
    """예측할 미래 토큰 수 (k=1,2,...,n). 논문에서 4가 최적"""

    mtp_head_type: str = "medusa"
    """MTP 헤드 타입: 'medusa' (권장, Medusa ResBlock), 'transformer', 'mlp', 'shared_lm_head'"""

    share_weights_across_steps: bool = True
    """FastMTP 스타일: 모든 예측 스텝에서 동일한 헤드 재사용"""

    head_hidden_size: int = 0
    """MTP 헤드의 hidden size (0이면 d_model과 동일)"""

    head_num_layers: int = 1
    """MTP 헤드의 transformer 레이어 수"""

    head_num_heads: int = 8
    """MTP 헤드의 어텐션 헤드 수 (transformer 타입 시)"""

    # Loss 설정
    loss_weight_decay: float = 0.5
    """지수 감쇠 계수 (β). α_k = β^(k-1) / Σβ^(j-1)"""

    main_loss_weight: float = 1.0
    """메인 (next-token) 예측 loss 가중치"""

    mtp_loss_weight: float = 0.5
    """MTP 보조 loss 총 가중치"""

    # Vocabulary compression (FastMTP)
    use_vocab_compression: bool = False
    """언어별 고빈도 토큰으로 제한"""

    compressed_vocab_size: int = 8192
    """압축된 어휘 크기"""

    # Speculative decoding 설정
    speculation_lookahead: int = 3
    """추론 시 한 번에 생성할 드래프트 토큰 수"""

    speculation_temperature: float = 0.0
    """드래프트 생성 온도 (0=greedy)"""

    acceptance_threshold: float = 0.9
    """드래프트 토큰 수락 임계값 (nucleus)"""

    # [OPTIMIZED] Gradient Checkpointing for MTP
    use_gradient_checkpointing: bool = True
    """MTP 헤드에 gradient checkpointing 적용 (메모리 절약)"""

    def __post_init__(self):
        if self.head_hidden_size == 0:
            log.info("MTP head_hidden_size가 0이므로 d_model과 동일하게 설정됩니다.")


@dataclass
class MTPTrainingConfig(Config):
    """MTP 학습 관련 설정"""

    warmup_steps: int = 1000
    """MTP loss 웜업 스텝 (초기에는 메인 loss만 사용)"""

    gradient_checkpointing: bool = True
    """MTP 헤드에 gradient checkpointing 적용 (메모리 절약)"""

    sequential_forward: bool = True
    """메모리 효율적 순차 forward/backward (O(V+d) vs O(nV+d))"""


class MTPProjection(nn.Module):
    """
    MTP 입력 프로젝션: 이전 hidden state + 임베딩을 결합.

    h^k_i = W_h * h^{k-1}_i + W_e * embed(t_{i+k})

    FastMTP 스타일의 효율적인 구현입니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        embed_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim or d_model

        # Hidden state projection
        self.hidden_proj = nn.Linear(
            d_model, d_model, bias=False, dtype=dtype, device=init_device
        )
        # Embedding projection (임베딩 차원이 다를 경우)
        if self.embed_dim != d_model:
            self.embed_proj = nn.Linear(
                self.embed_dim, d_model, bias=False, dtype=dtype, device=init_device
            )
        else:
            self.embed_proj = None

        # LayerNorm for stability
        self.norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 이전 스텝의 hidden states (B, T, D)
            embeddings: 다음 토큰의 임베딩 (B, T, E)

        Returns:
            결합된 hidden states (B, T, D)
        """
        h = self.hidden_proj(hidden_states)

        if self.embed_proj is not None:
            e = self.embed_proj(embeddings)
        else:
            e = embeddings

        return self.norm(h + e)


class MTPTransformerHead(nn.Module):
    """
    Transformer 스타일 MTP 헤드.

    DeepSeek-V3 및 Meta 논문에서 사용하는 방식입니다.
    단일 transformer 레이어를 사용하여 미래 토큰을 예측합니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                MTPTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    dtype=dtype,
                    init_device=init_device,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input hidden states (B, T, D)
            attention_mask: Causal mask (optional)

        Returns:
            Output hidden states (B, T, D)
        """
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class MTPTransformerLayer(nn.Module):
    """단일 MTP Transformer 레이어 (Self-Attention + FFN)"""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Self-attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=init_device)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False, dtype=dtype, device=init_device),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model, bias=False, dtype=dtype, device=init_device),
        )

        # Norms
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)
        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.attn_norm(x)

        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.o_proj(attn_output)
        x = residual + self.dropout(attn_output)

        # Pre-norm FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class MedusaResBlock(nn.Module):
    """
    Medusa-style Residual Block for MTP.

    Based on Medusa reference (FasterDecoding/Medusa):
    ```python
    class ResBlock(nn.Module):
        def __init__(self, hidden_size):
            self.linear = nn.Linear(hidden_size, hidden_size)
            torch.nn.init.zeros_(self.linear.weight)  # Zero init!
            self.act = nn.SiLU()

        def forward(self, x):
            return x + self.act(self.linear(x))
    ```

    Key insight:
    - Zero initialization → starts as identity mapping
    - Residual connection for stable training
    - SiLU activation (consistent with LLaMA)
    """

    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.linear = nn.Linear(
            hidden_size, hidden_size,
            dtype=dtype, device=init_device
        )
        # Zero initialization for identity mapping (from Medusa reference)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward: x + act(linear(x))"""
        return x + self.act(self.linear(x))


class MedusaMTPHead(nn.Module):
    """
    Medusa-style MTP Head.

    Based on Medusa reference (FasterDecoding/Medusa):
    ```python
    self.medusa_head = nn.ModuleList([
        nn.Sequential(
            *([ResBlock(hidden_size)] * num_layers),
            nn.Linear(hidden_size, vocab_size, bias=False),
        )
        for _ in range(num_heads)
    ])
    ```

    Each head predicts a different future token position.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Medusa-style: Multiple heads, each with ResBlocks + Linear
        self.medusa_head = nn.ModuleList([
            nn.Sequential(
                *[MedusaResBlock(d_model, dtype=dtype, init_device=init_device)
                  for _ in range(num_layers)],
                nn.Linear(d_model, vocab_size, bias=False, dtype=dtype, device=init_device),
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_orig: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Medusa MTP heads.

        Args:
            hidden_states: [B, T, D] from base model
            output_orig: Whether to also compute original LM head output

        Returns:
            logits: [num_heads, B, T, V] predictions for each head
        """
        medusa_logits = []
        for head in self.medusa_head:
            medusa_logits.append(head(hidden_states))

        # Stack: [num_heads, B, T, V]
        return torch.stack(medusa_logits, dim=0)


class MTPMLPHead(nn.Module):
    """
    간단한 MLP 스타일 MTP 헤드.

    FastMTP에서 효율성을 위해 제안된 방식입니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int = 0,
        num_layers: int = 2,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        hidden_size = hidden_size or d_model * 4

        layers = []
        in_dim = d_model
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_size, bias=False, dtype=dtype, device=init_device),
                nn.SiLU(),
            ])
            in_dim = hidden_size
        layers.append(
            nn.Linear(in_dim, d_model, bias=False, dtype=dtype, device=init_device)
        )

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.RMSNorm(d_model, eps=1e-6, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.norm(self.mlp(x))


class MTPHead(nn.Module):
    """
    MTP 예측 헤드 (통합).

    공유 가중치를 사용하여 n개의 미래 토큰을 순차적으로 예측합니다.

    수식:
    - k=1: M(h_i, embed(t_{i+1})) → logits for t_{i+2}
    - k>1: M(h^{k-1}, embed(t_{i+k})) → logits for t_{i+k+1}

    여기서 M은 MTP 헤드, h^k는 k번째 예측 스텝의 hidden state입니다.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        config: MTPConfig,
        embed_weight: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.config = config
        self.num_predict_tokens = config.num_predict_tokens

        # 입력 프로젝션 (hidden + embedding 결합)
        self.projection = MTPProjection(
            d_model=d_model,
            dtype=dtype,
            init_device=init_device,
        )

        # MTP 헤드 선택
        head_hidden_size = config.head_hidden_size or d_model
        if config.mtp_head_type == "medusa":
            # Medusa-style: Stack of ResBlocks (zero-init for identity start)
            self.mtp_head = nn.Sequential(
                *[MedusaResBlock(d_model, dtype=dtype, init_device=init_device)
                  for _ in range(config.head_num_layers)]
            )
        elif config.mtp_head_type == "transformer":
            self.mtp_head = MTPTransformerHead(
                d_model=d_model,
                num_heads=config.head_num_heads,
                num_layers=config.head_num_layers,
                dtype=dtype,
                init_device=init_device,
            )
        elif config.mtp_head_type == "mlp":
            self.mtp_head = MTPMLPHead(
                d_model=d_model,
                hidden_size=head_hidden_size,
                dtype=dtype,
                init_device=init_device,
            )
        else:
            # shared_lm_head: 헤드 없이 LM head만 사용
            self.mtp_head = nn.Identity()

        # Output projection (LM head와 가중치 공유 가능)
        if embed_weight is not None and config.mtp_head_type == "shared_lm_head":
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = embed_weight  # Tied weights
        else:
            self.lm_head = nn.Linear(
                d_model, vocab_size, bias=False, dtype=dtype, device=init_device
            )

        # Vocabulary compression (optional)
        self.vocab_mask: Optional[torch.Tensor] = None
        if config.use_vocab_compression:
            # 고빈도 토큰 마스크 (학습 후 설정)
            self.register_buffer(
                "vocab_mask",
                torch.ones(vocab_size, dtype=torch.bool),
            )

    def _forward_step(
        self,
        h_k: torch.Tensor,
        next_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single MTP step for gradient checkpointing.
        Returns (h_k, logits) tuple.
        """
        # 입력 프로젝션: h^{k-1} + embed(t_{i+k})
        h_k = self.projection(h_k, next_embeddings)

        # MTP 헤드 통과
        if isinstance(self.mtp_head, MTPTransformerHead):
            h_k = self.mtp_head(h_k, attention_mask=attention_mask)
        else:
            h_k = self.mtp_head(h_k)

        # Logits 계산
        logits = self.lm_head(h_k)  # (B, T, V)

        return h_k, logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embeddings: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        MTP forward pass.

        Args:
            hidden_states: Trunk의 출력 (B, T, D)
            input_ids: 입력 토큰 ID (B, T)
            embeddings: 임베딩 모듈
            attention_mask: 어텐션 마스크 (optional)

        Returns:
            List of logits for each prediction step [(B, T, V), ...]
            k=1의 logits[0]은 t_{i+2} 예측, k=2의 logits[1]은 t_{i+3} 예측, ...
        """
        B, T, D = hidden_states.shape
        all_logits = []

        # 현재 hidden states (k=0: trunk output)
        h_k = hidden_states

        # [OPTIMIZED] Use gradient checkpointing if enabled
        use_checkpoint = self.config.use_gradient_checkpointing and self.training

        for k in range(1, self.num_predict_tokens + 1):
            # 다음 토큰의 임베딩 가져오기 (t_{i+k})
            if k == 1:
                next_token_ids = input_ids[:, 1:]  # (B, T-1)
                next_token_ids = F.pad(next_token_ids, (0, 1), value=0)
            else:
                offset = k
                if offset < T:
                    next_token_ids = input_ids[:, offset:]
                    next_token_ids = F.pad(next_token_ids, (0, offset), value=0)
                else:
                    next_token_ids = torch.zeros_like(input_ids)

            next_embeddings = embeddings(next_token_ids)  # (B, T, E)

            # [OPTIMIZED] Gradient Checkpointing for MTP Steps
            if use_checkpoint:
                # Checkpoint each MTP step to save VRAM
                h_k, logits = torch.utils.checkpoint.checkpoint(
                    self._forward_step,
                    h_k,
                    next_embeddings,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                h_k, logits = self._forward_step(h_k, next_embeddings, attention_mask)

            # Vocabulary compression 적용 (optional)
            if self.vocab_mask is not None and self.config.use_vocab_compression:
                logits = logits.masked_fill(~self.vocab_mask, float("-inf"))

            all_logits.append(logits)

        return all_logits

    def set_vocab_mask(self, high_freq_tokens: torch.Tensor):
        """고빈도 토큰으로 vocabulary mask 설정"""
        if self.vocab_mask is not None:
            self.vocab_mask.zero_()
            self.vocab_mask[high_freq_tokens] = True


class MTPLoss(nn.Module):
    """
    MTP Loss 계산 모듈.

    지수 감쇠 가중치를 사용하여 여러 예측 스텝의 loss를 결합합니다.

    수식:
    L_mtp = Σ(k=1 to K) α_k * CE(logits_k, targets_k)
    α_k = β^(k-1) / Σβ^(j-1)  (지수 감쇠)

    References:
    - Meta MTP: arXiv:2404.19737
    """

    def __init__(
        self,
        config: MTPConfig,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.config = config
        self.ignore_index = ignore_index
        self.num_predict_tokens = config.num_predict_tokens
        self.loss_weight_decay = config.loss_weight_decay
        self.mtp_loss_weight = config.mtp_loss_weight

        # 지수 감쇠 가중치 계산
        self._compute_weights()

    def _compute_weights(self):
        """지수 감쇠 가중치 α_k 계산"""
        beta = self.loss_weight_decay
        K = self.num_predict_tokens

        # α_k = β^(k-1) / Σβ^(j-1)
        raw_weights = [beta ** (k - 1) for k in range(1, K + 1)]
        total = sum(raw_weights)
        self.step_weights = [w / total for w in raw_weights]

        log.info(f"MTP loss weights: {[f'{w:.4f}' for w in self.step_weights]}")

    def forward(
        self,
        mtp_logits: List[torch.Tensor],
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MTP loss 계산.

        Args:
            mtp_logits: MTP 헤드의 출력 리스트 [(B, T, V), ...]
            labels: 타겟 레이블 (B, T)
            reduction: Loss reduction 방식

        Returns:
            total_loss: 가중 합산된 총 loss
            step_losses: 각 스텝별 loss (디버깅용)
        """
        total_loss = torch.tensor(0.0, device=labels.device)
        step_losses = {}

        for k, (logits, weight) in enumerate(zip(mtp_logits, self.step_weights), start=1):
            # k번째 예측은 t_{i+k+1}을 예측
            # 따라서 타겟은 labels[:, k:]
            offset = k
            if offset >= labels.shape[1]:
                continue

            # 타겟 시프트
            targets = labels[:, offset:]  # (B, T-offset)

            # Logits 트리밍 (마지막 offset개 위치는 타겟이 없음)
            trimmed_logits = logits[:, :-offset] if offset > 0 else logits

            # Cross-entropy loss
            loss = F.cross_entropy(
                trimmed_logits.reshape(-1, trimmed_logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=self.ignore_index,
                reduction=reduction,
            )

            weighted_loss = weight * loss
            total_loss = total_loss + weighted_loss
            step_losses[f"mtp_loss_k{k}"] = loss.detach()

        # MTP 전체 가중치 적용
        total_loss = total_loss * self.mtp_loss_weight

        return total_loss, step_losses


class MTPSpeculativeDecoder:
    """
    MTP 기반 Self-Speculative Decoding.

    MTP 헤드를 사용하여 드래프트 토큰을 생성하고,
    메인 모델로 검증하여 빠른 추론을 수행합니다.

    References:
    - FastMTP: arXiv:2509.18362
    - Medusa: ICML 2024
    """

    def __init__(
        self,
        model: nn.Module,
        mtp_head: MTPHead,
        config: MTPConfig,
    ):
        self.model = model
        self.mtp_head = mtp_head
        self.config = config
        self.lookahead = config.speculation_lookahead
        self.temperature = config.speculation_temperature

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Self-speculative decoding으로 텍스트 생성.

        Args:
            input_ids: 입력 토큰 ID (B, T)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: Nucleus sampling p

        Returns:
            generated_ids: 생성된 토큰 포함한 전체 시퀀스
            stats: 생성 통계 (acceptance rate, speedup 등)
        """
        device = input_ids.device
        B = input_ids.shape[0]

        generated = input_ids.clone()
        stats = {
            "total_steps": 0,
            "accepted_tokens": 0,
            "generated_tokens": 0,
            "acceptance_rates": [],
        }

        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # 1. 현재 시퀀스로 trunk forward
            with torch.no_grad():
                trunk_output = self._get_trunk_output(generated)
                main_logits = self._get_main_logits(trunk_output)

            # 2. MTP 헤드로 드래프트 토큰 생성
            draft_tokens, draft_probs = self._generate_draft(
                trunk_output,
                generated,
                num_tokens=self.lookahead,
            )

            # 3. 드래프트 + 다음 토큰을 한 번에 검증
            draft_seq = torch.cat([generated, draft_tokens], dim=1)
            with torch.no_grad():
                verify_output = self._get_trunk_output(draft_seq)
                verify_logits = self._get_main_logits(verify_output)

            # 4. 드래프트 토큰 검증 및 수락
            accepted, num_accepted = self._verify_and_accept(
                draft_tokens,
                draft_probs,
                verify_logits[:, -self.lookahead - 1:-1],
                temperature,
                top_p,
            )

            # 5. 수락된 토큰 + 새 토큰 추가
            if num_accepted > 0:
                generated = torch.cat([generated, accepted[:, :num_accepted]], dim=1)
                tokens_generated += num_accepted
                stats["accepted_tokens"] += num_accepted
            else:
                # 드래프트 전부 거절: 기본 샘플링으로 1토큰 생성
                next_token = self._sample(main_logits[:, -1:], temperature, top_p)
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

            stats["total_steps"] += 1
            stats["acceptance_rates"].append(num_accepted / self.lookahead)
            stats["generated_tokens"] = tokens_generated

            # Early stop check
            if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
                break

        # 최종 통계
        stats["avg_acceptance_rate"] = (
            sum(stats["acceptance_rates"]) / len(stats["acceptance_rates"])
            if stats["acceptance_rates"] else 0
        )
        stats["speedup"] = stats["generated_tokens"] / stats["total_steps"]

        return generated, stats

    def _get_trunk_output(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Trunk (backbone) forward"""
        # 모델 구조에 따라 적절히 호출
        if hasattr(self.model, "transformer"):
            return self.model.transformer(input_ids)
        elif hasattr(self.model, "model"):
            return self.model.model(input_ids)
        else:
            # 기본: 전체 forward에서 hidden states 추출
            return self.model(input_ids, output_hidden_states=True).hidden_states[-1]

    def _get_main_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """메인 LM head로 logits 계산"""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(hidden_states)
        else:
            return hidden_states

    def _generate_draft(
        self,
        trunk_output: torch.Tensor,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """MTP 헤드로 드래프트 토큰 생성"""
        B = trunk_output.shape[0]
        device = trunk_output.device

        # MTP forward
        embeddings = self.model.embeddings if hasattr(self.model, "embeddings") else None
        if embeddings is None:
            # HuggingFace 스타일
            embeddings = self.model.model.embed_tokens

        mtp_logits = self.mtp_head(trunk_output, input_ids, embeddings)

        draft_tokens = []
        draft_probs = []

        for k, logits in enumerate(mtp_logits[:num_tokens]):
            # 마지막 위치의 logits만 사용
            last_logits = logits[:, -1]

            if self.temperature > 0:
                probs = F.softmax(last_logits / self.temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = last_logits.argmax(dim=-1, keepdim=True)
                probs = F.softmax(last_logits, dim=-1)

            draft_tokens.append(token)
            draft_probs.append(probs)

        draft_tokens = torch.cat(draft_tokens, dim=1)  # (B, num_tokens)
        return draft_tokens, draft_probs

    def _verify_and_accept(
        self,
        draft_tokens: torch.Tensor,
        draft_probs: List[torch.Tensor],
        verify_logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> Tuple[torch.Tensor, int]:
        """드래프트 토큰 검증 및 수락"""
        B, K = draft_tokens.shape
        accepted = draft_tokens.clone()
        num_accepted = 0

        for k in range(K):
            # 검증 logits에서 확률 계산
            if temperature > 0:
                verify_probs = F.softmax(verify_logits[:, k] / temperature, dim=-1)
            else:
                verify_probs = F.softmax(verify_logits[:, k], dim=-1)

            # 드래프트 토큰의 확률 비교
            draft_token = draft_tokens[:, k]
            draft_prob = draft_probs[k].gather(1, draft_token.unsqueeze(1)).squeeze(1)
            verify_prob = verify_probs.gather(1, draft_token.unsqueeze(1)).squeeze(1)

            # Speculative sampling 수락 기준
            # P(accept) = min(1, p_target / p_draft)
            accept_prob = torch.clamp(verify_prob / (draft_prob + 1e-10), max=1.0)
            rand = torch.rand_like(accept_prob)

            if (rand < accept_prob).all():
                num_accepted += 1
            else:
                break

        return accepted, num_accepted

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """기본 샘플링"""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        probs = F.softmax(logits / temperature, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # [FIX] Scatter back to original positions (must use new tensor)
            probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

        return torch.multinomial(probs.squeeze(1), num_samples=1)


def wrap_model_with_mtp(
    model: nn.Module,
    config: MTPConfig,
    vocab_size: Optional[int] = None,
) -> nn.Module:
    """
    기존 모델에 MTP 헤드를 추가하는 헬퍼 함수.

    Args:
        model: 원본 모델
        config: MTP 설정
        vocab_size: 어휘 크기 (None이면 모델에서 추출)

    Returns:
        MTP가 추가된 모델
    """
    # 모델 정보 추출
    if vocab_size is None:
        if hasattr(model, "config"):
            vocab_size = model.config.vocab_size
        elif hasattr(model, "lm_head"):
            vocab_size = model.lm_head.out_features
        else:
            raise ValueError("vocab_size를 자동으로 추출할 수 없습니다.")

    if hasattr(model, "d_model"):
        d_model = model.d_model
    elif hasattr(model, "config"):
        d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    else:
        raise ValueError("d_model을 자동으로 추출할 수 없습니다.")

    # 임베딩 가중치 공유 옵션
    embed_weight = None
    if config.mtp_head_type == "shared_lm_head":
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "weight"):
            embed_weight = model.embeddings.weight
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_weight = model.model.embed_tokens.weight

    # MTP 헤드 생성
    mtp_head = MTPHead(
        d_model=d_model,
        vocab_size=vocab_size,
        config=config,
        embed_weight=embed_weight,
    )

    # MTP Loss 모듈
    mtp_loss_fn = MTPLoss(config)

    # 모델에 MTP 추가
    model.add_module("mtp_head", mtp_head)
    model.mtp_loss_fn = mtp_loss_fn
    model.mtp_config = config

    # Forward 패치
    original_forward = model.forward

    def mtp_forward(
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_mtp_loss: bool = True,
        **kwargs,
    ):
        # 원본 forward (input_ids 또는 inputs_embeds 사용)
        if inputs_embeds is not None:
            outputs = original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=labels, **kwargs)
        else:
            outputs = original_forward(input_ids=input_ids, labels=labels, **kwargs)

        # MTP loss 계산 (학습 시)
        if labels is not None and return_mtp_loss:
            # Hidden states 추출
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(model, "_last_hidden_states"):
                hidden_states = model._last_hidden_states
            else:
                # MTP loss를 위해 다시 forward 필요
                log.warning("Hidden states를 가져올 수 없어 MTP loss를 건너뜁니다.")
                return outputs

            # 임베딩 모듈 가져오기
            embeddings = model.embeddings if hasattr(model, "embeddings") else model.model.embed_tokens

            # MTP forward
            mtp_logits = model.mtp_head(hidden_states, input_ids, embeddings)

            # MTP loss 계산
            mtp_loss, step_losses = model.mtp_loss_fn(mtp_logits, labels)

            # 총 loss 업데이트
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss + mtp_loss
                outputs.mtp_loss = mtp_loss
                outputs.mtp_step_losses = step_losses

        return outputs

    model.forward = mtp_forward

    # Speculative decoder 추가
    model.mtp_decoder = MTPSpeculativeDecoder(model, mtp_head, config)

    log.info(
        f"MTP 헤드 추가 완료: "
        f"num_predict_tokens={config.num_predict_tokens}, "
        f"head_type={config.mtp_head_type}, "
        f"loss_decay={config.loss_weight_decay}"
    )

    return model
