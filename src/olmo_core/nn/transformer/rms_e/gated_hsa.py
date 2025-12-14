"""
Gated Hierarchical Sparse Attention (Gated-HSA)

Hybrid of:
- GatedAttention: sigmoid gating for non-linearity, training stability
- HSA: hierarchical sparse attention for memory efficiency

Benefits:
- Memory efficient: O(L * topk * chunk_size) instead of O(L^2)
- Non-linearity preserved: sigmoid gate maintains GatedAttention's benefits
- Long context: 32K+ training with 4K memory budget
- Training stability: gating mitigates attention sink

Architecture:
    Input -> ChunkingLayer -> RetrievalLayer -> HSA -> Gate -> Output
                                                        ^
                                                        |
                                            sigmoid(g_proj(input))

References:
- GatedAttention: arxiv 2505.06708 (NeurIPS 2025)
- HSA: ant-research/long-context-modeling
- HSA-UltraLong: arxiv 2511.23319
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from olmo_core.config import Config
from olmo_core.nn.layer_norm import RMSNorm, ZeroCenteredRMSNorm

try:
    from .hsa_triton import HSA
    TRITON_AVAILABLE = True # Force Triton to be available
except ImportError:
    TRITON_AVAILABLE = False
    raise ImportError("Triton is required for HSA attention. Please install triton and ensure it's functional.") # Raise error if Triton is not importable


@dataclass
class GatedHSAConfig(Config):
    """Gated-HSA Configuration"""
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: Optional[int] = None  # GQA support
    head_dim: int = 128

    # HSA settings
    chunk_size: int = 64  # Tokens per chunk
    chunk_topk: int = 8   # Top-k chunks to retrieve
    retrieval_dim: int = 256  # Landmark embedding dimension

    # Gating settings (from GatedAttention)
    use_gate: bool = True

    # RoPE settings (Partial RoPE from GatedAttention)
    use_rotary: bool = True
    rope_percentage: float = 0.5
    rope_theta: float = 10000.0

    # Normalization
    use_qk_norm: bool = True
    norm_eps: float = 1e-5

    # Regularization
    dropout: float = 0.0
    enable_softmax: bool = False  # Use softplus-based scoring if False

    def build(self, layer_idx: Optional[int] = None) -> "GatedHSA":
        return GatedHSA(self, layer_idx=layer_idx)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)


class ChunkEncoder(nn.Module):
    """
    Chunk Encoder: Encodes input into chunks with K/V and landmarks.

    From HSA ChunkingLayer:
    - Divides input into fixed-size chunks
    - Projects to K/V for each chunk
    - Generates landmark embeddings for retrieval
    """
    def __init__(self, config: GatedHSAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size
        self.retrieval_dim = config.retrieval_dim

        kv_dim = self.num_kv_heads * self.head_dim

        # K/V projections
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)

        # Landmark projection for retrieval
        self.landmark_proj = nn.Linear(config.hidden_size, config.retrieval_dim, bias=False)

        # Optional K normalization
        if config.use_qk_norm:
            self.k_norm = ZeroCenteredRMSNorm(size=self.head_dim, eps=config.norm_eps)
        else:
            self.k_norm = None

        # Pre-norm for encoding
        self.pre_norm = RMSNorm(size=config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, L, D) input tensor

        Returns:
            mem_k: (B, L, num_kv_heads, head_dim) chunked keys
            mem_v: (B, L, num_kv_heads, head_dim) chunked values
            landmarks: (B, num_chunks, retrieval_heads, retrieval_dim // retrieval_heads)
        """
        B, L, D = hidden_states.shape
        assert L % self.chunk_size == 0, f"Sequence length {L} must be divisible by chunk_size {self.chunk_size}"

        num_chunks = L // self.chunk_size

        # Normalize input
        x = self.pre_norm(hidden_states)

        # Project to K/V
        k = self.k_proj(x)  # (B, L, kv_dim)
        v = self.v_proj(x)  # (B, L, kv_dim)

        # Reshape to (B, L, num_kv_heads, head_dim)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_kv_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_kv_heads)

        # Apply K normalization if enabled
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Generate landmark embeddings (mean pooling per chunk)
        # Reshape to (B, num_chunks, chunk_size, D)
        x_chunks = rearrange(x, 'b (c s) d -> b c s d', s=self.chunk_size)
        chunk_means = x_chunks.mean(dim=2)  # (B, num_chunks, D)
        landmarks = self.landmark_proj(chunk_means)  # (B, num_chunks, retrieval_dim)

        # Reshape landmarks for multi-head retrieval
        retrieval_heads = self.num_kv_heads
        landmarks = rearrange(landmarks, 'b c (h d) -> b c h d', h=retrieval_heads)

        return k, v, landmarks


class ChunkRetriever(nn.Module):
    """
    Chunk Retriever: Selects top-k relevant chunks based on query-landmark similarity.

    From HSA RetrievalLayer:
    - Computes query embeddings
    - Scores chunks using landmark similarity
    - Returns top-k chunk indices and weights
    """
    def __init__(self, config: GatedHSAConfig):
        super().__init__()
        self.config = config
        self.retrieval_dim = config.retrieval_dim
        self.chunk_size = config.chunk_size
        self.chunk_topk = config.chunk_topk
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.enable_softmax = config.enable_softmax

        # Query projection for retrieval
        self.query_proj = nn.Linear(config.hidden_size, config.retrieval_dim, bias=False)
        self.pre_norm = RMSNorm(size=config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, L, D)
        landmarks: torch.Tensor,  # (B, num_chunks, retrieval_heads, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, L, D)
            landmarks: (B, num_chunks, retrieval_heads, retrieval_dim // retrieval_heads)

        Returns:
            weights: (B, L // chunk_size, num_kv_heads, chunk_topk) chunk weights
            indices: (B, L // chunk_size, num_kv_heads, chunk_topk) chunk indices
        """
        B, L, D = hidden_states.shape
        num_chunks = L // self.chunk_size
        retrieval_heads = landmarks.shape[2]
        head_dim = landmarks.shape[3]

        # Get query embeddings (one per chunk position)
        x = self.pre_norm(hidden_states)
        # Sample one query per chunk (at chunk boundaries)
        q_emb = self.query_proj(x[:, ::self.chunk_size, :])  # (B, num_chunks, retrieval_dim)
        q_emb = rearrange(q_emb, 'b c (h d) -> b c h d', h=retrieval_heads)

        # Compute scores: (B, num_chunks, retrieval_heads, num_chunks)
        scores = torch.einsum('b c h d, b n h d -> b c h n', q_emb, landmarks)
        scores = scores / math.sqrt(head_dim)

        # Apply causal mask (chunk c can only attend to chunks < c)
        C = num_chunks
        c_indices = torch.arange(C, device=hidden_states.device).view(1, C, 1, 1)
        n_indices = torch.arange(C, device=hidden_states.device).view(1, 1, 1, C)
        causal_mask = c_indices <= n_indices  # (1, C, 1, C)
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Get top-k chunks
        chunk_topk = min(self.chunk_topk, C)
        _, indices = torch.topk(scores, k=chunk_topk, dim=-1)  # (B, C, h, k)

        # Sort indices for causal consistency
        indices, _ = indices.sort(dim=-1)

        # Gather scores for selected chunks
        scores_selected = scores.gather(dim=-1, index=indices)

        # Compute weights
        if self.enable_softmax:
            # Softmax-based weights
            all_neg_inf = (scores_selected == float('-inf')).all(dim=-1, keepdim=True)
            scores_selected = torch.where(scores_selected == float('-inf'), torch.tensor(-1e7, device=scores.device), scores_selected)
            weights = F.softmax(scores_selected.float(), dim=-1)
            weights = weights.masked_fill(all_neg_inf, 0.0)
        else:
            # Softplus-based weights (from HSA paper)
            scores_selected = scores_selected.float()
            softplus_x = F.softplus(scores_selected, threshold=15.0)
            softplus_cumsum = torch.cumsum(softplus_x, dim=-1)
            weights = (scores_selected - softplus_cumsum).exp()

        return weights, indices


class GatedHSA(nn.Module):
    """
    Gated Hierarchical Sparse Attention

    Combines:
    - HSA's memory-efficient sparse attention
    - GatedAttention's sigmoid gating for non-linearity
    """
    def __init__(
        self,
        config: GatedHSAConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size
        self.chunk_topk = config.chunk_topk
        self.scale = config.head_dim ** -0.5

        inner_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Q projection
        self.q_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.hidden_size, bias=False)

        # HSA components
        self.chunk_encoder = ChunkEncoder(config)
        self.chunk_retriever = ChunkRetriever(config)

        # Gate projection (from GatedAttention)
        if config.use_gate:
            self.g_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        else:
            self.g_proj = None

        # Normalizations (from GatedAttention)
        if config.use_qk_norm:
            self.q_norm = ZeroCenteredRMSNorm(size=self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None

        self.o_norm = RMSNorm(size=self.head_dim, eps=config.norm_eps)
        self.pre_norm = RMSNorm(size=config.hidden_size, eps=config.norm_eps)

        # RoPE (Partial, from GatedAttention)
        if config.use_rotary:
            self.rotary_dim = int(self.head_dim * config.rope_percentage)
            self.rotary = RotaryEmbedding(self.rotary_dim, base=int(config.rope_theta))
        else:
            self.rotary = None

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Partial RoPE to Q and K"""
        if self.rotary is None:
            return q, k

        q_rope, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        k_rope, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
        q_rope, k_rope = self.rotary(q_rope, k_rope, seq_len=seq_len)
        q = torch.cat([q_rope, q_pass], dim=-1)
        k = torch.cat([k_rope, k_pass], dim=-1)
        return q, k

    def _hsa_attention(
        self,
        q: torch.Tensor,  # (B, L, num_heads, head_dim)
        k: torch.Tensor,  # (B, L, num_kv_heads, head_dim)
        v: torch.Tensor,  # (B, L, num_kv_heads, head_dim)
        weights: torch.Tensor,  # (B, num_chunks, num_kv_heads, chunk_topk)
        indices: torch.Tensor,  # (B, num_chunks, num_kv_heads, chunk_topk)
    ) -> torch.Tensor:
        """
        HSA attention: Attend to selected chunks only using Triton kernel.
        """
        if not TRITON_AVAILABLE:
            raise ImportError("Triton is required for HSA attention. Please install triton.")

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        weights = weights.contiguous()
        indices = indices.contiguous().int() # Indices must be int32

        # Scale factor
        sm_scale = self.scale
        # sm_n: smoothing factor or similar? Paper says 1.0 usually or learned.
        # Based on test code: sm_n=1.0
        sm_n = 1.0

        # Call Triton HSA
        # HSA(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda=0.01, reg_C=50.0)
        output = HSA(
            q, k, v, 
            weights, indices, 
            sm_n, 
            self.chunk_size, 
            sm_scale, 
            reg_lamda=0.0, # Regularization off by default for basic usage
            reg_C=0.0
        )
        
        return output


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass for Gated-HSA.

        Args:
            hidden_states: (B, L, D) input tensor

        Returns:
            output: (B, L, D) attention output
            cache: Optional cache for inference
        """
        B, L, D = hidden_states.shape

        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        # 1. Encode chunks and get K/V, landmarks
        k, v, landmarks = self.chunk_encoder(hidden_states)

        # 2. Retrieve top-k chunks
        weights, indices = self.chunk_retriever(hidden_states, landmarks)

        # 3. Compute Q
        x = self.pre_norm(hidden_states)
        q = self.q_proj(x)
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)

        # Apply Q normalization
        if self.q_norm is not None:
            q = self.q_norm(q)

        # 4. Apply RoPE
        # Reshape K for RoPE: (B, L, num_kv_heads, head_dim) -> (B, num_kv_heads, L, head_dim)
        k_transposed = k.transpose(1, 2)
        q_transposed = q.transpose(1, 2)
        q_transposed, k_transposed = self._apply_rope(q_transposed, k_transposed, L_padded)
        q = q_transposed.transpose(1, 2)
        k = k_transposed.transpose(1, 2)

        # 5. HSA attention
        attn_output = self._hsa_attention(q, k, v, weights, indices)

        # 6. Apply output normalization
        attn_output = self.o_norm(attn_output)

        # 7. Apply gate (from GatedAttention)
        if self.g_proj is not None:
            g = self.g_proj(hidden_states)
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            attn_output = attn_output * torch.sigmoid(g)

        # 8. Output projection
        attn_output = rearrange(attn_output, 'b l h d -> b l (h d)')
        output = self.o_proj(attn_output)

        # Remove padding
        if pad_len > 0:
            output = output[:, :L, :]

        return output, None


# Convenience function for creating Gated-HSA config from existing configs
def create_gated_hsa_config(
    hidden_size: int = 2048,
    num_heads: int = 16,
    num_kv_heads: Optional[int] = None,
    head_dim: int = 128,
    chunk_size: int = 64,
    chunk_topk: int = 8,
    **kwargs
) -> GatedHSAConfig:
    """Create Gated-HSA config with sensible defaults"""
    return GatedHSAConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
        chunk_topk=chunk_topk,
        retrieval_dim=min(256, hidden_size // 4),
        **kwargs
    )
