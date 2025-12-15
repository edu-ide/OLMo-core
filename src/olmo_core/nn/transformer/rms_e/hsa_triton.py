"""
Absorbed MLA HSA (Hierarchical Sparse Attention with Multi-Head Latent Attention)

Key Innovation: Weight Absorption (DeepSeek V3 style)
- Instead of decompressing Latent -> K, V (memory explosion)
- Absorb Up-projection weight into Query
- Compute attention directly in latent space

Memory: O(L × C_dim) instead of O(L × H × D)
"""

import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat


def absorbed_mla_hsa(q, latent, weights, indices, sm_n, chunk_size, sm_scale,
                     kv_b_proj, v_head_dim=None):
    """
    Absorbed MLA HSA - Compute attention directly in latent space.

    Args:
        q: (N, L, H, D) - Query
        latent: (N, L_kv, C_Dim) - Compressed KV latent vectors
        weights: (N, L, H_kv, K) - Retrieval scores
        indices: (N, L, H_kv, K) - Chunk indices
        kv_b_proj: nn.Linear - Up-projection weight (C_Dim -> H * (D_k + D_v))
        v_head_dim: int - Value head dim (default: same as D)

    Returns:
        out: (N, L, H, D_v)
    """
    N, L, H, D = q.shape
    L_kv = latent.shape[1]
    C_Dim = latent.shape[2]
    H_retrieval = weights.shape[2]
    K = indices.shape[-1]

    if v_head_dim is None:
        v_head_dim = D

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(C_Dim)  # Scale by latent dim for absorbed attention

    # === Weight Absorption ===
    # kv_b_proj.weight: (H * (D_k + D_v), C_Dim)
    # Split into W_UK (for K) and W_UV (for V)
    W = kv_b_proj.weight  # (H * (D + D_v), C_Dim)
    W_UK = W[:H * D, :].view(H, D, C_Dim)      # (H, D, C)
    W_UV = W[H * D:, :].view(H, v_head_dim, C_Dim)  # (H, D_v, C)

    # Absorb W_UK into Query: Q_absorbed = Q @ W_UK
    # q: (N, L, H, D), W_UK: (H, D, C) -> Q_absorbed: (N, L, H, C)
    q_absorbed = torch.einsum('nlhd, hdc -> nlhc', q, W_UK)

    out = torch.zeros(N, L, H, v_head_dim, device=q.device, dtype=q.dtype)

    BLOCK_SIZE = 512
    ratio = H // H_retrieval  # GQA ratio

    for start in range(0, L, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, L)
        curr_bs = end - start

        q_block = q_absorbed[:, start:end]  # (N, B, H, C)
        w_block = weights[:, start:end]      # (N, B, H_ret, K)
        idx_block = indices[:, start:end]    # (N, B, H_ret, K)

        out_block_accum = torch.zeros(N, H, curr_bs, v_head_dim, device=q.device, dtype=q.dtype)

        for k_idx in range(K):
            # Get chunk indices for this k
            chunk_idx = idx_block[:, :, :, k_idx]  # (N, B, H_ret)

            # Gather latent chunks
            # chunk_idx: (N, B, H_ret) -> expand for gathering
            base_idx = chunk_idx.unsqueeze(-1) * chunk_size  # (N, B, H_ret, 1)
            offset = torch.arange(chunk_size, device=q.device)  # (chunk_size,)
            token_idx = (base_idx + offset).clamp(0, L_kv - 1)  # (N, B, H_ret, chunk_size)

            # Gather latents: (N, B, H_ret, chunk_size, C)
            # Use advanced indexing
            batch_idx = torch.arange(N, device=q.device).view(N, 1, 1, 1).expand_as(token_idx)
            latent_chunk = latent[batch_idx, token_idx]  # (N, B, H_ret, chunk_size, C)

            # === Absorbed Attention (in latent space) ===
            # Q_absorbed: (N, B, H, C) -> (N, B, H_ret, ratio, C)
            q_block_grouped = q_block.view(N, curr_bs, H_retrieval, ratio, C_Dim)

            # Attention scores: Q_absorbed @ Latent^T
            # (N, B, H_ret, ratio, C) @ (N, B, H_ret, C, chunk) -> (N, B, H_ret, ratio, chunk)
            latent_t = latent_chunk.transpose(-1, -2)  # (N, B, H_ret, C, chunk)
            scores = torch.einsum('nbhrc, nbhcs -> nbhrs', q_block_grouped, latent_t) * sm_scale

            # Softmax
            attn_weights = F.softmax(scores, dim=-1)  # (N, B, H_ret, ratio, chunk)

            # === Value Projection (only here, not before!) ===
            # V = Latent @ W_UV^T: (N, B, H_ret, chunk, C) @ (H, D_v, C)^T
            # But we need per-head V, so reshape W_UV for GQA
            W_UV_grouped = W_UV.view(H_retrieval, ratio, v_head_dim, C_Dim)  # (H_ret, ratio, D_v, C)

            # V: (N, B, H_ret, chunk, C) @ (H_ret, ratio, C, D_v) -> (N, B, H_ret, ratio, chunk, D_v)
            # Use einsum for clarity
            V = torch.einsum('nbhsc, hrdc -> nbhrsd', latent_chunk, W_UV_grouped)

            # Weighted sum: attn @ V
            # (N, B, H_ret, ratio, chunk) @ (N, B, H_ret, ratio, chunk, D_v) -> (N, B, H_ret, ratio, D_v)
            attn_out = torch.einsum('nbhrs, nbhrsd -> nbhrd', attn_weights, V)

            # Reshape: (N, B, H_ret, ratio, D_v) -> (N, B, H, D_v) -> (N, H, B, D_v)
            attn_out = attn_out.reshape(N, curr_bs, H, v_head_dim).transpose(1, 2)

            # Apply retrieval weights
            weight = w_block[:, :, :, k_idx]  # (N, B, H_ret)
            # Expand: (N, B, H_ret) -> (N, B, H_ret, ratio) -> (N, B, H) -> (N, H, B)
            weight = weight.unsqueeze(-1).expand(N, curr_bs, H_retrieval, ratio)
            weight = weight.reshape(N, curr_bs, H).transpose(1, 2)

            out_block_accum += weight.unsqueeze(-1) * attn_out

        out[:, start:end] = out_block_accum.transpose(1, 2)

    return out


def naive_hsa(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda=0.0, reg_C=50.0, kv_b_proj=None):
    """
    Naive HSA with optional Absorbed MLA.

    If kv_b_proj is provided, uses Absorbed MLA (memory efficient).
    Otherwise, uses standard GQA attention.
    """
    N, L, H, D = q.shape

    # === Absorbed MLA Path (Recommended) ===
    if kv_b_proj is not None:
        # k is actually latent: (N, L_kv, 1, C_Dim) -> (N, L_kv, C_Dim)
        latent = k.squeeze(2) if k.dim() == 4 else k
        return absorbed_mla_hsa(q, latent, weights, indices, sm_n, chunk_size, sm_scale, kv_b_proj)

    # === Standard GQA Path ===
    L_kv = k.shape[1]
    H_kv = k.shape[2]
    K = indices.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Flatten H_kv into N: (N*H_kv, L_kv, D)
    k_flat = k.transpose(1, 2).reshape(N*H_kv, -1, D)
    v_flat = v.transpose(1, 2).reshape(N*H_kv, -1, D)

    out = torch.zeros_like(q)
    BLOCK_SIZE = 512

    for start in range(0, L, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, L)
        curr_bs = end - start

        q_block = q[:, start:end]
        w_block = weights[:, start:end]
        idx_block = indices[:, start:end]

        idx_flat = idx_block.transpose(1, 2).reshape(N*H_kv, curr_bs, K)

        out_block_accum = torch.zeros(N, H, curr_bs, D, device=q.device, dtype=q.dtype)

        for k_idx in range(K):
            curr_chunk_indices = idx_flat[:, :, k_idx]

            base_idx = curr_chunk_indices.unsqueeze(-1) * chunk_size
            offset = torch.arange(chunk_size, device=q.device).view(1, 1, chunk_size)
            token_idx = base_idx + offset
            token_idx = token_idx.clamp(0, L_kv - 1)

            gather_idx = token_idx.reshape(H_kv * N, -1).unsqueeze(-1).expand(-1, -1, D)

            k_chunk = k_flat.gather(1, gather_idx).view(N, H_kv, curr_bs, chunk_size, D)
            v_chunk = v_flat.gather(1, gather_idx).view(N, H_kv, curr_bs, chunk_size, D)

            # SDPA
            q_sdpa = q_block.permute(0, 1, 2, 3).reshape(N*curr_bs, H, 1, D)
            k_sdpa = k_chunk.permute(0, 2, 1, 3, 4).reshape(N*curr_bs, H_kv, chunk_size, D)
            v_sdpa = v_chunk.permute(0, 2, 1, 3, 4).reshape(N*curr_bs, H_kv, chunk_size, D)

            # GQA Expansion
            if H != H_kv:
                num_rep = H // H_kv
                k_sdpa = k_sdpa.unsqueeze(2).expand(-1, -1, num_rep, -1, -1).reshape(N*curr_bs, H, chunk_size, D)
                v_sdpa = v_sdpa.unsqueeze(2).expand(-1, -1, num_rep, -1, -1).reshape(N*curr_bs, H, chunk_size, D)

            attn_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=sm_scale)
            attn_out = attn_out.view(N, curr_bs, H, D).transpose(1, 2)

            weight = w_block[:, :, :, k_idx].transpose(1, 2)
            if weight.shape[1] != H:
                ratio = H // weight.shape[1]
                weight = weight.unsqueeze(2).expand(N, -1, ratio, curr_bs).reshape(N, H, curr_bs)

            out_block_accum += weight.unsqueeze(-1) * attn_out

        out[:, start:end] = out_block_accum.transpose(1, 2)

    return out


# Interface wrapper
def HSA(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda=0.0, reg_C=50.0, kv_b_proj=None):
    return naive_hsa(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda, reg_C, kv_b_proj)
