"""
Naive PyTorch implementation of HSA (Hierarchical Sparse Attention)
Replaces Triton kernel to ensure gradient flow and stability.
Optimized with SDPA and MLA (Multi-Head Latent Attention).
"""

import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat

def naive_hsa(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda=0.0, reg_C=50.0, kv_b_proj=None):
    """
    Naive HSA implementation using SDPA.
    Supports MLA (Multi-Head Latent Attention) Decompression.
    
    Args:
        q: (N, L, H, D)
        k: (N, L_kv, H_kv, D) OR (N, L_kv, 1, Compressed_Dim) if MLA
        v: Same as k
        weights: (N, L, H_kv, K) - Retrieval scores
        indices: (N, L, H_kv, K) - Chunk indices
        kv_b_proj: nn.Linear (Up-Projector) if MLA is used.
        
    Returns:
        out: (N, L, H, D)
    """
    N, L, H, D = q.shape
    L_kv = k.shape[1]
    
    # Check if MLA
    use_mla = (kv_b_proj is not None)
    
    if use_mla:
        # k is Compressed Latent: (N, L_kv, 1, Compressed_Dim)
        # Treat as single head for gather, then expand after decompression
        H_kv = 1
        # Flatten for gather: (N, L_kv, Compressed_Dim)
        k_flat = k.squeeze(2) 
        # v is same object as k in MLA usually, or same shape
        v_flat = v.squeeze(2)
        
        # We need H_kv for weights/indices
        H_retrieval = weights.shape[2] 
    else:
        H_kv = k.shape[2]
        H_retrieval = H_kv
        # Flatten H_kv into N: (N*H_kv, L_kv, D)
        k_flat = k.transpose(1, 2).reshape(N*H_kv, -1, D)
        v_flat = v.transpose(1, 2).reshape(N*H_kv, -1, D)

    K = indices.shape[-1]
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
        
    out = torch.zeros_like(q)
    
    BLOCK_SIZE = 512
    
    for start in range(0, L, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, L)
        curr_bs = end - start
        
        q_block = q[:, start:end]
        w_block = weights[:, start:end]
        idx_block = indices[:, start:end]
        
        # Prepare indices/weights for gather/broadcast
        # Weights are usually (N, B, H_retrieval, K)
        
        # Case 1: MLA (Latent Gather -> Decompress -> Split Heads)
        if use_mla:
            # We gather Latent Vectors first. Latents are shared across heads (or single head).
            # indices are per retrieval head.
            # But Latent Vector is "common" source.
            
            # Optimization: If indices are same across heads, we gather once.
            # But HSA retrieval is per-head (usually).
            # So we must gather per-head indices from the single Latent pool.
            
            # idx_block: (N, B, H_retrieval, K) -> (N*H_retrieval, B, K)
            idx_flat = idx_block.transpose(1, 2).reshape(N*H_retrieval, curr_bs, K)
            w_flat = w_block.transpose(1, 2).reshape(N*H_retrieval, curr_bs, K)
            
            # k_flat (Latent): (N, L_kv, C_Dim) -> Expand to (N*H_retrieval, L_kv, C_Dim) for easy gather?
            # No, that duplicates memory. 
            # We can use the same k_flat for all heads if we adjust indices.
            # Gather indices: (N*H_retrieval, B, K)
            # k_flat: (N, L_kv, C_Dim)
            
            # To gather efficiently:
            # Expand k_flat to (N, H_retrieval, L_kv, C_Dim) -> (NH, L_kv, C_Dim)
            # This is a view/expand, so cheap.
            k_latents_expanded = k_flat.unsqueeze(1).expand(-1, H_retrieval, -1, -1).reshape(N*H_retrieval, L_kv, -1)
            
            gather_heads = H_retrieval
            
        else:
            # Standard GQA
            # If w/idx match k/v heads (H_kv)
            if idx_block.shape[2] == H_kv:
                 idx_flat = idx_block.transpose(1, 2).reshape(N*H_kv, curr_bs, K)
                 w_flat = w_block.transpose(1, 2).reshape(N*H_kv, curr_bs, K)
                 gather_heads = H_kv
            else:
                 # Fallback (Indices are H, K are H_kv) - Rare
                 idx_flat = idx_block.transpose(1, 2).reshape(N*H_kv, curr_bs, K) # Assume H_kv
                 w_flat = w_block.transpose(1, 2).reshape(N*H_kv, curr_bs, K)
                 gather_heads = H_kv

        out_block_accum = torch.zeros(N, H, curr_bs, D, device=q.device, dtype=q.dtype)
        
        for k_idx in range(K):
            curr_chunk_indices = idx_flat[:, :, k_idx]
            
            base_idx = curr_chunk_indices.unsqueeze(-1) * chunk_size
            offset = torch.arange(chunk_size, device=q.device).view(1, 1, chunk_size)
            token_idx = base_idx + offset
            token_idx = token_idx.clamp(0, L_kv - 1)
            
            gather_idx_dim = k_flat.shape[-1] if not use_mla else k_latents_expanded.shape[-1]
            gather_idx = token_idx.reshape(gather_heads * N, -1).unsqueeze(-1).expand(-1, -1, gather_idx_dim)
            
            if use_mla:
                # Gather Latents
                # k_latents_expanded: (N*H_retrieval, L_kv, C_Dim)
                latent_chunk = k_latents_expanded.gather(1, gather_idx).view(N*gather_heads, curr_bs, chunk_size, -1)

                # MLA Decompression (Up-Projection) with GQA-style head mapping
                #
                # HSA-MLA Logic:
                # - Retrieval is per H_retrieval heads (e.g., 8 heads select different chunks)
                # - Each retrieval head maps to a GROUP of attention heads (GQA ratio)
                # - Latent from retrieval head h -> decompress to attention heads in group h
                #
                # kv_b_proj.weight: (H * 2 * D, C_Dim)  where H = total attention heads
                # We reshape to: (C_Dim, H_retrieval, ratio, 2, D)
                # Then each retrieval head's latent projects to its group of attention heads

                C_Dim = latent_chunk.shape[-1]
                ratio = H // gather_heads  # GQA ratio (e.g., 32 / 8 = 4)

                # Reshape weight: (H*2*D, C) -> (C, H, 2, D) -> (C, H_retrieval, ratio, 2, D)
                W_U = kv_b_proj.weight.t().reshape(C_Dim, H, 2, D)
                W_U = W_U.view(C_Dim, gather_heads, ratio, 2, D)  # (C, H_ret, ratio, 2, D)

                # latent_chunk: (N*H_retrieval, B, S, C) -> (N, H_retrieval, B, S, C)
                latent_view = latent_chunk.view(N, gather_heads, curr_bs, chunk_size, C_Dim)

                # Einsum: (N, H_ret, B, S, C) @ (C, H_ret, ratio, 2, D) -> (N, H_ret, ratio, B, S, 2, D)
                # Contract on C, keep H_ret aligned (element-wise on h dimension)
                decompressed = torch.einsum('nhbsc, chrvd -> nhrbsvd', latent_view, W_U)

                # Reshape to (N, H, B, S, 2, D)
                decompressed = decompressed.reshape(N, H, curr_bs, chunk_size, 2, D)

                k_chunk = decompressed[..., 0, :]  # (N, H, B, S, D)
                v_chunk = decompressed[..., 1, :]

                # Flatten for SDPA: (N*H, B, S, D)
                k_chunk = k_chunk.reshape(N*H, curr_bs, chunk_size, D)
                v_chunk = v_chunk.reshape(N*H, curr_bs, chunk_size, D)

                # Update gather_heads for subsequent SDPA reshape
                gather_heads = H
                
            else:
                # Standard GQA Gather
                k_chunk = k_flat.gather(1, gather_idx).view(N, gather_heads, curr_bs, chunk_size, D)
                v_chunk = v_flat.gather(1, gather_idx).view(N, gather_heads, curr_bs, chunk_size, D)
                # Reshape for SDPA: (N*gather_heads, B, chunk, D)
                k_chunk = k_chunk.reshape(N*gather_heads, curr_bs, chunk_size, D)
                v_chunk = v_chunk.reshape(N*gather_heads, curr_bs, chunk_size, D)

            # 2. SDPA Prep
            q_sdpa = q_block.permute(0, 1, 2, 3).reshape(N*curr_bs, H, 1, D)
            
            # K/V for SDPA: (N*B, Heads, L, D)
            # k_chunk: (N*gather_heads, B, chunk, D) -> (N, gather_heads, B, chunk, D)
            # -> (N, B, gather_heads, chunk, D) -> (N*B, gather_heads, chunk, D)
            k_sdpa = k_chunk.view(N, gather_heads, curr_bs, chunk_size, D).permute(0, 2, 1, 3, 4).reshape(N*curr_bs, gather_heads, chunk_size, D)
            v_sdpa = v_chunk.view(N, gather_heads, curr_bs, chunk_size, D).permute(0, 2, 1, 3, 4).reshape(N*curr_bs, gather_heads, chunk_size, D)
            
            # GQA Expansion
            if H != gather_heads:
                num_rep = H // gather_heads
                k_sdpa = k_sdpa.unsqueeze(2).expand(-1, -1, num_rep, -1, -1).reshape(N*curr_bs, H, chunk_size, D)
                v_sdpa = v_sdpa.unsqueeze(2).expand(-1, -1, num_rep, -1, -1).reshape(N*curr_bs, H, chunk_size, D)

            # 3. SDPA
            attn_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=sm_scale)
            
            # 4. Accumulate
            attn_out = attn_out.view(N, curr_bs, H, D).transpose(1, 2)

            # weight: (N, B, H_retrieval, K) -> (N, H_retrieval, B)
            weight = w_block[:, :, :, k_idx].transpose(1, 2)
            H_weight = weight.shape[1]  # Original retrieval heads

            # Expand weights from H_retrieval to H if needed
            if H_weight != H:
                ratio = H // H_weight
                # (N, H_retrieval, B) -> (N, H_retrieval, ratio, B) -> (N, H, B)
                weight = weight.unsqueeze(2).expand(N, H_weight, ratio, curr_bs).reshape(N, H, curr_bs)

            out_block_accum += weight.unsqueeze(-1) * attn_out
            
        out[:, start:end] = out_block_accum.transpose(1, 2)
        
    return out

# Interface wrapper
def HSA(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda=0.0, reg_C=50.0, kv_b_proj=None):
    return naive_hsa(q, k, v, weights, indices, sm_n, chunk_size, sm_scale, reg_lamda, reg_C, kv_b_proj)
