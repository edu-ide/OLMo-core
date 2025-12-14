
"""
DeepSeek-V3 Multi-Token Prediction (MTP) Module

Reference: DeepSeek-V3 Technical Report
MTP allows predicting multiple future tokens sequentially to accelerate training and inference.

Structure of MTP Module k:
1. Input: Concatenation of (RMSNorm(Hidden_k-1), Embed(Token_t+k)) ??
   Actually, DeepSeek-V3 spec:
   - Input: Main Model Hidden States (for MTP 1)
   - For MTP k: Input is output of MTP k-1 ??
   - Each module has:
     - RMSNorm
     - Linear Projection (2*d_model -> d_model) : Combines Hidden + Embedding?
     - Transformer Block (Attention + MoE)
     - Output Head (Shared)

   Correct Flow (based on paper):
   - MTP_1 Input: Concat[RMSNorm(Main_Hidden), Embed(Token_t+1)] -> Proj -> Layer -> Head -> Pred(Token_t+2)
   - Wait, MTP is for predicting t+1, t+2...
   - Main Model predicts t+1.
   - MTP 1 predicts t+2.
   
   Input to MTP 1 (predicting t+2):
   - Needs info about t (Main Hidden) and t+1 (Ground Truth Embedding during training).
   - So Input = Concat(RMSNorm(Main_Hidden), Embed(Token_t+1))
   
   Input to MTP k (predicting t+k+1):
   - Needs info about t+k (Output of MTP k-1) and t+k (Ground Truth Embedding).
   - Input = Concat(RMSNorm(MTP_k-1_Hidden), Embed(Token_t+k))
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

from .config import RMSEConfig
from .utils import RMSNorm
from .layer import RMSEDecoderLayer

class DeepSeekMTPModule(nn.Module):
    """
    Single MTP Module for DeepSeek-V3.
    Predicts token t+k+1 given hidden state at t+k-1 and embedding of token t+k.
    """
    def __init__(self, config: RMSEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        
        # 1. RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 2. Projection (Input: Hidden + Embedding -> 2*d -> d)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        
        # 3. Transformer Layer (Lightweight or Full)
        # DeepSeek-V3 uses a full decoder layer (Attn + MoE) for MTP modules
        # We reuse RMSEDecoderLayer but potentially with lighter settings if config allows
        # For now, use standard layer (all layers have DeltaNet + MoE by default)
        self.layer = RMSEDecoderLayer(config, layer_idx=layer_idx)

    def forward(
        self, 
        hidden_states: torch.Tensor, # Output from previous stage (Main or MTP k-1)
        current_token_embeds: torch.Tensor, # Embedding of the "current" token for this stage
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, L, D]
            current_token_embeds: [B, L, D] (Embedding of token t+k)
        """
        # 1. Norm & Concat & Proj
        normed_hidden = self.norm(hidden_states)
        combined = torch.cat([normed_hidden, current_token_embeds], dim=-1)
        x = self.proj(combined)
        
        # 2. Transformer Layer
        # MTP layer maintains causal mask
        x, aux_loss = self.layer(x, **kwargs)
        
        return x, aux_loss

class DeepSeekMTPHead(nn.Module):
    """
    Container for multiple MTP modules.
    """
    def __init__(self, config: RMSEConfig):
        super().__init__()
        self.config = config
        self.num_mtp = config.num_mtp_tokens
        
        self.modules_list = nn.ModuleList([
            DeepSeekMTPModule(config, layer_idx=i)
            for i in range(self.num_mtp)
        ])
        
    def forward(
        self,
        main_hidden_states: torch.Tensor, # [B, L, D] from Main Model
        input_ids: torch.Tensor, # [B, L] Original Input IDs
        embed_tokens: nn.Embedding, # Shared embedding layer
        lm_head: nn.Linear, # Shared output head
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Prepare embeddings for all tokens (Ground Truth)
        # For MTP k (predicting t+k+1), we need embedding of token t+k.
        # Main model input was tokens 0..L-1.
        # Main model output at pos t is prediction for t+1.
        # MTP 1 at pos t predicts t+2. Input needs embedding of t+1.
        
        # During training, we have ground truth for t+1, t+2...
        # We need to shift input_ids to get embeddings for next tokens.
        
        # Full embedding sequence: [B, L] -> [B, L, D]
        # But we need shifted embeddings.
        # MTP 1 input at step t: Main_Hidden[t] + Embed(Token[t+1])
        
        B, L, D = main_hidden_states.shape
        
        # Get embeddings for the whole sequence (including potential future if available)
        # In standard training, input_ids is [0...L-1]. labels is [1...L].
        # Embed(Token[t+1]) corresponds to embedding of labels!
        
        # However, we only have embeddings for input_ids.
        # If we want Embed(Token[t+1]), we need the next token.
        # In causal LM training, 'labels' contains the next tokens.
        
        if labels is None:
            # Inference mode: MTP not used or needs speculative sampling logic
            return {}

        # Get embeddings of target tokens (labels)
        # Replace -100 with 0 for embedding lookup (masked later)
        target_ids = labels.clone()
        target_ids[target_ids == -100] = 0
        target_embeds = embed_tokens(target_ids) # [B, L, D] (Embedding of t+1)
        
        mtp_loss = 0.0
        mtp_aux_loss = 0.0
        current_hidden = main_hidden_states
        
        # We need to compute MTP for k=1..num_mtp
        # MTP 1: Predicts labels shifted by 1 (t+2)
        # Input: current_hidden (t), target_embeds (t+1)
        
        valid_L = L - self.num_mtp # Valid length decreases as we look further ahead
        
        for k, module in enumerate(self.modules_list):
            # 1. Prepare inputs for this MTP module
            # We can only compute up to L - 1 - k (since we need targets for k+1 steps ahead)
            
            # Slice inputs to valid length
            # current_hidden: [B, L, D] -> use [:, :-1, :] for next step?
            # No, MTP operates per position.
            # At pos t, MTP 1 predicts t+2. We need Embed(t+1).
            # Embed(t+1) is target_embeds[:, t].
            # Target for MTP 1 is labels[:, t+1] (Token t+2).
            
            # Valid range for MTP k (0-indexed):
            # Can compute for t where t+k+1 < Sequence Length?
            # Actually, we have fixed sequence length L.
            # labels covers 1..L.
            # MTP 1 target is 2..L+1 (we don't have L+1).
            # So MTP 1 can only train on 0..L-2.
            
            slice_len = L - (k + 1)
            if slice_len <= 0:
                break
                
            # Inputs
            h_in = current_hidden[:, :slice_len, :]
            emb_in = target_embeds[:, :slice_len, :] # Embedding of t+1
            
            # Forward
            h_out, aux = module(h_in, emb_in)
            mtp_aux_loss += aux if aux is not None else 0.0
            
            # Prediction
            logits = lm_head(h_out) # [B, slice_len, V]
            
            # Targets: Shift labels by k+1
            # labels is t+1. We need t+1 + (k+1) = t+k+2?
            # Wait.
            # Main model: Input t -> Hidden t -> Predict t+1 (Label[:, t])
            # MTP 1: Input Hidden t, Embed t+1 -> Predict t+2 (Label[:, t+1])
            
            mtp_targets = labels[:, 1+k : 1+k+slice_len]
            
            # Compute Loss
            loss_k = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                mtp_targets.reshape(-1),
                ignore_index=-100
            )
            
            mtp_loss += loss_k * self.config.mtp_loss_weight
            
            # Prepare for next MTP module
            # Next module needs Hidden t (from MTP 1) and Embed t+2.
            # But Embed t+2 is target_embeds[:, 1+k].
            # h_out is [B, slice_len, D]. Next iter needs slice_len - 1.
            current_hidden = h_out # Update hidden for next module chain
            
            # Adjust target embeds pointer?
            # Next iter k+1:
            # slice_len' = L - (k+2)
            # h_in = current_hidden[:, :slice_len', :]
            # emb_in = target_embeds[:, 1+k : 1+k+slice_len', :] 
            # (Embedding of t+2, aligned with hidden t)
            
            # Actually, let's keep full tensors and slice inside loop?
            # No, 'current_hidden' shrinks.
            # So 'target_embeds' must also be shifted/sliced relative to current_hidden.
            
            # Update target_embeds for next round:
            # We consumed target_embeds[:, 0] (t+1).
            # Next round needs t+2.
            target_embeds = target_embeds[:, 1:, :]
            
        return {
            "mtp_loss": mtp_loss,
            "mtp_aux_loss": mtp_aux_loss
        }

