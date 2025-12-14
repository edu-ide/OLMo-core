"""
TRM-Style RMS-E Model: Ouro + TRM + HSA-UltraLong

Combines:
- TRM: z_H/z_L dual latent states, nested loops
- Ouro: Early exit, LoopLM
- Our: HSA + Gated-SWA + GatedDeltaNet + MoE

Architecture:
- z_H: High-level reasoning state (output)
- z_L: Low-level reasoning state (detailed)
- H_cycles: Outer loop (update z_H)
- L_cycles: Inner loop (update z_L)

Reference:
- TRM: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Ouro: https://arxiv.org/abs/2510.25741
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .config import RMSEConfig
from .layer import RMSEDecoderLayer
from .utils import RMSNorm  # Used for final output norm


@dataclass
class TRMCarry:
    """TRM-style carry state for recursive reasoning."""
    z_H: torch.Tensor  # High-level reasoning state [B, L, D]
    z_L: torch.Tensor  # Low-level reasoning state [B, L, D]


class TRMReasoningModule(nn.Module):
    """
    TRM-style reasoning module using RMS-E layers.
    Same network updates both z_H and z_L.

    TRM original: hidden = hidden + injection (direct add, no norm)

    Architecture (matching TRM):
    - L_layers blocks in sequence
    - Same module used for both z_L and z_H updates
    """

    def __init__(self, config: RMSEConfig):
        super().__init__()
        self.config = config

        # L_layers blocks (TRM default: 2)
        L_layers = getattr(config, 'L_layers', 2)
        self.layers = nn.ModuleList([
            RMSEDecoderLayer(config, layer_idx=i)
            for i in range(L_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional input injection.

        Args:
            hidden_states: Current state [B, L, D]
            input_injection: Optional injection (z_H + input for z_L update)

        Returns:
            updated_states, aux_loss
        """
        # TRM: direct addition without normalization
        if input_injection is not None:
            hidden_states = hidden_states + input_injection

        # Process through all L_layers blocks
        total_aux_loss = 0.0
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, **kwargs)
            total_aux_loss = total_aux_loss + aux_loss

        return hidden_states, total_aux_loss


class TRMStyleRMSE(nn.Module):
    """
    TRM-Style RMS-E Model with dual latent states.

    Architecture:
    - z_H: High-level state (updated H_cycles times, used for output)
    - z_L: Low-level state (updated H_cycles × L_cycles times)

    Loop structure:
        for H in H_cycles:
            for L in L_cycles:
                z_L = block(z_L, z_H + input)
            z_H = block(z_H, z_L)
    """

    def __init__(self, config: RMSEConfig):
        super().__init__()
        self.config = config

        # TRM-style cycles
        self.H_cycles = config.H_cycles if hasattr(config, 'H_cycles') else 3
        self.L_cycles = config.L_cycles if hasattr(config, 'L_cycles') else 6

        # ACT (Adaptive Computation Time)
        self.use_act = getattr(config, 'use_act', True)
        self.act_max_steps = getattr(config, 'act_max_steps', self.H_cycles)
        self.act_exploration_prob = getattr(config, 'act_exploration_prob', 0.1)

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared reasoning module (TRM uses single network)
        self.reasoning = TRMReasoningModule(config)

        # Initial state projections (learned init like TRM)
        self.H_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

        # Q-head for ACT halting decision (TRM style)
        # Outputs: [q_halt, q_continue] - if q_halt > q_continue, halt
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)
        # Initialize Q-head to prefer continuing (negative bias for halt)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # Start with low halt probability

        # Output
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def _init_carry(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> TRMCarry:
        """Initialize z_H and z_L states."""
        return TRMCarry(
            z_H=self.H_init.view(1, 1, -1).expand(batch_size, seq_len, -1).to(dtype),
            z_L=self.L_init.view(1, 1, -1).expand(batch_size, seq_len, -1).to(dtype),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with TRM-style nested loops and ACT.

        Args:
            input_ids: [B, L]
            attention_mask: [B, L] (optional)
            labels: [B, L] (optional, for loss computation)

        Returns:
            Dict with logits, loss, aux_loss, q_halt_logits, q_continue_logits, halt_step
        """
        B, L = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        # Input embeddings
        input_embeds = self.embed_tokens(input_ids)

        # Initialize carry states
        carry = self._init_carry(B, L, device, dtype)
        z_H, z_L = carry.z_H, carry.z_L

        # TRM-style nested loops with ACT
        total_aux_loss = 0.0
        halt_step = self.H_cycles  # Default: run all steps
        q_halt_logits = None
        q_continue_logits = None

        for h_step in range(self.H_cycles):
            is_last_step = (h_step == self.H_cycles - 1)

            # Gradient truncation: only compute grad for last step
            context = torch.no_grad() if not is_last_step else torch.enable_grad()

            with context:
                # Inner loop: Update z_L multiple times
                for l_step in range(self.L_cycles):
                    injection = z_H + input_embeds
                    z_L, aux = self.reasoning(z_L, input_injection=injection, **kwargs)
                    if is_last_step:
                        total_aux_loss += aux if aux is not None else 0.0

                # Outer loop: Update z_H once using z_L
                z_H, aux = self.reasoning(z_H, input_injection=z_L, **kwargs)
                if is_last_step:
                    total_aux_loss += aux if aux is not None else 0.0

            # ACT: Compute Q-values for halting decision
            if self.use_act:
                # Use first token position for halting decision (like TRM)
                q_logits = self.q_head(z_H[:, 0]).float()  # [B, 2]
                q_halt_logits = q_logits[:, 0]
                q_continue_logits = q_logits[:, 1]

                # During inference: early exit if q_halt > q_continue
                if not self.training and h_step < self.H_cycles - 1:
                    if (q_halt_logits > q_continue_logits).all():
                        halt_step = h_step + 1
                        break

                # During training: exploration (random early exit for Q-learning)
                if self.training and h_step < self.H_cycles - 1:
                    import random
                    if random.random() < self.act_exploration_prob:
                        # Random exploration: might halt early
                        min_steps = random.randint(1, self.H_cycles)
                        if h_step + 1 >= min_steps and (q_halt_logits > q_continue_logits).any():
                            halt_step = h_step + 1
                            break

        # Output from z_H (high-level state)
        hidden_states = self.norm(z_H)
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "z_H": z_H,  # For analysis/debugging
            "z_L": z_L,
            # ACT outputs for Q-learning
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "halt_step": halt_step,
        }


class TRMStyleRMSEForCausalLM(nn.Module):
    """Wrapper for causal LM with TRM-style RMS-E."""

    def __init__(self, config: RMSEConfig):
        super().__init__()
        self.config = config
        self.model = TRMStyleRMSE(config)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids, attention_mask, labels, **kwargs)

        # Combine losses (add aux_loss weighted by alpha)
        total_loss = outputs["loss"]
        aux_loss = outputs["aux_loss"]
        if total_loss is not None and aux_loss is not None:
            if isinstance(aux_loss, (int, float)):
                aux_loss = torch.tensor(aux_loss, device=input_ids.device)
            total_loss = total_loss + self.config.aux_loss_alpha * aux_loss

        # Return dict format matching RMSEForCausalLM interface
        return {
            "loss": total_loss,
            "logits": outputs["logits"],
            "aux_loss": outputs["aux_loss"],
            "z_H": outputs.get("z_H"),
            "z_L": outputs.get("z_L"),
        }

    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, **kwargs):
        """Simple greedy/sampling generation."""
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids, **kwargs)
                logits = outputs["logits"][:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Stop at EOS (assuming 151643 is EOS for Qwen)
                if next_token.item() == 151643:
                    break

        return input_ids

    def apply_fp8(self, use_mxfp8: bool = False) -> None:
        """Apply FP8 training (stub for compatibility)."""
        # TRM-style model uses same block structure, FP8 can be applied similarly
        pass

    def get_fp8_recipe(self):
        """Get FP8 recipe (stub for compatibility)."""
        return None

    def apply_fsdp(self, dp_mesh=None) -> None:
        """Apply FSDP sharding (stub for compatibility)."""
        pass

    def reset_memory(self):
        """Reset any stateful memory (for gradient accumulation)."""
        # TRM doesn't have persistent memory states between forward passes
        pass

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        # Apply to the reasoning block
        if hasattr(self.model, 'reasoning') and hasattr(self.model.reasoning, 'block'):
            if hasattr(self.model.reasoning.block, 'gradient_checkpointing_enable'):
                self.model.reasoning.block.gradient_checkpointing_enable()
