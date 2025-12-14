
import torch
import torch.nn as nn
from olmo_core.nn.transformer.triple_hybrid import GatedAttention, GatedAttentionConfig

def verify_gated_attention():
    config = GatedAttentionConfig(
        hidden_size=128,
        num_heads=4,
        head_dim=32,
        use_rotary=True,
        rope_percentage=0.5,
        use_gate=True
    )
    model = GatedAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        use_rotary=config.use_rotary,
        rope_percentage=config.rope_percentage,
        use_gate=config.use_gate
    )

    print("Verifying Gated Attention Implementation...")

    # 1. Check Q/K Norm
    if hasattr(model, 'q_norm') and isinstance(model.q_norm, nn.Module):
        print("[PASS] Q Norm exists")
    else:
        print("[FAIL] Q Norm missing")
    
    if hasattr(model, 'k_norm') and isinstance(model.k_norm, nn.Module):
        print("[PASS] K Norm exists")
    else:
        print("[FAIL] K Norm missing")

    # 2. Check Partial RoPE
    if hasattr(model, 'rotary_dim'):
        expected_dim = int(config.head_dim * config.rope_percentage)
        if model.rotary_dim == expected_dim:
            print(f"[PASS] Rotary dim is {model.rotary_dim} (Expected: {expected_dim})")
        else:
            print(f"[FAIL] Rotary dim is {model.rotary_dim} (Expected: {expected_dim})")
    else:
        print("[FAIL] Rotary dim missing")

    # 3. Check Sigmoid Gate (Functional check)
    # We can't easily check the activation function instance since it's functional, 
    # but we can check if the code runs without error and maybe inspect source if needed.
    # Here we just run a forward pass to ensure shapes and logic hold.
    x = torch.randn(1, 10, 128)
    try:
        out, _ = model(x)
        print("[PASS] Forward pass successful")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")

if __name__ == "__main__":
    verify_gated_attention()
