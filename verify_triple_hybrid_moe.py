
import torch
import torch.nn as nn
from olmo_core.nn.transformer.triple_hybrid import (
    TripleHybridConfig,
    TripleHybridTransformer,
    LayerType,
    Mamba3BlockMoE,
    GatedDeltaNetBlockMoE,
    GatedAttentionBlock
)
from olmo_core.nn.transformer.moe import MoELayer

def verify_triple_hybrid_moe():
    print("Verifying Triple Hybrid + MoE Architecture...")
    
    # 1. Configure Model
    config = TripleHybridConfig(
        hidden_size=512,
        num_layers=10,
        mamba3_ratio=0.4,
        deltanet_ratio=0.4,
        attention_ratio=0.2,
        # MoE Config
        use_moe=True,
        moe_num_experts=8,
        moe_num_shared_experts=1,
        moe_num_routed_experts=2,
        moe_intermediate_size=1024,
    )
    
    # 2. Build Model
    model = TripleHybridTransformer(config)
    print("Model built successfully.")
    
    # 3. Verify Layer Types and MoE Presence
    layer_types = config.get_layer_types()
    print(f"Layer Types: {[t.value for t in layer_types]}")
    
    expected_counts = {
        LayerType.MAMBA3_MOE: 4,
        LayerType.GATED_DELTANET_MOE: 4,
        LayerType.GATED_ATTENTION_MOE: 2
    }
    
    counts = {t: 0 for t in expected_counts}
    for t in layer_types:
        if t in counts:
            counts[t] += 1
            
    for t, count in expected_counts.items():
        assert counts[t] == count, f"Expected {count} layers of type {t}, got {counts[t]}"
    print("Layer distribution verified.")
    
    # 4. Verify MoE Layer Instantiation
    for idx, layer in enumerate(model.layers):
        layer_type = layer_types[idx]
        if layer_type in [LayerType.MAMBA3_MOE, LayerType.GATED_DELTANET_MOE]:
            assert isinstance(layer.mlp, MoELayer), f"Layer {idx} ({layer_type}) should have MoELayer MLP"
        elif layer_type == LayerType.GATED_ATTENTION_MOE:
             assert isinstance(layer.mlp, MoELayer), f"Layer {idx} ({layer_type}) should have MoELayer MLP"
             
    print("MoE layers verified.")
    
    # 5. Run Forward Pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print("Running forward pass...")
    output, _ = model(hidden_states)
    
    assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape} vs {hidden_states.shape}"
    print("Forward pass successful.")
    
    print("\nTriple Hybrid + MoE Verification Passed!")

if __name__ == "__main__":
    try:
        verify_triple_hybrid_moe()
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()
