import base64

def generate_mermaid_link(mermaid_code):
    """
    Generates a link to render the Mermaid diagram using mermaid.ink.
    """
    mermaid_code_bytes = mermaid_code.encode('utf-8')
    base64_bytes = base64.b64encode(mermaid_code_bytes)
    base64_string = base64_bytes.decode('ascii')
    return f"https://mermaid.ink/img/{base64_string}"

diagrams = {
    "verification": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
graph TD
    classDef default fill:#fff,stroke:#333,stroke-width:1px;
    classDef cluster fill:#f9f9f9,stroke:#666,stroke-width:1px,rx:5,ry:5;
    classDef highlight fill:#e6f3ff,stroke:#333,stroke-width:1px;
    
    subgraph GatedDeltaNet ["Gated DeltaNet Block (Qwen3-Next)"]
        direction TB
        Input1["Input"] --> Norm1["RMSNorm (Zero-Centered)"]
        Norm1 --> Proj1["Linear Projections (Q, K, V, Beta)"]
        
        Proj1 --> Conv["Conv1D (Short Conv)"]
        Conv --> Delta["Delta Rule SSM<br/>(v_t = v_{t-1} + beta * (K*v_{t-1} - V))"]
        
        Proj1 --> Gate1["Output Gate (SiLU)"]
        Delta --> Mult1["(x)"]
        Gate1 --> Mult1
        
        Mult1 --> OutProj1["Output Projection"]
        OutProj1 --> Res1["(+)"]
        Input1 --> Res1
        
        Res1 --> Norm2["RMSNorm"]
        Norm2 --> MoE1["Ultra-Sparse MoE<br/>(Router + Experts)"]
        MoE1 --> Res2["(+)"]
        Res1 --> Res2
        Res2 --> Output1["Output"]
    end
    
    subgraph GatedAttention ["Gated Attention Block (Qwen3-Next)"]
        direction TB
        Input2["Input"] --> Norm3["RMSNorm (Zero-Centered)"]
        Norm3 --> Proj2["Linear Projections (Q, K, V)"]
        
        Proj2 --> RoPE["Partial RoPE<br/>(Applied to Q, K)"]
        RoPE --> Attn["Flash Attention<br/>(Scaled Dot Product)"]
        
        Proj2 --> Gate2["Output Gate (Sigmoid)"]
        Attn --> Mult2["(x)"]
        Gate2 --> Mult2
        
        Mult2 --> OutProj2["Output Projection"]
        OutProj2 --> Res3["(+)"]
        Input2 --> Res3
        
        Res3 --> Norm4["RMSNorm"]
        Norm4 --> MoE2["Ultra-Sparse MoE<br/>(Router + Experts)"]
        MoE2 --> Res4["(+)"]
        Res3 --> Res4
        Res4 --> Output2["Output"]
    end
    
    linkStyle default stroke:#333,stroke-width:1px;""",

    "full_architecture": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
graph TD
    subgraph FullArchitecture ["FULL ARCHITECTURE (Data Flow)"]
        direction TB
        Input[Input Sequence<br/>Question Tokens] --> Backbone
        
        subgraph Backbone ["BACKBONE + ETD (Quad-Hybrid layers organized by ETD)"]
            direction LR
            Encoder[ENCODER 8<br/>ATL|M3|GD<br/>GD|GA|GD<br/>M3|GA| ] --> Think[THINK 24, N iterations<br/>ATL|M3|M3|GD|GD|GA<br/>... x24<br/>+Router +Expert]
            Think --> Decoder[DECODER 8<br/>GA|GA|GD<br/>GA|M3|GA<br/>GD|M3| ]
        end
        
        Backbone --> LaDiR
        
        subgraph LaDiR ["LaDiR: LATENT REASONING (Append Memory Slots)"]
            direction TB
            Hidden[Hidden States] --> Slots[+ MEM1 MEM2 MEM3]
            Slots --> VAE_Enc[VAE Encoder: Memory Slots -> Latent z]
            VAE_Enc --> Flow[Flow Matching: Noise -> Latent z]
            Flow --> Prophet[Prophet: Early Exit via Confidence Gap]
            Prophet --> VAE_Dec[VAE Decoder: Latent z -> Memory Slots]
            VAE_Dec --> OutputHidden[Hidden States with Reasoning]
        end
        
        LaDiR --> Output
        
        subgraph Output ["OUTPUT: Block Diffusion (AR + Diffusion Hybrid)"]
            direction TB
            Block1[Block 1: MASK -> Diffusion -> Tokens]
            Block2[Block 2: MASK -> Diffusion -> Tokens]
            Block1 --> Block2
            Note[Unified Diffusion Paradigm with LaDiR]
        end
        
        Output --> FinalTokens[OUTPUT TOKENS]
    end""",

    "quad_hybrid_memory": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
mindmap
  root((QUAD-HYBRID<br/>MEMORY SYSTEMS))
    ATLAS["1. ATLAS (10%)<br/>Neural Long-term Memory"]
      Capacity["10M+ Tokens"]
      Omega["Omega Rule<br/>(Window Optimization)"]
      Poly["Polynomial Kernel<br/>(O(d^p) Capacity)"]
      DeepMLP["Deep MLP Memory<br/>(2+ Layers)"]
      Muon["Muon Optimizer<br/>(2nd Order)"]
    Mamba3["2. Mamba-3 (35%)<br/>Deep Long-term Memory"]
      CapacityM["~100K Tokens"]
      Complex["Complex SSM"]
      Oscill["Oscillatory Dynamics"]
      LongDep["Long-term Dependency"]
    DeltaNet["3. Gated DeltaNet (35%)<br/>High-speed Working Memory"]
      CapacityD["~10K Tokens"]
      Delta["Delta Rule Update"]
      Assoc["Associative Matrix"]
      RealTime["Real-time Update"]
    Attention["4. Gated Attention (20%)<br/>Focused Attention"]
      CapacityA["Context Window (8K-128K)"]
      Precise["Precise Retrieval"]
      KVCache["KV Cache"]
      CoreInfo["Core Info Access"]""",

    "layer_layout": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
graph TD
    subgraph Layout ["QUAD-HYBRID LAYER LAYOUT (10 Layer Example)"]
        direction TB
        L0[Layer 0: ATLAS 10M+ Tokens]
        
        subgraph DLM [Deep Long-term Memory 35%]
            L1[Layer 1: Mamba-3]
            L2[Layer 2: Mamba-3]
            L3[Layer 3: Mamba-3]
            L4[Layer 4: Mamba-3]
        end
        
        subgraph HWM [High-speed Working Memory 35%]
            L5[Layer 5: Gated DeltaNet]
            L6[Layer 6: Gated DeltaNet]
            L7[Layer 7: Gated DeltaNet]
        end
        
        subgraph FA [Focused Attention 20%]
            L8[Layer 8: Gated Attention]
            L9[Layer 9: Gated Attention]
        end
        
        L0 --> L1
        L4 --> L5
        L7 --> L8
    end""",

    "etd_ladir_relation": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
graph TD
    Quad[Quad-Hybrid: Layer Types<br/>ATLAS / Mamba-3 / DeltaNet / Attention] --> ETD
    
    ETD[ETD: Layer Organization & Adaptive Depth<br/>Dr.LLM: Skip/Execute/Repeat<br/>Think Controller: Adaptive Iteration] --> LaDiR
    
    LaDiR[LaDiR: Adaptive Latent Reasoning<br/>VAE + Diffusion: Latent Thought Generation<br/>Prophet: Adaptive Exit] --> Output
    
    Output[Output: Block Diffusion<br/>Efficient Generation<br/>Parallel Gen + Bi-directional Coherence]""",

    "data_flow": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
graph TD
    Input[Input Tokens] --> ATLAS
    
    subgraph Layer0 [Layer 0]
        ATLAS[ATLAS: 10M+ Token Memory<br/>Omega Rule, Polynomial Kernel, Deep MLP]
    end
    
    ATLAS --> Encoder
    
    subgraph ETD_Structure [ETD Structure]
        Encoder[ENCODER: Quad-Hybrid Layers<br/>Mamba-3 + DeltaNet] --> Think
        
        Think[THINK: Adaptive Iteration<br/>Think Controller<br/>Quad-Hybrid Block + Router + Experts] --> Decoder
        
        Decoder[DECODER: Quad-Hybrid Layers<br/>Attention + DeltaNet]
    end
    
    Decoder --> LaDiR_Mod
    
    subgraph LaDiR_Layer [Sequence End]
        LaDiR_Mod[LaDiR: Latent Reasoning<br/>Hidden States + Memory Slots<br/>VAE -> Diffusion -> Prophet]
    end
    
    LaDiR_Mod --> BlockDiff
    
    subgraph Output_Layer [Output Generation]
        BlockDiff[Block Diffusion<br/>AR between blocks + Diffusion within blocks]
    end
    
    BlockDiff --> OutTokens[Output Tokens]""",

    "summary": """%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
mindmap
  root((ARCHITECTURE<br/>SUMMARY))
    Questions
      LayerType["Quad-Hybrid"]
      Structure["ETD (Adaptive Depth)"]
      Reasoning["LaDiR (Adaptive Reasoning)"]
      Output["Block Diffusion (Efficient Gen)"]
    MemorySystems
      ATLAS["ATLAS (10M+)"]
      Mamba3["Mamba-3 (~100K)"]
      DeltaNet["DeltaNet (~10K)"]
      Attention["Attention (Context)"]
    Files
      Backbone["atlas_memory.py<br/>mamba_memory.py<br/>gated_deltanet.py<br/>triple_hybrid.py"]
      Structure["etd.py"]
      Reasoning["latent_reasoning.py"]
      Output["block_diffusion.py"]"""
}

for name, code in diagrams.items():
    print(f"--- {name} ---")
    print(generate_mermaid_link(code))
    print()
