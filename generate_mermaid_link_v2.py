import base64

def generate_mermaid_link(mermaid_code):
    """
    Generates a link to render the Mermaid diagram using mermaid.ink.
    """
    mermaid_code_bytes = mermaid_code.encode('utf-8')
    base64_bytes = base64.b64encode(mermaid_code_bytes)
    base64_string = base64_bytes.decode('ascii')
    return f"https://mermaid.ink/img/{base64_string}"

diagram_code = """graph TD
    %% Global Styles
    classDef norm fill:#f9e4ce,stroke:#dbb084,color:black;
    classDef module fill:#e1e5f2,stroke:#9fa6bf,color:black;
    classDef gate fill:#fff9c4,stroke:#d4c667,color:black;
    classDef op fill:#ffffff,stroke:#333,color:black;
    classDef input fill:#ffffff,stroke:#333,stroke-dasharray: 5 5,color:black;

    %% ==========================================
    %% 1. Macro Architecture (Left Main Stack)
    %% ==========================================
    subgraph MacroArch [Macro Architecture]
        direction BT
        M_Input((Input))
        
        %% 3x Block (Gated DeltaNet)
        subgraph Block3x [3x Loop Block]
            direction BT
            M_N1[Zero-Centered RMSNorm]:::norm
            M_GDN[Gated DeltaNet]:::gate
            M_Add1((+)):::op
            
            M_N2[Zero-Centered RMSNorm]:::norm
            M_MoE1[Mixture of Experts]:::module
            M_Add2((+)):::op
            
            %% Connections
            M_Input --> M_N1
            M_N1 --> M_GDN
            M_GDN --> M_Add1
            M_Input --> M_Add1
            
            M_Add1 --> M_N2
            M_N2 --> M_MoE1
            M_MoE1 --> M_Add2
            M_Add1 --> M_Add2
        end

        %% 1x Block (Gated Attention)
        subgraph Block1x [1x Block]
            direction BT
            M_N3[Zero-Centered RMSNorm]:::norm
            M_GA[Gated Attention]:::gate
            M_Add3((+)):::op
            
            M_N4[Zero-Centered RMSNorm]:::norm
            M_MoE2[Mixture of Experts]:::module
            M_Add4((+)):::op
            
            %% Connections
            M_Add2 --> M_N3
            M_N3 --> M_GA
            M_GA --> M_Add3
            M_Add2 --> M_Add3
            
            M_Add3 --> M_N4
            M_N4 --> M_MoE2
            M_MoE2 --> M_Add4
            M_Add3 --> M_Add4
        end
        
        M_Add4 --> M_Output((Output))
    end

    %% ==========================================
    %% 2. Gated Attention Detail (Top Right)
    %% ==========================================
    subgraph GADetail [Detail: Gated Attention]
        direction BT
        GA_In[Input]:::input
        
        %% Layers
        GA_L_Q[Linear]:::module
        GA_L_K[Linear]:::module
        GA_L_V[Linear]:::module
        GA_L_Gate[Linear]:::module
        
        GA_In --> GA_L_Q & GA_L_K & GA_L_V & GA_L_Gate
        
        %% Q path
        GA_L_Q --> GA_N_Q[Zero-Centered RMSNorm]:::norm
        GA_N_Q --> GA_Rope_Q[Partial Rope]:::op
        GA_Rope_Q --> GA_q[q]
        
        %% K path
        GA_L_K --> GA_N_K[Zero-Centered RMSNorm]:::norm
        GA_N_K --> GA_Rope_K[Partial Rope]:::op
        GA_Rope_K --> GA_k[k]
        
        %% V path
        GA_L_V --> GA_v[v]
        
        %% Attention
        GA_q & GA_k & GA_v --> GA_SDPA[Scaled Dot Product Attention]:::gate
        
        %% Gate path
        GA_L_Gate --> GA_Sig[Sigmoid]:::op
        
        %% Output Merge
        GA_SDPA --> GA_Mul((x)):::op
        GA_Sig -- Output Gate --> GA_Mul
        
        GA_Mul --> GA_Out[Linear]:::module
    end

    %% ==========================================
    %% 3. Gated DeltaNet Detail (Bottom Right)
    %% ==========================================
    subgraph GDNDetail [Detail: Gated DeltaNet]
        direction BT
        GD_In[Input]:::input
        
        %% Linear Splits
        GD_L_QK[Linear]:::module
        GD_L_V[Linear]:::module
        GD_L_AB[Linear]:::module
        GD_L_Gate[Linear]:::module
        
        GD_In --> GD_L_QK & GD_L_V & GD_L_AB & GD_L_Gate
        
        %% q, k Path
        GD_L_QK --> GD_Conv1[Conv]:::op
        GD_Conv1 --> GD_Sig1[sigma]:::op
        GD_Sig1 --> GD_L2[L2]:::norm
        GD_L2 --> GD_qk[q, k]
        
        %% v Path
        GD_L_V --> GD_Conv2[Conv]:::op
        GD_Conv2 --> GD_Sig2[sigma]:::op
        GD_Sig2 --> GD_v[v]
        
        %% alpha, beta Path
        GD_L_AB --> GD_LinTri[Lin.]:::op
        GD_LinTri --> GD_ab[alpha, beta]
        
        %% Core Rule
        GD_qk & GD_v & GD_ab --> GD_Rule[Gated Delta Rule]:::gate
        
        %% Gate Path
        GD_L_Gate --> GD_SiLU[SiLU]:::op
        
        %% Output Merge
        GD_Rule --> GD_Norm[Zero-Centered RMSNorm]:::norm
        GD_Norm --> GD_Mul((x)):::op
        GD_SiLU -- Output Gate --> GD_Mul
        
        GD_Mul --> GD_Out[Linear]:::module
    end

    %% ==========================================
    %% Relationships
    %% ==========================================
    M_GA -.-> GADetail
    M_GDN -.-> GDNDetail"""

print(generate_mermaid_link(diagram_code))
