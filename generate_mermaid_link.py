
import base64
import json

mermaid_code = """
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontFamily': 'Times New Roman', 'fontSize': '14px', 'primaryColor': '#fff', 'edgeLabelBackground':'#fff', 'clusterBkg': '#f9f9f9', 'clusterBorder': '#666' }}}%%
graph TD
    classDef default fill:#fff,stroke:#333,stroke-width:1px;
    classDef cluster fill:#f9f9f9,stroke:#666,stroke-width:1px,rx:5,ry:5;
    
    Input["Input Sequence"] --> L0
    
    subgraph Backbone ["Triple Hybrid + MoE Backbone (4:4:2)"]
        direction TB
        
        subgraph DLM ["Deep Long-term Memory (40%)"]
            direction TB
            L0["Layer 0: Mamba-3 + MoE"] --> L1["Layer 1: Mamba-3 + MoE"]
            L1 --> L2["Layer 2: Mamba-3 + MoE"]
            L2 --> L3["Layer 3: Mamba-3 + MoE"]
        end
        
        subgraph HWM ["High-speed Working Memory (40%)"]
            direction TB
            L3 --> L4["Layer 4: Gated DeltaNet + MoE"]
            L4 --> L5["Layer 5: Gated DeltaNet + MoE"]
            L5 --> L6["Layer 6: Gated DeltaNet + MoE"]
            L6 --> L7["Layer 7: Gated DeltaNet + MoE"]
        end
        
        subgraph FA ["Focused Attention (20%)"]
            direction TB
            L7 --> L8["Layer 8: Gated Attention + MoE"]
            L8 --> L9["Layer 9: Gated Attention + MoE"]
        end
    end
    
    L9 --> Output["Output"]

    subgraph MoE ["Ultra-Sparse MoE Structure"]
        direction TB
        Token["Token Input"] --> RouterNode["Router"]
        RouterNode -->|Top-K| Routed["Routed Experts (10 / 512)"]
        Token --> Shared["Shared Expert (1)"]
        Routed --> AddNode["(+)"]
        Shared --> AddNode
        AddNode --> MoEOut["MoE Output"]
    end
    
    %% Styling connections
    linkStyle default stroke:#333,stroke-width:1px;
"""

def generate_link(code):
    # mermaid.ink expects base64 encoded string of the code
    code_bytes = code.encode('utf-8')
    base64_bytes = base64.b64encode(code_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return f"https://mermaid.ink/img/{base64_string}"

print(generate_link(mermaid_code))
