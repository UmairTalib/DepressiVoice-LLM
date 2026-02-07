# Phase 3.5 Handover: Transition to v11 "Bidirectional Gated Fusion"

This handover bridges the gap between the successful Phase 3 (v10) and the upcoming **Phase 4: The Classifier**, introducing the **v11 Co-Attention & Gated Fusion** architecture.

---

## 📍 1. Infrastructure Overview

### Execution Environment
- **OS**: Mac (Local Development) / Linux (Cluster Execution).
- **Python**: `3.10` with `conda` environment `depressivoice-phase1`.
- **Primary Dependencies**: 
    - `torch >= 2.0` (Core Deep Learning)
    - `transformers` (Embedding extraction)
    - `scikit-learn` (Metrics calculation)

### Remote Cluster Access
- **SSH**: `g755389@omni.zimt.uni-siegen.de`
- **Working Dir**: `~/DepressiVoice-LLM/`
- **Log Dir**: `~/DepressiVoice-LLM/logs/`

---

## 📂 2. Data & Feature Specifications

### Audio Feature Specification (v11 Upgrade)
- **Model**: `microsoft/wavlm-large`
- **Layers Used**: **Layers 6–12** (Weighted Sum).
- **Weighting Strategy**: Exponential bias (`[1, 1, 1, 2, 3, 4, 5]`) towards Layer 12.
- **Rationale**: This configuration focuses strictly on **Paralinguistics & Prosody** (tone, flatness, rhythm). By discarding lower-level phonetics (Layers 1-5) and upper-level semantics, we reduce redundancy with the Text modality and capture the pure "acoustic signature" of depression.

### Text Feature Specification
- **Model**: `Mental-RoBERTa`
- **Output**: `[T, 768]` embeddings representing clinical semantic content.

---

## 🛡️ 3. Risk Mitigation Strategy

To ensure stability and generalizability on the small E-DAIC dataset (approx. 275 sessions), the following safeguards are implemented:

1. **Frozen Backbones**: All massive pre-trained models (`WavLM` and `Mental-RoBERTa`) are **Strictly FROZEN**. No fine-tuning is performed on these layers to prevent catastrophic forgetting and overfitting.
2. **High Regularization**: The fusion layer employs a high **Dropout (0.4)** to force the model to learn robust multimodal representations.
3. **Residual Connections**: Included in the fusion module to stabilize gradient flow and prevent the bottlenecking of modality-specific information.

---

## 🧠 4. v11 Architecture: Co-Attention & Gated Fusion

The v11 upgrade introduces **Bidirectional Co-Attention** and a **Gated Multimodal Unit (GMU)**, allowing the model to dynamically trust one modality over the other.

### `models.py` (v11 Implementation)

```python
import torch
import torch.nn as nn

class CoAttentionFusion(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=768, shared_dim=256, nhead=8, dropout=0.4):
        super(CoAttentionFusion, self).__init__()
        
        # 1. Dimension Projection
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        
        self.norm_a = nn.LayerNorm(shared_dim)
        self.norm_t = nn.LayerNorm(shared_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Parallel Co-Attention Blocks
        self.audio_query_attn = nn.MultiheadAttention(shared_dim, nhead, dropout=dropout, batch_first=True)
        self.text_query_attn = nn.MultiheadAttention(shared_dim, nhead, dropout=dropout, batch_first=True)
        
        # 3. Gated Multimodal Unit (GMU)
        self.fusion_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid()
        )

    def forward(self, audio, text, audio_mask=None, text_mask=None):
        # --- Projections ---
        a_proj = self.dropout(self.norm_a(self.audio_proj(audio)))
        t_proj = self.dropout(self.norm_t(self.text_proj(text)))
        
        # --- Bidirectional Co-Attention ---
        head_a, _ = self.audio_query_attn(query=a_proj, key=t_proj, value=t_proj, key_padding_mask=text_mask)
        head_b, _ = self.text_query_attn(query=t_proj, key=a_proj, value=a_proj, key_padding_mask=audio_mask)
        
        # --- Gated Fusion (GMU) ---
        gate_input = torch.cat([head_a, head_b], dim=-1)
        z = self.fusion_gate(gate_input) # Learned Gate Value
        
        fused_output = z * head_a + (1 - z) * head_b
        
        # Residual Connections for stability
        fused_output = fused_output + a_proj + t_proj
        
        # Strict Output Masking
        if text_mask is not None:
            out_mask = text_mask.unsqueeze(-1).logical_not().float() 
            fused_output = fused_output * out_mask
            
        return fused_output, z # Return z for explainability/plotting
```

---

## 🕵️ 5. Presentation Intuition: "Explainable Dynamic Trust"

When presenting the v11 upgrade, pivot to **Explainability** using the GMU:

> *"The Gated Multimodal Unit (GMU) isn't just a fusion layer; it's an explainability feature. Because the gate **z** is a learned probability, we can plot it alongside the interview transcript. This allows clinicians to see exactly when the model is trusting the **Audio** (prosody) vs. the **Text** (semantics). This is revolutionary for detecting **'Masked Depression'**, where a patient's words are neutral or positive, but the model—tracking a sob or a flat tone—shifts its trust to the audio stream to make the correct diagnosis."*

---

## 🚀 6. Ready for Phase 4
The output preserves the sequence length (`[Batch, Seq, 256]`), providing the **Phase 4 Classifier** with a detailed,Fused, and Explainable temporal map of the interview.
