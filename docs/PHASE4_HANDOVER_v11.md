# Phase 4 Handover: The v11 Bidirectional Gated Classifier

**Status**: ✅ Phase 3.5 COMPLETE (Validated & Ready for Temporal Aggregation)  
**Date**: 2026-02-07  
**Architecture**: `FusionV11` (Co-Attention + Gated Multimodal Unit)

---

## 📍 1. Infrastructure & Environment

### Cluster Access
- **Host**: `g755389@omni.zimt.uni-siegen.de`
- **Working Directory**: `~/DepressiVoice-LLM/`
- **Environment**: `depressivoice-phase1` (Python 3.10)

### Key Directories
| Path | Description |
| :--- | :--- |
| `src/phase3_fusion_v11/` | **Source Code** for the v11 Fusion Model (Gating & Attention). |
| `experiments/fusion_v11_handover/` | **Artifacts**. Contains `best_fusion_v11.pth` (Weights) & Logs. |
| `data/phase2_clean/` | **Frozen Features**. Pre-extracted WavLM & RoBERTa embeddings. |
| `output_plots/` | **Validation Proof**. `gate_histogram.png` & `confusion_matrix.png` (in `docs/reports/v11_validation/`). |

---

## 📂 2. Data Specification (Frozen Features)

> **Critical Design Choice**: To prevent overfitting on the small E-DAIC dataset (275 sessions), the backbones are **FROZEN**. Phase 4 must use these exact pre-calculated embeddings.

| Modality | Shape | Configuration | Description |
| :--- | :--- | :--- | :--- |
| **Audio** | `[B, T, 1024]` | WavLM-Large (Layers 6-12) | Weighted sum biased to Layer 12. Captures Prosody/Tone, filters out Phonetics. |
| **Text** | `[B, T, 768]` | Mental-RoBERTa | Aligned to 8s audio windows. Captures Semantics. |
| **Scalars** | `[B, 3]` | Normalized Stats | Latency, Variance, Articulation Rate (Z-Scored). |

---

## 🧠 3. Validated Model Architecture (v11)

### Core Logic (`src/phase3_fusion_v11/models.py`)
1.  **Bidirectional Co-Attention**:
    *   Audio queries Text (Contextualization).
    *   Text queries Audio (Contextualization).
2.  **Gated Multimodal Unit (GMU)**:
    *   Learns a scalar gate **$z \in [0, 1]$** per time step.
    *   **$z \approx 1$**: Trust Audio (Prosody).
    *   **$z \approx 0$**: Trust Text (Semantics).
    *   *Findings*: The gate is Bimodal, heavily favoring Audio ($z \approx 1$) for depression detection.
3.  **Output**:
    *   Produces a fused vector sequence $H_{fused} \in \mathbb{R}^{T \times 256}$.

### Performance Benchmarks (Dev Set)
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Best AUC** | **0.6279** | Achieved at Epoch 8. |
| **Max F1** | **0.4571** | Significantly higher than default thresholding. |
| **Optimal Threshold** | **0.1681** | The model is conservative; requires calibration. |
| **Recall** | **66.7%** | 8/12 Depressed patients detected. |

---

## 🚀 4. Mission Instructions for Phase 4 Agent

**Objective**: Build the **Temporal Classifier** (LSTM/Transformer).
Phase 3.5 handled the local (8-second) fusion. Phase 4 must now handle the global (session-level) diagnosis.

### Step A: Feature Extraction (The Bridge)
**Do not retrain Phase 3.5**. Load the weights and extract the fused sequences.

```python
from src.phase3_fusion_v11.models import FusionV11Classifier
import torch

# 1. Initialize v11
model = FusionV11Classifier(audio_dim=1024, text_dim=768, shared_dim=256, dropout=0.0)
checkpoint_path = "experiments/fusion_v11_handover/best_fusion_v11.pth"
tensor = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(tensor)
model.eval()

# 2. Extract Fused Sequence (Do not use logits!)
# Input: audio [1, T, 1024], text [1, T, 768] (Scalars not needed for sequence extraction)
with torch.no_grad():
    # We need the intermediate 'fused_embedding' from the model's forward pass
    # NOTE: You may need to call model.fusion() directly or ensure forward returns seq.
    # Current implementation: model.fusion(audio, text) returns (fused_seq, z)
    fused_seq, gate_vals = model.fusion(audio, text)

# Output: fused_seq is [1, T, 256]. Save this to disk for Phase 4.
```

### Step B: Phase 4 Modeling
- **Input**: The `[T, 256]` sequences generated above.
- **Architecture**: Train a lightweight **LSTM** or **Transformer Encoder** on top of these sequences.
- **Goal**: **Aggregation**. The model must decide if a patient with 20% "Depressed Frames" is actually depressed.

---

## ⚠️ 5. Critical Constraints

1.  **Threshold Sensitivity**: The v11 model is "shy" (low raw probabilities). The optimal decision boundary is `0.1681`, not `0.5`. Phase 4 must learn to calibrate this.
2.  **Gate Interpretation**: Use the `gate_vals` ($z$) for the final dashboard.
    *   **High $z$ + High Risk Score** = "Detected via Voice".
    *   **Low $z$ + High Risk Score** = "Detected via Words".
3.  **No Fine-Tuning**: Do not unfreeze the WavLM/RoBERTa backbones in Phase 4. We do not have enough data. Stick to training the temporal aggregator only.

**Handover Complete.**
