# Phase 3.5: Gated Multimodal Unit (v11)

This directory contains the source code for the v11 "Bidirectional Gated Fusion" architecture, designed to bridge Phase 3 (local windowed fusion) and Phase 4 (global temporal aggregation).

## 📂 Structure

- `models.py`: Core architecture implementing `CoAttentionFusion` and `FusionV11Classifier`.
- `dataset.py`: PyTorch `Dataset` for loading normalized WavLM (Audio) and RoBERTa (Text) features.
- `train.py`: Training pipeline with label smoothing, class weighting, and patient-level validation.
- `validate_v11.py`: Comprehensive validation suite for ROC-AUC, F1 threshold tuning, and explainability gating.
- `verify_setup.py`: Local architecture validator for forward-pass sanity checks.

## 🧠 Key Innovations

1. **Gated Multimodal Unit (GMU)**: Learns a dynamic trust gate $z \in [0, 1]$. $z \to 1$ trusts Audio (prosody), $z \to 0$ trusts Text (semantics).
2. **Bidirectional Co-Attention**: Cross-modality attention heads that contextualize audio with text and vice-versa.
3. **Wait-and-Gate Architecture**: Designed to be frozen in Phase 4 to act as a robust feature extractor.

## 🚀 Quick Start (Inference)

```python
from models import FusionV11Classifier
import torch

model = FusionV11Classifier(audio_dim=1024, text_dim=768, shared_dim=256)
model.load_state_dict(torch.load("../../experiments/fusion_v11_handover/best_fusion_v11.pth"))
model.eval()

# Forward pass
logits, gate_vals = model(audio_tensor, text_tensor, scalar_tensor)
```

## 📊 Deployment

Deployment scripts are located in `scripts/phase3_5_v11/`:
- `train_fusion_v11.sb`: Cluster training script.
- `run_validation_v11.sb`: Cluster validation script.
