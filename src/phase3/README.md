# Phase 3: Gated Multimodal Fusion (v11-PRO)

## Overview

This directory contains the **final Phase 3 implementation** - a Gated Multimodal Unit (GMU) with Multi-Task Learning for window-level depression detection.

## Architecture

**Model**: v11-PRO (Phenotype-Anchored Gated Fusion)
- Cross-modal attention between audio and text
- Dynamic gating mechanism for adaptive fusion
- Multi-task learning on 8 PHQ-8 symptoms
- Orthogonal loss for modality diversity

**Performance** (DAIC-WOZ Dev Set):
- Window-level AUC: 0.659
- F1-Score: 0.40
- Audio-only AUC: 0.633
- Text-only AUC: 0.503 (proves no interviewer bias)

## Files

| File | Description |
|------|-------------|
| `models.py` | v11-PRO architecture with GMU and MTL heads |
| `dataset.py` | Multimodal dataset loader (audio + text + scalars) |
| `train.py` | Training script with 3-objective loss function |
| `evaluate.py` | Evaluation utilities for window-level predictions |

## Usage

### Training
```bash
# On cluster (recommended)
sbatch scripts/phase3/train_final.sb

# Local (for debugging)
python src/phase3/train.py --config configs/phase3_final.yaml
```

### Evaluation
```bash
python src/phase3/evaluate.py \
  --model experiments/v11_PRO/best_model_max.pt \
  --split dev
```

## Documentation

**Full Technical Report**: [PHASE3_MULTIMODAL_FUSION_REPORT.pdf](../../docs/reports/PHASE3_MULTIMODAL_FUSION_REPORT.pdf)

**Key Sections**:
- Architecture evolution (v10 → v11 → v11-PRO)
- 4-pillar validation framework
- Comprehensive ablation studies
- Clinical integrity tests

## Model Evolution

For historical Phase 3 iterations, see:
- `src/phase3_legacy/v9_coattention/` - Early attention experiments
- `src/phase3_legacy/v10_baseline/` - Concatenation baseline
- `src/phase3_legacy/v11_iterations/` - GMU development iterations
