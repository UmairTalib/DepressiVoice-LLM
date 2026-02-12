# Phase 3 Archived Iterations

This directory contains historical Phase 3 experiments and iterations. **These are preserved for reference only** - use `scripts/phase3/` for the final implementation.

## v10_experiments/
Concatenation baseline experiments (0.60 AUC)

| Script | Description | Status |
|--------|-------------|--------|
| `train_fusion_v10_BATTLE.sb` | Grid search battle royale | Superseded |
| `train_fusion_v10_REVISED.sb` | First revision | Superseded |
| `train_fusion_v10_REVISED_v2.sb` | Second revision | Superseded |

## v11_experiments/
Gated Multimodal Unit development iterations

| Script | Description | Final AUC | Status |
|--------|-------------|-----------|--------|
| `train_fusion_v11.sb` | Initial GMU | 0.63 | Superseded |
| `train_fusion_v11_GOLDEN.sb` | Pos_weight tuning | 0.64 | Superseded |
| `train_fusion_v11_MAX.sb` | Max pooling variant | 0.67 | Superseded |
| `train_fusion_v11_ORIGINAL.sb` | Original GMU config | 0.63 | Superseded |
| `train_fusion_v11_ORIG_GOLDEN.sb` | Combined config | 0.64 | Superseded |
| `train_fusion_v11_PRO.sb` | **MTL addition** | **0.659** | ✅ **Final → scripts/phase3/train_final.sb** |
| `train_fusion_v11_REVISED.sb` | Revised hyperparams | 0.64 | Superseded |

## Other
- `train_fusion_v9_seeds.sb` - V9 co-attention multi-seed experiments (archived)

## Why These Were Superseded

The v11-PRO model (`train_fusion_v11_PRO.sb`) became the final version because it:
1. Combined all successful innovations (GMU + MTL)
2. Achieved best window-level performance (0.659 AUC)
3. Passed comprehensive 4-pillar validation
4. Demonstrated scientific integrity (text-only = chance)

All other experiments either:
- Achieved lower performance
- Lacked MTL regularization
- Were intermediate development steps

## Usage

These scripts are **reference only**. For active development, use:
```bash
sbatch scripts/phase3/train_final.sb  # The final v11-PRO model
```
