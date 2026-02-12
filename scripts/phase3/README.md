# Phase 3 Training Scripts

## Final Scripts (Use These)

### `train_final.sb`
**Purpose**: Train v11-PRO model (Gated Multimodal Unit + Multi-Task Learning)

**Usage**:
```bash
sbatch scripts/phase3/train_final.sb
```

**Configuration**:
- Model: v11-PRO with GMU and PHQ-8 symptom heads
- Loss: BCE + Orthogonal (λ=0.1) + MSE (λ=0.5)
- Training: 100 epochs with early stopping
- GPU: 1x A100 (40GB)
- Time: ~1 hour

**Outputs**:
- Model checkpoint: `experiments/v11_PRO/best_model_max.pt`
- Training logs: `experiments/v11_PRO/training.log`

---

### `validate.sb`
**Purpose**: Run 4-pillar validation framework

**Usage**:
```bash
sbatch scripts/phase3/validate.sb
```

**Validation Pillars**:
1. Statistical Robustness (AUC, F1, Precision, Recall)
2. Clinical Integrity (Text-only ablation test)
3. Clinical Reliability (Calibration + symptom correlations)
4. Multimodal Synergy (Full ablation study)

**Outputs**:
- Validation report: `experiments/phase3_final_validation/validation_report.json`
- 4 validation plots (PNG)

---

## Historical Scripts

All experimental iterations are archived in `scripts/phase3_legacy/`:
- `v10_experiments/` - Baseline concatenation experiments (BATTLE, REVISED)
- `v11_experiments/` - GMU development iterations (GOLDEN, MAX, ORIGINAL)

See `scripts/phase3_legacy/README.md` for details.
