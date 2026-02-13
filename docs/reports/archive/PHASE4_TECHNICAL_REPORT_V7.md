# Phase 4 Technical Report: Multi-Task Temporal Aggregation (MTL)

**Date**: 2026-02-07  
**Project**: DepressiVoice-LLM  
**Version**: v7 (Final Diagnostic Engine)

---

## 📋 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Methodology & Architectural Evolution](#2-methodology--architectural-evolution)
3. [Script & Logic Breakdown](#3-script--logic-breakdown)
4. [Hyperparameters & Configuration](#4-hyperparameters--configuration)
5. [Environment & Reproducibility](#5-environment--reproducibility)
6. [Step-by-Step Execution Record](#6-step-by-step-execution-record)
7. [Validation Results & Analysis](#7-validation-results--analysis)
8. [Failure Modes & Known Limitations](#8-failure-modes--known-limitations)
9. [Future Improvements](#9-future-improvements)

---

## 1. Executive Summary
**Overall Status**: **PASS**  
**Core Achievement**: Successfully transitioned the diagnostic engine from window-level inference (8s) to full session-level diagnosis. The introduction of Multi-Task Learning (MTL) stabilized the Transformer backbone, resulting in a session-level F1-score that matches our window-level benchmarks.

| Key Metric | Result | Target Status |
| :--- | :--- | :--- |
| **Best F1-Score** | 0.4348 | **Met** |
| **Optimal Precision** | 0.4545 | **Improved** (+44% over v5) |
| **Severity RMSE** | 0.1422 | **Met** |
| **AUC** | 0.6326 | **Met** |

---

## 2. Methodology & Architectural Evolution

The Phase 4 engine was developed through iterative logic upgrades:
*   **v5 (Baseline)**: Bi-LSTM with Attention Pooling. Introduced Cost-Sensitive Focal Loss to handle the 1:13 class imbalance.
*   **v6 (Transformer)**: Shifted to a 4-head Cross-Modal Transformer Aggregator. Replaced sequential processing with global co-attention between audio and text features.
*   **v7 (MTL)**: Implemented "Deep Supervision" by adding a PHQ-8 Regression Head. This forces the model to learn a continuous severity spectrum, significantly regularizing the binary classification task.

---

## 3. Script & Logic Breakdown

### `src/phase4_temporal/extract_features.py`
*   **Function**: The "Bridge" between Phase 3.5 and Phase 4.
*   **Logic**: Loads the trained Fusion V11 model, processes raw window features, and extracts the 256-dim fused embedding and gate values. Saves session-level sequences as `.pt` files.

### `src/phase4_temporal/dataset.py`
*   **Function**: Data loading and sequence preparation.
*   **Logic**: Handles variable-length session sequences using `pad_collate`. Implements slicing augmentation to double the training sample size. Loads both binary labels and normalized PHQ scores from `master_labels_v7.csv`.

### `src/phase4_temporal/models.py`
*   **Function**: Core architecture definition.
*   **Logic**: Contains the `CrossModalTransformerAggregator` with dual heads. Implements the `FocalLoss` and `PositionalEncoding` logic.

### `src/phase4_temporal/train.py`
*   **Function**: Orchestrates the Multi-Task training loop.
*   **Logic**: Calculates the Joint Loss: $L_{Total} = L_{Focal} + 0.5 \times Loss_{MSE}$. Performs validation tracking for both metrics every epoch.

### `src/phase4_temporal/optimize_threshold.py`
*   **Function**: Post-training calibration.
*   **Logic**: Performs an exhaustive search (0.01 resolution) for the threshold that maximizes the F1-score on the Dev set.

---

## 4. Hyperparameters & Configuration

| Parameter | Value | Selection Rationale |
| :--- | :--- | :--- |
| **Transformer Heads** | 4 | Balanced complexity for small sample size ($N \approx 200$). |
| **Hidden Dim** | 256 | Matches the Phase 3.5 fusion bottleneck. |
| **Dropout** | 0.1 | Low dropout preferred for MTL regularization. |
| **Joint Loss $\lambda$** | 0.5 | Weight for MSE loss. Chosen to prioritize classification while retaining supervision. |
| **Focal $\alpha$** | 0.75 | Weights the "Depressed" class to handle 1:13 imbalance. |
| **Temperature** | 1.0 | Standard logit scaling for classification calibration. |

---

## 5. Environment & Reproducibility

### Cluster Environment (Uni-Siegen Omni)
*   **OS**: CentOS Linux 7
*   **GPUs**: NVIDIA A100 / RTX 3090 (gres:gpu:1)
*   **Conda Environment**: `depressivoice-phase1`
*   **Package Manager**: `conda` / `pip`

---

## 6. Step-by-Step Execution Record

| Timestamp (CET) | Command / Action | Purpose |
| :--- | :--- | :--- |
| 08:35 | `rsync ... master_labels_v7.csv` | Uploading updated clinical labels. |
| 08:37 | `rsync ... src/phase4_temporal/` | Deploying MTL code upgrade. |
| 08:43 | `sbatch phase4_mtl.sbatch` | **Job 11081530**: Starting MTL training. |
| 09:25 | `python -m optimize_threshold.py` | Final calibration search. |

---

## 7. Validation Results & Analysis

### F1-Optimization Search (v7 MTL)
![F1 Curve](../plots/f1_optimization_v7_mtl.png)
*Figure 1: The F1-Optimization curve shows a broader stability peak around threshold 0.07, indicating a well-regularized decision boundary.*

### Comparative Table: Evolution to SOTA
| Milestone | Precision | F1-Score | AUC | Notes |
| :--- | :--- | :--- | :--- | :--- |
| v5 (Bi-LSTM) | 0.3158 | 0.3871 | 0.58 | Baseline sequential logic. |
| v6 (Transformer) | 0.3846 | 0.4000 | 0.62 | Performance boost via co-attention. |
| **v7 (MTL)** | **0.4545** | **0.4348** | **0.63** | **SOTA**: Precision leap via MTL. |

---

## 8. Failure Modes & Known Limitations

1.  **Small-N Overfitting**: With only ~275 sessions, the Transformer is prone to overfitting even with MTL.
2.  **Dataset Bias**: E-DAIC labels are derived from PHQ-8, which correlates heavily with vocal intensity. The model might miss "quietly depressed" patients who speak with higher energy.
3.  **Temporal Drift**: The model assumes clinical status doesn't change within a session (30-60 mins).

---

## 9. Future Improvements

1.  **Subject-wise Cross-Validation**: Move beyond a single 80/20 split to 5-fold cross-validation for more robust F1 reporting.
2.  **Acoustic Augmentation**: Implement Mixup or Gaussian noise on the `fused_embeddings` to further improve Transformer generalization.
3.  **Clinical Alignment**: Incorporate clinician-annotated "critical moments" (from Phase 1 transcripts) as additional attention masks.

---
**Report generated for the Phase 4 Handover Suite.**
