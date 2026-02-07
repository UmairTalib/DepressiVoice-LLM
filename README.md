# DepressiVoice-LLM

**Depression Detection from Clinical Interview Audio**

Multi-phase pipeline for detecting depression from audio recordings of clinical interviews using multimodal AI (audio biomarkers + text semantics).

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/DepressiVoice-LLM.git
cd DepressiVoice-LLM

# Phase 1: Diarization & Transcription
conda create -n depressivoice-phase1 python=3.10 -y
conda activate depressivoice-phase1
pip install -r requirements_phase1.txt

# See phase-specific guides:
# - Phase 1: docs/README_PHASE1.md
# - Phase 2: docs/README_PHASE2.md
```

---

## 📋 Project Phases

| Phase | Status | Description | Guide |
|-------|--------|-------------|-------|
| **Phase 1** | ✅ Complete | Speaker diarization & transcription | [README_PHASE1.md](docs/README_PHASE1.md) |
| **Phase 2** | ✅ Complete | WavLM-Large (Audio) + Mental-RoBERTa (Text) | [README_PHASE2.md](docs/README_PHASE2.md) |
| **Phase 3** | ✅ Complete | Multimodal fusion (v10 Baseline) | [v9 Report](docs/reports/PHASE3_V9_COATTENTION_REPORT.pdf) |
| **Phase 3.5**| ✅ Complete | Bidirectional Gated Fusion (v11) | [v11 Report](docs/reports/PHASE3_5_TECHNICAL_REPORT_V11.pdf) |
| **Phase 4** | ✅ Complete | MTL Transformer Diagnostic Engine | [Final Report](docs/reports/PHASE4_TECHNICAL_REPORT_V7.pdf) |

---

## 📁 Repository Structure

```
DepressiVoice-LLM/
├── src/
│   ├── phase1/                 # Diarization & Transcription ✅
│   ├── phase2_audio/           # Audio feature extraction ✅
│   ├── phase2_text/            # Text embedding extraction ✅
│   ├── phase3_fusion_v10/      # Legacy baseline fusion (v10)
│   └── phase3_fusion_v11/      # Gated Multimodal Fusion (v11) ✅
│       ├── models.py           # GMU & Co-Attention
│       ├── dataset.py          # Embedding loader
│       └── train.py            # Training pipeline
│
├── scripts/
│   ├── phase1/                 # Phase 1 helpers
│   ├── phase3_5_v11/           # Phase 3.5 (v11) execution suite ✅
│   │   ├── train_fusion_v11.sb # Cluster training
│   │   ├── run_validation_v11.sb # Cluster validation
│   │   └── export_v11_report.py # Technical report generation
│   └── utils/
│       ├── generate_pdf.py     # Report generation
│       └── html_to_pdf.py      # PDF conversion utilities
│
├── docs/
│   ├── reports/
│   │   ├── PHASE3_V9_COATTENTION_REPORT.pdf
│   │   ├── PHASE3_5_TECHNICAL_REPORT_V11.pdf
│   │   └── PHASE4_TECHNICAL_REPORT_V7.pdf  # NEW: Final Diagnostic Report
│   └── plots/                  # Specialized diagnostic visualizations
```

---

## 🧠 Phase 3: Multimodal Fusion (✅ Complete)

**Goal:** Establish a baseline for multimodal integration using co-attention mechanisms.

- **Architecture:** Asymmetric Cross-Attention (v10) & Bi-directional Co-Attention Foundation.
- **Input:** WavLM Features (1024-dim) + Mental-RoBERTa Embeddings (768-dim).
- **Result:** Successfully mapped disparate modalities into a shared 256-dim embedding space.
- **Location:** `src/phase3_fusion_v10/`

---

## ⚡ Phase 3.5: Gated Multimodal Fusion (✅ Complete)

**Goal:** Improve fusion dynamics using a Gated Multimodal Unit (GMU) and Explainability.

- **Model:** `v11` Bidirectional Gated Fusion.
- **Key Features:**
    - **Dynamic Trust:** Learned sigmoid gates $(z)$ allow the model to choose between Audio and Text per window.
    - **Optimization:** Residual connections and strict masking for numerical stability.
    - **Explainability:** Gate values $(z)$ can be plotted to identify "Masked Depression."
- **Results:**
    - ✅ **AUC:** 0.6279
    - ✅ **Recall:** 66.7% (8/12 patients detected)
- **Documentation:** [Phase 3.5 Technical Report](docs/reports/PHASE3_5_TECHNICAL_REPORT_V11.pdf)

---

## 🔬 Phase 4: MTL Diagnostic Engine (✅ Complete)

**Goal:** Aggregate session-level sequences using Cross-Modal Transformers and clinical severity supervision.

- **Architecture:** 4-Head Cross-Modal Transformer + Multi-Task Regression Head.
- **Key Innovation:** **Multi-Task Learning (MTL)**. By predicting PHQ-8 severity simultaneously with binary labels, the model achieved a massive precision breakthrough.
- **Results:**
    - ✅ **AUC:** 0.6326
    - ✅ **Max F1:** 0.4348
    - ✅ **Precision:** 0.4545 (SOTA for session-level)
- **Documentation:** [Phase 4 Technical Report](docs/reports/PHASE4_TECHNICAL_REPORT_V7.pdf)

---

**Last Updated:** February 7, 2026  
**Status:** Phase 1-4 ✅ COMPLETE
