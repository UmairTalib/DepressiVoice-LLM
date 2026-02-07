# DepressiVoice-LLM

**Depression Detection from Clinical Interview Audio**

Multi-phase pipeline for detecting depression from audio recordings of clinical interviews using multimodal AI (audio biomarkers + text semantics).

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/UmairTalib/DepressiVoice-LLM.git
cd DepressiVoice-LLM

# Phase 1: Diarization & Transcription
conda create -n depressivoice-phase1 python=3.10 -y
conda activate depressivoice-phase1
pip install -r requirements_phase1.txt
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
| **Phase 5** | ✅ Complete | Clinical Reasoning & Gemini Narratives | [Final Report](docs/reports/PHASE5_REPORT_FINAL.pdf) |
| **Phase 6** | ✅ Complete | Research Dashboard & Visualization | [src/phase6_dashboard/](src/phase6_dashboard/) |

---

## 📁 Repository Structure

```
DepressiVoice-LLM/
├── src/
│   ├── phase1/                 # Diarization & Transcription ✅
│   ├── phase2_audio/           # Audio feature extraction ✅
│   ├── phase2_text/            # Text embedding extraction ✅
│   ├── phase3_fusion_v11/      # Gated Multimodal Fusion (v11) ✅
│   ├── phase4_temporal/        # Transformer & MTL models ✅
│   ├── phase5_llm/             # Gemini reasoning & explanations ✅
│   └── phase6_dashboard/       # Streamlit Research Dashboard ✅
│
├── docs/
│   ├── reports/
│   │   ├── PHASE3_5_TECHNICAL_REPORT_V11.pdf
│   │   ├── PHASE4_TECHNICAL_REPORT_V7.pdf
│   │   └── PHASE5_REPORT_FINAL.pdf
│   └── plots/                  # Specialized diagnostic visualizations
```

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

## 🤖 Phase 5: Clinical Reasoning & Gemini Narratives (✅ Complete)

**Goal:** Provide "Glass Box" clinical transparency using High-Reasoning LLMs.

- **Technology:** `gemini-3-pro-preview` with Thinking enabled.
- **Key Features:**
    - **Attention Rollout Calibration**: Extracting spikes from the Transformer to ground AI decisions.
    - **Psychiatrist Persona**: Calibrated clinical narratives synthesizing audio dissonance and textual cues.
- **Documentation:** [Phase 5 Technical Report](docs/reports/PHASE5_REPORT_FINAL.pdf)

---

## 📊 Phase 6: Research Dashboard (✅ Complete)

**Goal:** Provide an interactive interface for exploring model decisions and clinical reports.

- **Technology:** Streamlit.
- **Features**: Interactive patient selection, Gate value $(z)$ visualization, and clinical assessment display.

---

**Last Updated:** February 7, 2026  
**Status:** Phase 1-6 ✅ SYSTEM COMPLETE
