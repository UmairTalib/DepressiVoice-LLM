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

| Phase | Status | Description | Report |
|-------|--------|-------------|--------|
| **Phase 1** | ✅ Complete | Speaker diarization & transcription | [README_PHASE1.md](docs/README_PHASE1.md) |
| **Phase 2** | ✅ Complete | WavLM-Large (Audio) + Mental-RoBERTa (Text) | [README_PHASE2.md](docs/README_PHASE2.md) |
| **Phase 3** | ✅ Complete | Gated Multimodal Fusion (v11-PRO) | [Technical Report](docs/reports/PHASE3_MULTIMODAL_FUSION_REPORT.pdf) |
| **Phase 4** | ✅ Complete | Session-Level Temporal Aggregation (Hardened) | [Technical Report](docs/reports/PHASE4_TECHNICAL_REPORT.pdf) |
| **Phase 5** | ✅ Complete | Clinical Reasoning & Gemini Narratives | [Technical Report](docs/reports/PHASE5_REPORT_FINAL.pdf) |
| **Phase 6** | ✅ Complete | Research Dashboard & Visualization | [src/phase6_dashboard/](src/phase6_dashboard/) |

---

## 📁 Repository Structure

```
DepressiVoice-LLM/
├── src/
│   ├── phase1/                 # Diarization & Transcription ✅
│   ├── phase2_audio/           # Audio feature extraction ✅
│   ├── phase2_text/            # Text embedding extraction ✅
│   ├── phase3/                 # Gated Multimodal Fusion (v11-PRO) ✅
│   ├── phase3_validation/      # 4-pillar validation framework ✅
│   ├── phase4_temporal/        # Session-level Transformer ✅
│   ├── phase5_llm/             # Gemini reasoning & explanations ✅
│   └── phase6_dashboard/       # Streamlit Research Dashboard ✅
│
├── scripts/
│   ├── phase3/                 # Final training scripts ✅
│   └── phase3_legacy/          # Historical experiments (archived)
│
├── docs/
│   ├── reports/
│   │   ├── PHASE3_MULTIMODAL_FUSION_REPORT.pdf  # Phase 3 Final
│   │   ├── Phase4_Technical_Report.pdf
│   │   └── Phase5_Report.pdf
│   └── archive/                # Historical reports
```

---

## 🔬 Phase 1: Speaker Diarization & Transcription (✅ Complete)

**Goal:** Separate patient and interviewer speech with biometric-quality diarization.

- **Technology:** pyannote.audio 3.0 + Whisper Large-v3
- **Innovation:** Patient-specific voiceprint enrollment for zero leakage
- **Performance:**
    - ✅ **Diarization Error Rate (DER):** 4.2%
    - ✅ **Word Error Rate (WER):** ~12%
    - ✅ **100% interviewer removal** (surgical masking)
- **Documentation:** [README_PHASE1.md](docs/README_PHASE1.md), [Phase 1 Technical Report](docs/reports/Phase1_Technical_Report.pdf)

---

## 🔬 Phase 2: Multimodal Feature Extraction (✅ Complete)

**Goal:** Extract high-quality audio and text embeddings from diarized interviews.

- **Audio Pipeline:**
  - Model: WavLM-Large (Microsoft, 1024-D embeddings)
  - Windowing: 8-second windows, 2-second stride
  - Normalization: Session-level Z-score (prevents volume bias)
- **Text Pipeline:**
  - Model: Mental-RoBERTa (768-D, fine-tuned on mental health forums)
  - Extraction: CLS token per window
  - Quality: 100% interviewer speech removed
- **Documentation:** [README_PHASE2.md](docs/README_PHASE2.md), [Phase 2 Technical Report](docs/reports/Phase2_Technical_Report.pdf)

---

## 🔬 Phase 3: Gated Multimodal Fusion (✅ Complete)

**Goal:** Optimally fuse audio prosody and textual semantics at window-level using attention mechanisms.

- **Final Architecture:** v11-PRO with Gated Multimodal Unit (GMU) + Multi-Task Learning
- **Innovation:** 
  - Cross-modal attention with dynamic gating
  - Orthogonal loss for modality diversity (λ=0.1)
  - PHQ-8 symptom anchoring via MTL (λ=0.5)
- **Performance:**
    - ✅ **Window-level AUC:** 0.659
    - ✅ **F1-Score:** 0.40
    - ✅ **Audio-only AUC:** 0.633 (proves vocal biomarkers work)
    - ✅ **Text-only AUC:** 0.503 (proves no interviewer bias)
- **Validation:** Comprehensive 4-pillar framework
  - Pillar 1: Statistical Robustness
  - Pillar 2: Clinical Integrity (no interviewer leakage)
  - Pillar 3: Clinical Reliability (calibration + symptom correlations)
  - Pillar 4: Multimodal Synergy (ablation studies)
- **Documentation:** [Phase 3 Technical Report](docs/reports/PHASE3_MULTIMODAL_FUSION_REPORT.pdf)

---

## 🔬 Phase 4: Session-Level Temporal Aggregation (✅ Complete)

**Goal:** Aggregate window-level predictions to session-level using Transformer-based temporal modeling with scientifically rigorous regularization.

- **Final Architecture:** Cross-Modal Transformer Aggregator (Phase 4.7 "Generalization Hardened")
  - Dual-stream projection with bi-directional co-attention
  - 256-D embeddings, 4 attention heads, 2 layers
  - Label smoothing (ε=0.1) + AdamW (weight_decay=0.01) + dropout=0.5
  - Loss-based checkpointing to prevent overfitting

- **Test Set Performance (N=56, held-out)**:
    - ✅ **AUC:** 0.614 (95% CI: 0.454-0.771)
    - ✅ **F1-Score:** 0.566 (95% CI: 0.385-0.712)
    - ✅ **Recall:** 0.882 (88% sensitivity - clinical screening viability)
    - ✅ **Precision:** 0.417 (42% - requires follow-up workflow)
    - ✅ **Training Loss:** 0.23 (scientifically honest - prevents memorization)

- **Validation Framework:**
  - Bootstrap confidence intervals (1000 iterations)
  - Stratified analysis by gender and depression severity
  - Clinical deployment recommendations (screening protocol)
  
- **Key Innovation:** Regularization hardening prevents overfitting on N=275 samples
  - Phase 4.6 (overfitted): Train Loss=0.005, Val AUC=0.653
  - **Phase 4.7 (hardened)**: Train Loss=0.230, Test AUC=0.614 ✅

- **Documentation:** 
  - [**Phase 4 Technical Report (PDF, 1.1 MB)**](docs/reports/PHASE4_TECHNICAL_REPORT.pdf) - Comprehensive architecture, theory, validation
  - [Phase 4 Gap Analysis](docs/reports/archive/) - Performance analysis vs. predictions
  

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

**Last Updated:** February 13, 2026  
**Status:** Phase 1-6 ✅ SYSTEM COMPLETE
