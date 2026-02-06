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
| **Phase 2** | ✅ Complete (Audio) | WavLM-Large feature extraction | [README_PHASE2.md](docs/README_PHASE2.md) |
| **Phase 3** | 📋 Planned | Multimodal fusion & classification | Coming soon |

---

## 📁 Repository Structure

```
DepressiVoice-LLM/
├── src/
│   ├── phase1/                 # Diarization & Transcription ✅
│   │   ├── diarization.py      # Speaker diarization (Pyannote 3.1)
│   │   ├── transcription.py    # Speech-to-text (Whisper Large-v3)
│   │   └── generate_anchor.py  # Biometric anchor generation
│   │
│   ├── phase2_audio/           # Audio feature extraction ✅
│   │   ├── extract_features.py  # Main WavLM extraction
│   │   ├── normalize.py         # Batch normalization
│   │   ├── validate.py          # Feature validation
│   │   ├── audiomodel.py        # WaveLM model wrapper
│   │   └── dataset.py           # Audio dataset loader
│   │
│   ├── phase2_text/            # Text embedding extraction 📋
│   │   └── extract_embeddings.py  # Mental-RoBERTa (planned)
│   │
│   └── phase3_fusion/          # Multimodal fusion 📋
│       ├── fusion_model.py     # Fusion architecture
│       └── train_fusion.py     # Training script
│
├── scripts/
│   ├── phase1/                 # Phase 1 helpers
│   │   ├── validate_outputs.py    # Output validation
│   │   ├── generate_charts.py     # Validation visualizations
│   │   └── rescue_single_speaker.py  # Edge case handling
│   │
│   └── utils/
│       ├── generate_pdf.py     # Report generation
│       └── html_to_pdf.py      # PDF conversion utilities
│
├── cluster_scripts/            # SLURM batch scripts
│   ├── phase1/
│   │   ├── diarization.sbatch  # Batch diarization job
│   │   └── transcription.sbatch  # Batch transcription job
│   │
│   ├── phase2/
│   │   ├── extract_audio_features.sbatch  # WavLM extraction
│   │   └── extract_text_embeddings.sbatch  # Mental-RoBERTa (planned)
│   │
│   └── phase3/
│       └── train_fusion.sbatch  # Fusion model training
│
├── docs/
│   ├── README_PHASE1.md        # Phase 1 guide
│   ├── README_PHASE2.md        # Phase 2 guide  
│   ├── README_CLUSTER.md       # Cluster deployment guide
│   └── reports/
│       ├── Phase1_Technical_Report.pdf  # Phase 1 report
│       ├── Phase2_Technical_Report.pdf  # Phase 2 report ✨
│       └── *.png               # Validation charts
│
└── data/ (gitignored)
    ├── phase1_rerun_outputs/    # Fresh Phase 1 outputs (patient audio + metadata)
    ├── phase2_clean/            # Phase 2 outputs ✨
    │   ├── audio_embeddings_raw/       # Raw WavLM features
    │   └── audio_embeddings_normalized/  # Normalized features
    ├── 03_features/             # Legacy features (archived)
    └── 04_results/              # Phase 3 outputs (models, predictions)
```

---

## 🎯 Phase 1: Speaker Diarization & Transcription (✅ Complete)

**Goal:** Extract clean patient speech and generate word-level transcriptions from clinical interview audio.

**Pipeline:**
```
Master Audio → Diarization → Role Assignment → Patient Audio → Transcription
```

**Key Features:**
- **Pyannote 3.1** speaker diarization with biometric anchor
- **Duration-based patient selection** (handles same-gender speakers)
- **Confidence scoring** (HIGH/MEDIUM/LOW/VERY_LOW)
- **Quality flagging** for low-confidence sessions
- **Faster-Whisper Large-v3** transcription with word timestamps

**Results:**
- ✅ 275/275 sessions processed (100%)
- ✅ 34.1 hours patient audio extracted
- ✅ 74.2% high-confidence assignments
- ✅ 1.71% average contamination
- ✅ Comprehensive validation & documentation

**Documentation:** [docs/README_PHASE1.md](docs/README_PHASE1.md)

---

## 🔊 Phase 2: Feature Extraction (✅ Audio Complete, 📋 Text Planned)

**Goal:** Extract acoustic and semantic features for depression detection modeling.

### Part 1: Audio Features (WaveLM-Large) ✅ Complete

- **Model:** `microsoft/wavlm-large` (315M parameters)
- **Input:** Patient audio files (16kHz mono WAV)
- **Output:** 1024-dim embeddings per 8-second window
- **Location:** `src/phase2_audio/`, `cluster_scripts/phase2/`

**Results:**
- ✅ 275/275 sessions processed
- ✅ 15.1 GB normalized features
- ✅ 89.2% anisotropy reduction via layer normalization
- ✅ Layer-weighted extraction (layers 6-12, 62% weight on layer 12)
- ✅ Comprehensive validation & technical report

### Part 2: Text Embeddings (Mental-RoBERTa) 📋 Planned

- **Model:** `mental/mental-roberta-base`
- **Input:** Patient transcripts (plain text or JSON)
- **Output:** 768-dim embeddings per session
- **Location:** `src/phase2_text/`, `cluster_scripts/phase2/`
- **Status:** Awaiting implementation

**Documentation:** 
- [Phase 2 Guide](docs/README_PHASE2.md)
- [Phase 2 Technical Report](docs/reports/Phase2_Technical_Report.pdf)

---

## 🧠 Phase 3: Multimodal Fusion (📋 Planned)

**Goal:** Fuse audio and text features for depression classification.

- **Input:** WaveLM features (1024-dim) + Mental-RoBERTa embeddings (768-dim)
- **Architecture:** Concatenation / Attention fusion
- **Output:** Binary classification (depressed / not depressed)
- **Location:** `src/phase3_fusion/`, `cluster_scripts/phase3/`

---

## 📊 Dataset

**Source:** DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)

- **Total Sessions:** 275 (sessions 300-718, non-continuous)
- **Total Duration:** 34.1 hours patient speech
- **Average Duration:** 7.4 minutes per session
- **Interview Format:** Clinical depression screening (semi-structured)
- **Speakers:** Patient + Virtual interviewer (ELLIE)

**Phase 1 Outputs:**
- Patient audio files (`.wav`, 16kHz mono)
- Word-level transcriptions (`.json` + `.txt`)
- Confidence scores and quality flags

---

## ⚙️ Environment Setup

### Phase 1 (Diarization & Transcription)

```bash
conda create -n depressivoice-phase1 python=3.10 -y
conda activate depressivoice-phase1

pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install pyannote.audio==3.1.1 faster-whisper==1.0.0 soundfile==0.12.1

export HF_TOKEN="your_huggingface_token"
export TORCHAUDIO_BACKEND="soundfile"
```

### Phase 2 (Feature Extraction)

```bash
conda create -n depressivoice-phase2 python=3.10 -y
conda activate depressivoice-phase2

pip install torch transformers sentence-transformers librosa numpy
```

---

## 📖 Documentation

- **[Phase 1 Guide](docs/README_PHASE1.md):** Setup, cluster deployment, troubleshooting
- **[Phase 2 Guide](docs/README_PHASE2.md):** Feature extraction, paths, filtering strategies
- **[Phase 2 Handover](docs/PHASE2_HANDOVER_PATHS.md):** Complete path reference for WaveLM/Mental-RoBERTa agents
- **[Technical Report](docs/reports/Phase1_Technical_Report.pdf):** Comprehensive Phase 1 report (methodology, validation, results)

---

## 🔬 Methodology Highlights

### Phase 1: Duration-Based Patient Identification

**Challenge:** Female-female interviews have very similar voice embeddings (biometric similarity fails).

**Solution:**
1. **ELLIE** = Speaker with highest similarity to biometric anchor
2. **PATIENT** = Speaker with longest speaking duration (patients naturally speak more)

**Results:** 74.2% high-confidence assignments, robust to same-gender pairs.

### Confidence Scoring

Based on **separation margin** = `sim(ELLIE) - sim(PATIENT)`:

| Level | Margin | Sessions | % |
|-------|--------|----------|---|
| HIGH | ≥ 0.2 | 204 | 74.2% |
| MEDIUM | 0.1-0.2 | 34 | 12.4% |
| LOW | 0.05-0.1 | 14 | 5.1% |
| VERY_LOW | < 0.05 | 22 | 8.0% |

---

## 🚀 Cluster Deployment (SLURM)

**Phase 1 Diarization:**
```bash
sbatch cluster_scripts/phase1/diarization.sbatch
# 275 sessions, ~7.5 hours wall time
```

**Phase 1 Transcription:**
```bash
sbatch cluster_scripts/phase1/transcription.sbatch
# 275 sessions, ~2.5 hours wall time
```

**Phase 2 Feature Extraction:**
```bash
sbatch cluster_scripts/phase2/extract_audio_features.sbatch
sbatch cluster_scripts/phase2/extract_text_embeddings.sbatch
```

---

## 📚 References

- **Pyannote 3.1:** https://github.com/pyannote/pyannote-audio
- **Whisper Large-v3:** https://github.com/openai/whisper
- **Faster-Whisper:** https://github.com/guillaumekln/faster-whisper
- **WaveLM:** https://huggingface.co/microsoft/wavlm-large
- **Mental-RoBERTa:** https://huggingface.co/mental/mental-roberta-base

---

## 📝 Citation

```bibtex
@misc{depressivoice2026,
  title={DepressiVoice: Multimodal Depression Detection from Clinical Interview Audio},
  author={Your Name},
  year={2026},
  note={Speaker diarization, audio biomarkers, and semantic text features for depression classification}
}
```

---

## 📄 License

[Add license information]

---

**Last Updated:** February 6, 2026  
**Status:** Phase 1 Complete ✅, Phase 2 Audio Complete ✅, Phase 2 Text Planned 📋
