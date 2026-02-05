# Phase 3 Data Handover: Multimodal Modeling

This document serves as the official handover for Phase 3, detailing the location and structure of all verified features and ground-truth labels on the **Omni Zimt Cluster**.

## 1. Cluster Data Root
All paths are relative to:
`/home/g755389/DepressiVoice-LLM/`

---

## 2. Feature Directory Map

### 🔊 Audio Features (WavLM-Large)
*High-resolution acoustic representations capturing prosodic and spectral cues.*
- **Path:** `data/03_features/audio_embeddings_clean/`
- **Format:** `{ID}_features_norm.pt` (torch tensor)
- **Shape:** `[T, 1024]` (T = sequence length)
- **Normalizaton:** Z-score normalized per-session.

### 📝 Text Features (MentalRoBERTa)
*Linguistic biomarkers extracted via domain-specific transformer encoding.*

#### A. Pooled Features (Sliding Window)
*Single vector per session using 100% of transcript context.*
- **Path:** `data/03_features/text_embeddings_v2/`
- **Format:** `{ID}_text.pt`
- **Shape:** `[1, 768]`

#### B. Sequential Features (Segment-level)
*Sequence of vectors, one for every timestamped segment in Phase 1 output.*
- **Path:** `data/03_features/text_embeddings_seq/`
- **Format:** `{ID}_text_seq.pt`
- **Shape:** `[S, 768]` (S = number of segments)

---

## 3. Labels and Metadata
This file is the single source of truth for training targets and dataset splits.

- **Path:** `data/01_raw/detailed_lables.csv`
- **ID Column:** `Participant` (maps to the `{ID}` in filenames above).
- **Target Label:** `Depression_label` (Binary: 0=Healthy, 1=Depressed).
- **Split Column:** `split` (Values: `train`, `dev`, `test`).

---

## 4. How to Access and Use
To start training your Multimodal DataLoader in Phase 3:

### Option A: Local Download (via SCP)
```bash
# From your local terminal
scp -r g755389@omni.zimt.uni-siegen.de:/home/g755389/DepressiVoice-LLM/data/03_features/ ./data/
```

### Option B: Cluster Training (SLURM)
Ensure your Python scripts reference the absolute paths:
```python
AUDIO_PATH = "/home/g755389/DepressiVoice-LLM/data/03_features/audio_embeddings_clean/"
TEXT_PATH  = "/home/g755389/DepressiVoice-LLM/data/03_features/text_embeddings_v2/"
LABEL_CSV  = "/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv"
```

## 5. Verification Status
| Component | Sessions | Integrity |
|-----------|----------|-----------|
| Audio     | 275      | Verified (Mean 0, Std 1) |
| Text      | 275      | Verified (60.36% Linear Probe) |
| Labels    | 275      | Ground Truth synced |
