# Phase 3 Data Handover: Multimodal Fusion

**Status:** APPROVED & VERIFIED ✅  
**Date:** February 6, 2026  
**Target:** Phase 3 Lead (Multimodal Modeling)

This document provides the definitive map of all assets, scripts, and data required to begin Phase 3. All processing for Audio (Phase 2A) and Text (Phase 2B) is complete.

---

## 1. Cluster Environment & Data Root
**Hostname:** `omni.zimt.uni-siegen.de`  
**User:** `g755389`  
**Project Root:** `~/DepressiVoice-LLM/`

### 📂 Directory Structure (Cluster)
```text
~/DepressiVoice-LLM/
├── data/
│   ├── 01_raw/
│   │   └── detailed_lables.csv          <-- GROUND TRUTH LABELS (Targets)
│   ├── 03_features/
│   │   └── audio_embeddings_clean/      <-- PHASE 2A: Audio Features (WavLM)
│   │       ├── 300_features_norm.pt
│   │       └── ... (275 files)
│   └── phase2_clean/
│       └── text_embeddings_clean/       <-- PHASE 2B: Text Features (MentalRoBERTa)
│           ├── 300_features.pt
│           └── ... (275 files)
├── src/
│   ├── phase2_text/                     <-- Text Extraction Source Code
│   │   ├── extract_embeddings.py        (Main Script)
│   │   ├── validate_alignment.py        (Alignment Check)
│   │   └── validate_text_quality.py     (Quality/NaN Check)
│   └── ...
└── phase1_rerun_outputs/                <-- Intermediate Transcript Data
    ├── session_300/
    │   └── transcript.json
    └── ...
```

---

## 2. Feature Specifications (Inputs X)

### 🔊 Audio Features (Phase 2A)
*   **Path:** `data/03_features/audio_embeddings_clean/{ID}_features_norm.pt`
*   **Format:** PyTorch Tensor (`torch.float32`)
*   **Shape:** `[T, 1024]`
    *   `T`: Number of 8-second windows (variable per session).
    *   `1024`: Feature dimension (WavLM Large).
*   **Normalization:** Z-Score normalized per session (Mean=0, Std=1).

### 📝 Text Features (Phase 2B)
*   **Path:** `data/phase2_clean/text_embeddings_clean/{ID}_features.pt`
*   **Format:** PyTorch Tensor (`torch.float32`)
*   **Shape:** `[T, 768]`
    *   `T`: Exact match to Audio `T` (Verified 100%).
    *   `768`: Feature dimension (Mental-RoBERTa).
*   **Normalization:** Z-Score normalized per session (Mean=0, Std=1).
*   **Content:** 8-second non-overlapping windows of patient speech. Silent windows tokenized as `[SILENCE]`.

---

## 3. Labels & Targets (Y)

**Master File:** `data/01_raw/detailed_lables.csv`

### 📊 Schema & Columns
| Column | Type | Description | Usage in Phase 3 |
|:-------|:-----|:------------|:-----------------|
| **Participant** | Int | Patient ID (300-718) | **Primary Key** (Matches filenames) |
| **Depression_label** | Int | 0 (Healthy) or 1 (Depressed) | **Classification Target** |
| **Depression_severity** | Int | PHQ-8 Score (0-24) | **Regression Target** |
| **split** | String | `train`, `dev`, `test` | **Data Splitting** |
| `gender` | String | `male`, `female` | Bias analysis (Optional) |

### 📈 Distribution (Verified)
*   **Total Sessions:** 275
*   **Train:** 163
*   **Dev:** 56
*   **Test:** 56
*   **Class Balance:** ~24% Positive (Depressed), ~76% Negative (Healthy).

---

## 4. How to Load Data (Python Snippet)

Use this logic to create your Phase 3 `Dataset` class:

```python
import torch
import pandas as pd
from pathlib import Path

AUDIO_DIR = Path("/home/g755389/DepressiVoice-LLM/data/03_features/audio_embeddings_clean")
TEXT_DIR = Path("/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean")
LABELS_CSV = "/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv"

def load_session(session_id):
    # 1. Load Features
    audio = torch.load(AUDIO_DIR / f"{session_id}_features_norm.pt")
    text = torch.load(TEXT_DIR / f"{session_id}_features.pt")
    
    # 2. Verify Alignment (Runtime Check)
    assert audio.shape[0] == text.shape[0], f"Misalignment in {session_id}!"
    
    return audio, text

# Example Usage
df = pd.read_csv(LABELS_CSV)
row = df.iloc[0]
x_audio, x_text = load_session(row['Participant'])
y_target = row['Depression_label']

print(f"Loaded ID {row['Participant']}: Audio={x_audio.shape}, Text={x_text.shape}, Label={y_target}")
```

---

## 5. Validation Artifacts
If you need to re-verify anything, use these scripts:

1.  **Check Alignment:** `src/phase2_text/validate_alignment.py`
    *   *Checks if Audio Window Count == Text Window Count.*
2.  **Check Quality:** `src/phase2_text/validate_text_quality.py`
    *   *Checks for NaNs, Infs, and Vector Diversity.*
3.  **Check Labels:** `scripts/analyze_labels.py`
    *   *Verifies CSV, Splits, and Missing Values.*

---

## 6. Known Edge Cases
*   **Session 309:** Validated and recovered (Metadata was missing, fixed by transcript fallback).
*   **Silence:** Some text windows are `[SILENCE]` embeddings. This is By Design to preserve temporal alignment with silent audio frames.
*   **Single-Window Sessions:** Handled. Normalization uses `unbiased=False` to prevent NaNs.

---

## 7. Next Steps (Phase 3)
1.  **Create Multimodal Dataset Class:** Implement `__getitem__` to load matched Audio+Text pairs.
2.  **Design Fusion Model:** Late Fusion (concat embeddings) or Cross-Attention.
3.  **Training:** Use the `split` column in `detailed_lables.csv` to respect Train/Dev/Test boundaries.
