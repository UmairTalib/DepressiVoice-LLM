---
title: "Phase 2B: Mental-RoBERTa Text Embedding Extraction - Technical Report"
author: "DepressiVoice-LLM Project"
date: "February 6, 2026"
geometry: margin=1in
fontsize: 11pt
documentclass: article
header-includes:
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{booktabs}
  - \usepackage{hyperref}
  - \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
---

\newpage
\tableofcontents
\newpage

# Executive Summary

We have successfully completed **Phase 2B**, the extraction of semantic text embeddings for the DepressiVoice-LLM dataset. This phase focused on processing clinical interview transcripts to generating high-dimensional vector representations aligned with the audio features from Phase 2A.

**Key Achievements:**
*   **100% Completion:** All **275 sessions** (IDs 300-718) have been processed.
*   **Recovery of Session 309:** Successfully recovered the previously missing session by implementing a robust transcript-only fallback strategy.
*   **Perfect Alignment:** Validated that every text embedding file has the exact same number of temporal windows as its corresponding audio feature file.
*   **Normalization:** Applied per-session Z-score normalization (`Mean=0, Std=1`) to match the statistical properties of the audio features.

**Final Status:** ✅ **PASS** (Ready for Phase 3 Fusion)

\newpage

# Methodology

The objective was to extract linguistic biomarkers from patient speech using a domain-specific Large Language Model (LLM), ensuring temporal alignment with audio features for multimodal analysis.

## 1. Model Selection
We selected **`mental/mental-roberta-base`** (Hugging Face), a RoBERTa model fine-tuned on mental health subreddits. 
*   **Why:** Standard BERT/RoBERTa models are trained on general text (Wikipedia/Books). `mental-roberta` is adapted to the lexicon of depression, anxiety, and mental health conditions, making it superior for capturing clinical nuances.
*   **Architecture:** Transformer-based, 768-dimensional hidden state.

## 2. Text Processing Pipeline

### A. Data Ingestion
The pipeline reads from `phase1_rerun_outputs/`, specifically:
*   `transcript.json`: Word-level timestamps from WhisperX.
*   `roles.json` (Optional): Speaker role assignments.
*   `*.rttm` (Optional): Diarization timestamps.

### B. Patient Speech Isolation (Robust Strategy)
Initially, the pipeline filtered strictly based on `RTTM` timestamps. However, this caused data loss for **Session 309** (missing metadata). 
**Refined Approach:** 
*   We verified that the input audio/transcripts were *already* pre-filtered for the patient in Phase 1.
*   We implemented a `--skip_filtering` flag to bypass redundant RTTM checks, ensuring **100% of the patient's spoken words** are included, even if diarization metadata is imperfect.

### C. Windowing Strategy (Alignment Core)
To fuse with Audio (WaveLM), text must be segmented identically:
*   **Window Size:** 8.0 seconds
*   **Stride:** 8.0 seconds (Non-overlapping)
*   **Logic:** All words starting within `[t, t+8)` are concatenated into a single string.
*   **Silence Handling:** Windows with no speech are tokenized as `[SILENCE]`.

### D. Feature Extraction
*   **Input:** Concatenated text string for the window.
*   **Forward Pass:** `model.encode(text)`
*   **Output:** 768-dimensional vector (CLS token or mean pooling).

### E. Normalization
Raw embeddings have varying scales. To assist the Phase 3 Fusion model (which receives Z-score normalized Audio):
*   **Method:** Per-session Z-score.
*   **Formula:** $x_{norm} = \frac{x - \mu}{\sigma}$
*   **Result:** Features centered at 0 with unit variance.

\newpage

# Hyperparameters & Configuration

| Parameter | Value | Reasoning |
|:----------|:------|:----------|
| **Model** | `mental/mental-roberta-base` | Domain-specific efficacy for depression detection. |
| **Window Size** | 8.0 seconds | Matches WaveLM audio feature window. |
| **Stride** | 8.0 seconds | Non-overlapping to prevent data leakage between frames. |
| **Max Seq Length** | 512 tokens | Standard RoBERTa limit; 8s of speech rarely exceeds this. |
| **Normalization** | Z-Score (Per Session) | Essential for convergence in multimodal fusion. |
| **Pooling** | Mean Pooling | Captures global semantics of the window effectively. |

**Alternatives Considered:**
*   *Global Normalization:* Rejected. Speaker volumes and verbosity vary too much; per-session is more robust.
*   *Sliding Window (Overlap):* Rejected. Increases data size 2x without significant gain; strict alignment preferred.

\newpage

# Environment & Reproducibility

## Cluster Environment
*   **System:** OMNI HPC (Uni-Siegen)
*   **OS:** Rocky Linux 8.x
*   **Job Scheduler:** SLURM
*   **Hardware:** CPU Nodes (initial tests), GPU Nodes (A100/V100 for final run).

## Software Stack
*   **Python:** 3.11.2
*   **Key Libraries:**
    *   `sentence-transformers==2.2.2`
    *   `torch==2.1.2`
    *   `librosa==0.10.1`
    *   `numpy`

## Deployment Command
**Script:** `cluster_scripts/phase2/extract_text_embeddings.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=MentalRoBERTa_V2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-274%20

python src/phase2_text/extract_embeddings.py \
    --session_id $SESSION_ID \
    --input_dir phase1_rerun_outputs \
    --output_dir data/phase2_clean/text_embeddings_clean \
    --skip_filtering \
    --verbose
```

\newpage

# Validation Results

## 1. Alignment Verification
We ran `src/phase2_text/validate_alignment.py` across all 275 sessions.

| Metric | Result |
|:-------|:-------|
| **Total Sessions** | 275 |
| **Aligned (Text windows == Audio windows)** | **275 (100%)** |
| **Mismatched** | 0 |
| **Missing** | 0 |

## 2. Recovery of Session 309
*   **Issue:** Missing `roles.json` and `*.rttm` in Phase 1 output.
*   **Fix:** `extract_embeddings.py` updated to support optional metadata.
*   **Result:** Session 309 successfully processed (24 windows), matching audio.

## 3. Data Integrity
*   **Shape:** `[N_windows, 768]`
*   **Dtype:** `torch.float32` (saved as `.pt`)
*   **Values:** Normalized (Mean $\approx$ 0.0, Std $\approx$ 1.0).

## 4. Label Verification
We verified the existence of ground-truth labels for all processed sessions.
*   **Source:** `data/01_raw/detailed_lables.csv`
*   **Coverage:** 275/275 sessions matched.
*   **Splits:** Train (163), Dev (56), Test (56).
*   **Target:** `Depression_label` (Binary) and `Depression_severity` (Score) are complete.

\newpage

# Known Limitations

1.  **Silence Tokens:** 
    8-second windows sometimes contain no speech (only silence). These are embedded as `[SILENCE]`. The fusion model must learn to ignore these.
    
2.  **Diarization Residuals:**
    Although we use patient-only audio, small snippets of the interviewer (Ellie) might remain if Phase 1 diarization was imperfect. The `mental-roberta` model might embed these "interviewer" phrases ("How are you?"), introducing slight noise.

# Future Improvements

1.  **End-to-End Fine-Tuning:**
    In Phase 3, we can unfreeze the last layer of Mental-RoBERTa and fine-tune it jointly with the fusion model for better task adaptation.

2.  **Dynamic Windowing:**
    Instead of fixed 8s windows, align text based on natural sentence boundaries, then pool audio to match. This requires complex re-alignment of the Audio pipeline but could capture more coherent semantic units.

\newpage

# Conclusion

Phase 2B is **complete**. The text features are robust, normalized, and perfectly aligned with the audio features. The dataset is now fully prepared for **Phase 3: Multimodal Fusion**.

**Artifacts Location:**
*   **Script:** `src/phase2_text/extract_embeddings.py`
*   **Data:** `data/phase2_clean/text_embeddings_clean/{ID}_features.pt`

**Report Generated:** February 6, 2026
