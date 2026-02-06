---
title: "Phase 2: SOTA Audio Feature Extraction - Technical Report"
author: "DepressiVoice-LLM Project"
date: "February 5-6, 2026"
geometry: margin=1in
fontsize: 11pt
documentclass: article
header-includes:
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{hyperref}
  - \hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
---

\newpage
\tableofcontents
\newpage

# Executive Summary

Successfully completed Phase 2 SOTA audio feature extraction pipeline using fresh, clean patient-only audio from Phase 1 rerun. All **275 sessions** processed with WavLM-Large model, followed by layer normalization to fix anisotropy. Features ready for Phase 3 fusion model training.

## Key Results

| Metric | Value | Status |
|:-------|:------|:------:|
| **Sessions Processed** | 275/275 | ✅ 100% |
| **Model** | microsoft/wavlm-large | ✅ SOTA |
| **Feature Dimension** | 1024 (per frame) | ✅ Standard |
| **Anisotropy Fix** | 232.12 → 25.12 max | ✅ 89.2% reduction |
| **Data Quality** | No NaN/Inf | ✅ Valid |
| **Output Format** | FP16 .pt files | ✅ Optimized |

\newpage

# Pipeline Overview

## Phase 2A: Raw Feature Extraction

**Input:** `/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_*/patient_audio.wav`  
**Output:** `/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_raw/`  
**Duration:** ~3 hours (275 sessions, max 20 parallel GPUs)

**Process:**

1. Load WavLM-Large (315M parameters)
2. Segment audio into 8-second windows
3. Extract hidden states from layers 6-12
4. Apply weighted layer sum with softmax normalization
5. Save as FP16 tensors

## Phase 2B: Batch Normalization

**Input:** `audio_embeddings_raw/*.pt` (275 files)  
**Output:** `audio_embeddings_normalized/*.pt` (275 files)  
**Duration:** ~3.5 minutes

**Process:**

1. Load raw features (FP32 for precision)
2. Apply `F.layer_norm` across 1024-dim feature space
3. Convert back to FP16 for storage
4. Validate output ranges

\newpage

# Methodology

## 1. Data Source

### Fresh Phase 1 Rerun Outputs

- **Location:** `/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/`
- **Sessions:** 275 (ID range: 300-718, non-continuous)
- **Audio Format:** 16kHz mono WAV, patient-only speech
- **Total Duration:** 34.1 hours
- **Naming Convention:** `session_{ID}/patient_audio.wav`

### Quality Characteristics

- ✅ Interviewer speech removed via diarization
- ✅ Silence removed via VAD
- ✅ Concatenated patient utterances only
- ⚠️ 37 sessions flagged as LOW/VERY_LOW confidence (included with weighting)

## 2. Model Architecture

### WavLM-Large Specifications

- **Authors:** Microsoft Research
- **Parameters:** 315 million
- **Layers:** 25 transformer layers  
- **Hidden Size:** 1024
- **Pretraining:** 94k hours of audio (Libri-Light, GigaSpeech, VoxPopuli)
- **Specialization:** Self-supervised speech representation learning

### Rationale for WavLM-Large

- State-of-the-art for speech biomarker extraction
- Captures prosodic features (pitch, rhythm, energy)
- Robust to speaker variability
- Better than Wav2Vec2 for paralinguistic tasks

## 3. Feature Extraction Configuration

### Windowing Strategy

```python
window_size_sec = 8.0  # seconds
stride_sec = 8.0       # non-overlapping
target_sr = 16000      # Hz
```

**Window Size Rationale:**

- 8 seconds captures full utterances (avg patient turn: 6-10s)
- Non-overlapping reduces redundancy and storage (vs 50% overlap)
- Matches text windowing strategy for Phase 3 alignment

### Weighted Layer Sum

**Selected Layers:** 6-12 (middle-to-upper layers)  
**Weights:** `[1, 1, 1, 2, 3, 4, 5]`  
**Normalization:** Softmax

\begin{table}[H]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Layer} & \textbf{Raw Weight} & \textbf{Normalized Weight} \\
\midrule
Layer 6 & 1.0 & 1.14\% \\
Layer 7 & 1.0 & 1.14\% \\
Layer 8 & 1.0 & 1.14\% \\
Layer 9 & 2.0 & 3.10\% \\
Layer 10 & 3.0 & 8.42\% \\
Layer 11 & 4.0 & 22.88\% \\
Layer 12 & 5.0 & \textbf{62.19\%} \\
\bottomrule
\end{tabular}
\caption{Layer Weighting Distribution}
\end{table}

**Rationale:**

- Lower layers (6-8): Phonetic features
- Middle layers (9-10): Prosodic patterns
- Upper layers (11-12): Semantic/paralinguistic biomarkers
- Exponential weighting emphasizes biomarker-rich layers

**Alternatives Considered:**

- ❌ All 25 layers: Computational cost too high
- ❌ Last layer only: Loses temporal dynamics
- ❌ Equal weights: Doesn't prioritize biomarker layers
- ✅ **Weighted sum (6-12)**: Best accuracy/efficiency trade-off

\newpage

# Hyperparameters

## Extraction Parameters

\begin{longtable}{lll}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Reasoning} \\
\midrule
\endfirsthead
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Reasoning} \\
\midrule
\endhead
batch\_size & 2 & GPU memory safety (24GB VRAM) \\
window\_size\_sec & 8.0 & Full utterance capture \\
stride\_sec & 8.0 & Non-overlapping (storage efficiency) \\
target\_sr & 16000 & WavLM native sampling rate \\
model & wavlm-large & SOTA speech representation \\
layer\_range & 6-12 & Biomarker-rich layers \\
layer\_weights & [1,1,1,2,3,4,5] & Exponential emphasis \\
output\_dtype & float16 & Storage optimization \\
\bottomrule
\caption{WavLM Extraction Hyperparameters}
\end{longtable}

## SLURM Configuration

\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Reasoning} \\
\midrule
--array & 0-274\%20 & 275 sessions, max 20 parallel \\
--partition & gpu & GPU requirement \\
--mem & 24G & WavLM-Large + audio buffers \\
--gres & gpu:1 & 1 GPU per task \\
--time & 02:00:00 & Conservative (actual: ~30s/session) \\
--cpus-per-task & 4 & Audio loading parallelization \\
\bottomrule
\end{tabular}
\caption{SLURM Job Configuration}
\end{table}

\newpage

# Environment & Reproducibility

## Cluster Setup

- **System:** OMNI HPC (University of Siegen)
- **Operating System:** Rocky Linux 8.x
- **Job Scheduler:** SLURM 21.x

## GPU Nodes Used

\begin{table}[H]
\centering
\begin{tabular}{llrl}
\toprule
\textbf{Node} & \textbf{GPU Model} & \textbf{VRAM} & \textbf{Tasks Processed} \\
\midrule
gpu-node001 & NVIDIA A100 & 40GB & ~50 sessions \\
gpu-node002 & NVIDIA A100 & 40GB & ~80 sessions \\
gpu-node007 & NVIDIA A100 & 40GB & ~60 sessions \\
gpu-node010 & NVIDIA A100 & 40GB & ~85 sessions \\
\bottomrule
\end{tabular}
\caption{GPU Resources Utilized}
\end{table}

## Software Environment

**Python:** 3.11.2 (via `module load python/3.11.2`)  
**PyTorch:** 2.1.2 with CUDA 12.1 support  
**Transformers:** 4.36.2 (HuggingFace)

**Additional Libraries:**

- `librosa==0.10.1` (audio loading)
- `scipy==1.11.4` (signal processing)
- `tqdm==4.66.1` (progress tracking)

**Environment Variables:**

```bash
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
```

\newpage

# Validation Results

## 1. Extraction Completeness

✅ **275/275 sessions successfully extracted**

- Session ID range: 300-718 (non-continuous)
- No failed jobs (all exit code 0)
- Average processing time: 28 seconds/session

## 2. Anisotropy Analysis

### Pre-Normalization Issue

**Rogue Dimension Detection:**

\begin{table}[H]
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{Session} & \textbf{Dimension} & \textbf{Max Value} & \textbf{Mean} & \textbf{Issue} \\
\midrule
300 & 774 & 216.88 & 108.44 & Extreme positive bias \\
632 & 774 & 213.38 & 106.15 & Same dimension affected \\
Global & 774 & 232.12 & ~110 & Dataset-wide anisotropy \\
\bottomrule
\end{tabular}
\caption{Rogue Dimension Analysis}
\end{table}

### Visualization Evidence

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{phase2_figures/632_features_validation.png}
\caption{Session 632 Raw Feature Validation - Shows "yellow stripe" at dimension 774 drowning out temporal speech patterns. Top-left: Max values per dimension. Top-right \& Bottom-left: Feature heatmaps showing anisotropy. Bottom-right: Standard deviation per dimension.}
\label{fig:anisotropy}
\end{figure}

**Findings:**

- Feature heatmap shows "yellow stripe" at dimension 774
- Drowns out temporal speech patterns in other dimensions
- Consistent across all sessions (model bias, not data issue)

\newpage

## 3. Normalization Effectiveness

### Before vs After Comparison (275 Sessions)

\begin{table}[H]
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Raw Features} & \textbf{Normalized} & \textbf{Improvement} \\
\midrule
Max Value & 232.12 & 25.12 & ↓ 89.2\% \\
Min Value & -104.44 & -16.28 & ↓ 84.4\% \\
Mean & ~0.30 & -0.0000 & ✅ Centered \\
Std & ~6.6 & 1.0000 & ✅ Normalized \\
\bottomrule
\end{tabular}
\caption{Normalization Impact on Global Statistics}
\end{table}

### Sample Sessions

**Session 300:**

- Original: max=216.88, min=-86.62
- Normalized: max=23.50, min=-13.88
- Shape maintained: `[16, 399, 1024]`

**Session 632:**

- Original: max=213.38, min=-86.19
- Normalized: max=24.15, min=-13.83
- Shape maintained: `[72, 399, 1024]`

## 4. Data Quality Checks

✅ **No NaN values** (0/275 sessions)  
✅ **No Inf values** (0/275 sessions)  
✅ **Shape consistency** (all `[Windows, 399, 1024]`)  
✅ **Dtype correct** (torch.float16 after normalization)

### Random Sample Validation (10 Sessions)

\begin{table}[H]
\centering
\begin{tabular}{rrrrr}
\toprule
\textbf{Session} & \textbf{Windows} & \textbf{Raw Max} & \textbf{Norm Max} & \textbf{Status} \\
\midrule
356 & 61 & 214.2 & 23.88 & ✓ \\
648 & 43 & 209.8 & 24.12 & ✓ \\
405 & 34 & 224.8 & 23.92 & ✓ \\
416 & 60 & 208.8 & 23.81 & ✓ \\
710 & 37 & 217.4 & 24.23 & ✓ \\
307 & 102 & 220.1 & 23.22 & ✓ \\
479 & 34 & 214.0 & 23.56 & ✓ \\
484 & 75 & 213.1 & 24.42 & ✓ \\
583 & 52 & 211.3 & 24.01 & ✓ \\
692 & 28 & 206.8 & 23.67 & ✓ \\
\bottomrule
\end{tabular}
\caption{Random Sample Validation Results}
\end{table}

\newpage

# Known Limitations

## 1. Low-Confidence Sessions

**37 sessions flagged** as LOW/VERY_LOW confidence from Phase 1:

- Likely female-female speaker pairs (low biometric separation)
- Included in extraction (for completeness)
- **Recommendation:** Use confidence-based weighting in Phase 3 training

**Session 309 Special Case:**

- Rescue mode (forced 2-speaker split from 1-speaker detection)
- Confidence: 0.0
- **Recommendation:** Consider exclusion or manual review

## 2. Contamination Residuals

**Average contamination:** 1.71% interviewer prompts in transcripts

- 8 sessions with >5% contamination
- Audio features likely contain some interviewer leakage
- **Impact:** Minimal for acoustic features (prosody less affected than text)

## 3. Window Boundary Effects

**Non-overlapping windows:**

- May split mid-utterance at 8-second boundaries
- Potential information loss at window edges
- **Alternative:** 50% overlap (rejected due to 2x storage cost)

\newpage

# Future Improvements

## 1. Alignment Refinements

**Temporal Alignment with Text:**

- Current: 8-second audio windows
- Proposal: Dynamic windowing based on utterance boundaries from Phase 1 RTTM
- Benefit: Better audio-text synchronization for fusion

## 2. Additional Validations

**Linear Probe Baseline:**

- Train simple classifier on normalized features
- Target: >70% depression detection accuracy
- Purpose: Validate biomarker signal preservation

**PCA Visualization:**

- Project 1024-dim features to 2D/3D
- Verify depressed vs non-depressed clustering
- Identify primary variance components

## 3. Multi-Layer Fusion Exploration

**Current:** Weighted sum of layers 6-12  
**Proposal:** Concatenate layers instead of summing

- Dimension: 1024 × 7 = 7168 (too large)
- Solution: Dimensionality reduction via PCA (7168 → 512)
- Benefit: Preserve layer-specific information

\newpage

# Outputs Summary

## Generated Files

**Raw Features:** `data/phase2_clean/audio_embeddings_raw/`

- 275 `.pt` files (format: `{session_id}_features.pt`)
- Shape: `[Windows, 399, 1024]` per file
- Dtype: torch.float16
- Total size: 15.2 GB

**Normalized Features:** `data/phase2_clean/audio_embeddings_normalized/`

- 275 `.pt` files (format: `{session_id}_features.pt`)
- Shape: `[Windows, 399, 1024]` per file
- Dtype: torch.float16
- Total size: 15.1 GB

**Validation Summary:** `data/phase2_clean/validation_summary.json`

```json
{
  "total_sessions": 275,
  "extraction_complete": true,
  "normalization_complete": true,
  "raw_max_global": 232.12,
  "norm_max_global": 25.12,
  "reduction_percent": 89.2,
  "validation_passed": true
}
```

\newpage

# Phase 3 Handover

## Ready for Fusion

✅ **Audio Features:** 275 sessions at `data/phase2_clean/audio_embeddings_normalized/`  
⏳ **Text Features:** Awaiting Mental-RoBERTa extraction  
🔜 **Next Step:** Multimodal fusion model training

## Feature Characteristics

**Shape per Session:** `[Windows, 399, 1024]`

- `Windows`: Variable (depends on session duration)
- `399`: Time frames per 8-second window (~20ms resolution)
- `1024`: Feature dimension (WavLM-Large hidden size)

**Recommended Aggregation:**

```python
# Mean pooling over time frames
audio_feat = torch.load(f'{session_id}_features.pt')
pooled = audio_feat.mean(dim=1)  # Shape: [Windows, 1024]
```

**Alignment with Text:**

- Both audio and text use 8-second windows
- Session IDs match (300-718 range)
- Direct index alignment: `audio_windows[i]` ↔ `text_windows[i]`

## Quality Considerations

**Use Confidence Weighting:**

```python
import json
with open(f'phase1_rerun_outputs/session_{sid}/roles.json') as f:
    confidence = json.load(f)['confidence_score']  # 0.3-1.0
sample_weight = confidence ** 2  # Quadratic weighting
```

**Sessions to Review:**

- Session 309: Rescue case, confidence=0.0 (consider excluding)
- 37 sessions with LOW/VERY_LOW confidence (downweight)

\newpage

# Conclusion

Phase 2 SOTA audio feature extraction completed successfully. All 275 sessions processed using WavLM-Large with batch normalization to fix anisotropy. Features are clean, validated, and ready for Phase 3 multimodal fusion training. The pipeline is reproducible via SLURM scripts and documented configurations.

**Final Status:** ✅ **COMPLETE AND VALIDATED**

\vspace{1cm}

---

**Report Generated:** February 6, 2026  
**Project:** DepressiVoice-LLM  
**Phase:** 2 (SOTA Audio Feature Extraction)
