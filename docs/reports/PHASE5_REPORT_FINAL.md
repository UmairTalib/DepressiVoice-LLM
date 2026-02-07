# Phase 5 Technical Report: Clinical Transformer Explainer & Gemini Narratives

**Date**: February 7, 2026  
**Status**: **PASS (Batch Generation Complete)**  
**Version**: 1.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Methodology](#methodology)
3. [Hyperparameters and Configuration](#hyperparameters-and-configuration)
4. [Environment and Reproducibility](#environment-and-reproducibility)
5. [Step-by-Step Command Record](#step-by-step-command-record)
6. [Validation Results](#validation-results)
7. [Failure Modes and Known Limitations](#failure-modes-and-known-limitations)
8. [Future Improvements](#future-improvements)

---

## Executive Summary

Phase 5 successfully implemented a "Glass Box" clinical explanation system for the DepressiVoice project. By integrating a Cross-Modal Transformer model with Google's Gemini 3 Pro high-reasoning engine, the system now provides professional-grade diagnostic narratives for 112 patients (Dev/Test splits).

**Key Results:**
- **Model Integration**: Successfully linked attention spikes from the V7 Champion MTL model to clinical grounding data.
- **Narrative Depth**: Utilized `gemini-3-pro-preview` to generate "Senior Psychiatrist" narratives with internal thinking traces.
- **Performance**: Leveraged SLURM Array Jobs to process the entire batch in ~10 minutes.
- **Pass Status**: All target patients have valid, reasoning-anchored clinical reports in JSON format.

---

## Methodology

The Phase 5 pipeline is orchestrated across several specialized scripts:

### 1. `extract_v7_explanations.py`
The core logic script. It performs the following:
- Loads the Champion V7 MTL Transformer model (`best_mtl_transformer.pth`).
- Performs a forward pass on patient features to extract logits, severity scores, and attention weights.
- **Attention Spike Extraction**: Aggregates attention weights across heads and layers to identify the top 3 segments of highest clinical importance.
- **Transcript Grounding**: Maps timestamps to specific segments in the Ellie interview transcripts.
- **Gemini Consulting**: Sends a structured prompt (including AI scores, spikes, and metadata) to `gemini-3-pro-preview` to obtain a reasoning-heavy narrative.

### 2. `batch_summaries.sbatch`
A SLURM submission script used to parallelize generation on the Uni-Siegen Omni cluster.
- Implements a **SLURM Array Job** (1-113 tasks).
- Maps each task index to a Patient ID via `patient_ids.txt`.
- Limits concurrency to 20 tasks to stay within Tier 1 API limits (300 RPM).
- Ensures unbuffered logging for real-time monitoring.

### 3. `batch_generate_v7_summaries.py`
A high-level wrapper used for local verification and final result aggregation.
- Synchronizes individual JSON reports into a master `summary_log.csv`.
- Implements checkpointing to skip already-processed patients.

### 4. `gemini_report.py` (Diagnostic Tool)
Used for early verification of the Gemini API connectivity and testing thinking traces.

---

## Hyperparameters and Configuration

| Parameter | Value | Rationale | Alternatives Considered |
| :--- | :--- | :--- | :--- |
| `PROB_THRESHOLD` | 0.06 | Calibrated based on V7 MTL optimization to maximize F1-score. | 0.5 (Default) - discarded due to class imbalance. |
| `RMSE_CALIBRATION` | 3.0 | Estimated from 0.1422 Normalized RMSE on the validation set. | None - directly derived from performance metrics. |
| `MAX_WORKERS` | 20 (Array) | Optimized for Tier 1 300 RPM limits (approx 5 req/sec). | 10 (Too slow); 50 (Risk of 429 errors). |
| `thinking_config` | `include_thoughts=True` | Required for "Glass Box" transparency in clinical settings. | False - discarded to maintain reasoning traceability. |
| `temperature` | 1.0 | Standard for reasoning models to allow diverse but structured clinical thought. | 0.0 - too deterministic; 0.7 - standard but less "creative" reasoning. |

---

## Environment and Reproducibility

### Cluster Specifications
- **Host**: Uni-Siegen Omni Cluster (`omni.zimt.uni-siegen.de`)
- **Node Type**: Standard Compute Node (via SLURM)
- **OS**: Linux RedHat 8.7 (4.18.0 Kernel)

### Software Environment (Conda)
- **Conda Env**: `depressivoice-phase1`
- **Python**: 3.10.13
- **Primary Packages**:
    - `google-genai`: 0.3.0
    - `torch`: 2.2.0 (CPU Mode)
    - `pandas`: 2.1.4
    - `numpy`: 1.26.2
    - `markdown`: 3.5.1
    - `weasyprint`: 60.1

### Version Tweaks & Patches
- **CPU Force**: The Transformer model is explicitly loaded on the CPU (`DEVICE='cpu'`) in script to avoid hanging on cluster head nodes and to ensure compatibility across diverse compute nodes.
- **Python Unbuffered**: `PYTHONUNBUFFERED=1` is set in all background tasks to ensure logs are flushed immediately to `.out` files.

---

## Step-by-Step Command Record

| Timestamp | Action | Command/Change |
| :--- | :--- | :--- |
| 10:45 | SDK Setup | `pip install google-genai` inside conda env. |
| 11:20 | Single Patient Test | `python src/phase5_llm/extract_v7_explanations.py --patient_ids 300` |
| 11:45 | Parallel Integration | Initial 20-thread `ThreadPoolExecutor` implementation. |
| 12:30 | Sequential Pivot | Switched to sequential loop for login-node stability. |
| 13:05 | SLURM Array Transition | Created `batch_summaries.sbatch` for compute node offloading. |
| 13:15 | Full Batch Submission | `sbatch src/phase5_llm/batch_summaries.sbatch` |
| 13:30 | Result Aggregation | `python src/phase5_llm/batch_generate_v7_summaries.py --splits dev test` |

---

## Validation Results

### 1. Batch Success Metrics
The master log `summary_log.csv` confirms:
- **Total Attempted**: 112 patients
- **Successful Completions**: 112
- **Error Rate**: 0%
- **Avg. Latency**: ~5.2s per patient (Reasoning included)

### 2. Sample Report Highlights (Patient 300)
Patient 300 was identified as **HIGH RISK** (Prob: 0.2022).
- **Top Spike**: Timestamp 1:32 ("I feel like I'm failing").
- **Narrative**: Gemini synthesized the textual "feelings of failure" with the acoustic cues detected by the transformer.

---

## Failure Modes and Known Limitations

1.  **Rate Limiting**: While Tier 1 (300 RPM) is high, excessive parallelization beyond 20 concurrent tasks could lead to `429 Resource Exhausted` errors.
2.  **Context Window**: Extremely long interviews (30+ mins) might truncated if the transcript exceeds token limits, though current interview lengths are safe.
3.  **Non-verbal Cues**: The system relies on textual segments for spikes; "Non-verbal distressed cues" are used as placeholders when text is missing.

---

## Future Improvements

1.  **Alignment Refinement**: Use finer-grained attention rollout (per-head analysis) to distinguish between acoustic and textual importance more clearly.
2.  **Validation Loop**: Implement a "Clinical Feedback" loop where psychiatrists can rate the narrations to further fine-tune the Gemini prompts.
3.  **Real-time Dashboard**: Integrate directly with a streaming interface for live interview monitoring.

---

### End of Report
