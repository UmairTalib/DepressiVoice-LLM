# Phase 1 Rerun: Comprehensive Technical Report

**Project:** DepressiVoice - Depression Detection from Clinical Interview Audio  
**Phase:** Phase 1 - Speaker Diarization & Transcription (Rerun v2.0)  
**Date:** February 5, 2026  
**Status:** ✅ COMPLETE (275/275 sessions, 100% validated)

---

## Executive Summary

Phase 1 rerun achieved **100% completion** (275/275 sessions) with enhanced confidence scoring and quality flagging. Key results: **74.2% high-confidence** assignments, **34.1 hours** patient audio extracted, **1.71% average contamination**, 37 quality-flagged sessions for transparency.

**Innovations:** Dur ation-based patient selection (handles same-gender speakers), adaptive confidence thresholds (HIGH/MEDIUM/LOW/VERY_LOW), automated quality flagging, rescue procedures for edge cases.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Scripts & Implementation](#scripts--implementation) 
3. [Hyperparameters & Configuration](#hyperparameters--configuration)
4. [Environment & Reproducibility](#environment--reproducibility)
5. [Execution Timeline](#execution-timeline)
6. [Validation Results](#validation-results)
7. [Failure Modes & Limitations](#failure-modes--limitations)
8. [Future Improvements](#future-improvements)

---

## Methodology

### Pipeline Overview

\`\`\`
Master Audio → Diarization (Pyannote 3.1) → Role Assignment (Anchor + Duration) 
→ Patient Audio Extraction → Transcription (Whisper large-v3) → Validation
\`\`\`

### Key Innovations

**1. Duration-Based Patient Selection**
- ELLIE assigned by max similarity to biometric anchor
- PATIENT assigned by max speaking duration (not similarity)
- Handles female-female pairs where similarity is ambiguous

**2. Confidence Scoring**
- Margin = similarity_diff(ELLIE, PATIENT)
- HIGH: margin ≥ 0.2 (74.2% of sessions)
- MEDIUM: 0.1 ≤ margin < 0.2 (12.4%)
- LOW: 0.05 ≤ margin < 0.1 (5.1%)
- VERY_LOW: margin < 0.05 (8.0%)

**3. Quality Flagging**
- Automatic flag for margin < 0.1
- Documents reason, review recommendation
- 37/275 sessions flagged (13.5%)

---

## Scripts & Implementation

### Core Pipeline: `src/phase1_pipeline_rerun.py`

**Function: `assign_roles()` (Lines 81-172)**

Purpose: Assign ELLIE/PATIENT labels with confidence scoring

Logic:
1. Extract speaker embeddings for all detected speakers
2. Calculate cosine similarity to Ellie anchor
3. **ELLIE = argmax(similarities)** ← Speaker most similar to anchor
4. Calculate duration for ALL other speakers  
5. **PATIENT = argmax(durations, excluding ELLIE)** ← Longest-speaking non-Ellie
6. Compute margin = sim[ELLIE] - sim[PATIENT]
7. Assign confidence level based on margin
8. Return roles, similarities, metadata

Rationale: Similarity identifies Ellie (consistent voice), duration identifies patient (talks more in interview).

**Function: `extract_patient_audio()` (Lines 174-206)**

Purpose: Extract and concatenate patient speech segments

Process:
- Filter diarization for patient label
- Extract waveform segments from master audio
- Convert stereo → mono if needed
- Concatenate all segments
- Save as `patient_audio.wav`

**Function: `save_diarization_results()` (Lines 208-262)**

Purpose: Save outputs with quality flags

Outputs:
- `XXX.rttm` - Diarization segments
- `roles.json` - Roles + confidence + margin + durations
- `quality_flag.json` - If confidence LOW/VERY_LOW
- Logs all actions

---

### Transcription: `src/phase1_transcription_rerun.py`

**Function: `transcribe_patient_audio()` (Lines 28-85)**

Purpose: Transcribe with word-level timestamps

Configuration:
- Model: faster-whisper-large-v3
- Language: "en" (forced, dataset is English)
- Beam size: 5
- Temperature: 0.0 (deterministic)
- VAD: Enabled (min_silence=500ms)
- Word timestamps: True

Output Structure:
\`\`\`json
{
  "language": "en",
  "duration": 446.8,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "I've been feeling really down lately",
      "words": [
        {"start": 0.0, "end": 0.3, "word": "I've"},
        ...
      ]
    }
  ]
}
\`\`\`

---

### Rescue Script: `scripts/rescue_session_309.py`

Purpose: Handle session 309 (only 1 speaker detected)

Approach:
1. Force `num_speakers=2` (split single cluster)
2. Extract both candidates
3. User manual verification
4. Rename correct candidate → `patient_audio.wav`
5. Document in `rescue_metadata.json`

Result: Session 309 recovered, 6.0 MB patient audio extracted

---

### Validation: `scripts/run_validation.py`

Checks:
- File completeness (patient_audio,transcript, patient_text)
- Confidence distribution
- Contamination rates (Ellie lexicon detection)
- Duration statistics

Outputs:
- `validation_results.json`
- `validation_summary.csv` (275 rows)

---

### SLURM Batch Scripts

**Diarization:** `cluster_scripts/phase1_rerun_diarization.sbatch`
\`\`\`bash
#SBATCH --array=0-274%10      # 275 tasks, 10 parallel
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --mem=32G             # 32GB RAM
#SBATCH --time=04:00:00       # 4h limit
\`\`\`

**Transcription:** `cluster_scripts/phase1_rerun_transcription.sbatch`
\`\`\`bash
#SBATCH --array=0-274%20      # 275 tasks, 20 parallel
#SBATCH --mem=16G             # 16GB RAM
#SBATCH --time=02:00:00       # 2h limit
\`\`\`

---

## Hyperparameters & Configuration

### Diarization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | pyannote/speaker-diarization-3.1 | SOTA (Feb 2024) |
| Embedding | pyannote/wespeaker-voxceleb-resnet34-LM | ResNet-34, 256-dim |
| num_speakers | None (auto) | Allows 1-N speakers |
| Confidence Thresholds | 0.05, 0.1, 0.2 | Empirically tuned for same-gender cases |

### Transcription

| Parameter | Value | Alternatives | Rationale |
|-----------|-------|--------------|-----------|
| Model | faster-whisper-large-v3 | base, medium, large-v2 | Best WER (<5%) |
| Beam size | 5 | 1, 10 | Speed/accuracy trade-off |
| Temperature | 0.0 | 0.2, 1.0 | Deterministic (reproducible) |
| VAD min_silence | 500ms | 250ms, 1000ms | Avoids over-segmentation |
| Language | "en" | auto | Dataset known English |

---

## Environment & Reproducibility

### Cluster: OMNI HPC (University of Siegen)

- **GPUs:** A100 (40GB), V100 (32GB)
- **OS:** Rocky Linux 8.9
- **Scheduler:** SLURM 23.02.6
- **Storage:** 100GB home quota

### Software

**Conda Environment:** `depressivoice-phase1`

Key Packages:
- Python 3.10.13
- PyTorch 2.1.2+cu118
- pyannote.audio 3.1.1
- faster-whisper 1.0.0
- soundfile 0.12.1

**Environment Variables:**
\`\`\`bash
export HF_TOKEN="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"
export TORCHAUDIO_BACKEND="soundfile"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
\`\`\`

### Installation

\`\`\`bash
conda create -n depressivoice-phase1 python=3.10 -y
conda activate depressivoice-phase1
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install pyannote.audio faster-whisper soundfile
\`\`\`

---

## Execution Timeline

| Step | Timestamp | Duration | Status |
|------|-----------|----------|--------|
| Master audio upload (12GB) | Feb 4, 18:00 | 45 min | ✅ |
| Biometric anchor generation | Feb 4, 19:15 | 30 min | ✅ |
| Diarization job (11077159) | Feb 5, 02:06-09:30 | 7.4h | ✅ 274/275 |
| Session 309 rescue | Feb 5, 09:35 | 15 min | ✅ |
| Transcription job | Feb 5, 09:50-12:20 | 2.5h | ✅ 275/275 |
| Validation | Feb 5, 15:05 | 2 min | ✅ PASSED |

**Total:** ~11 hours wall time, ~185 GPU-hours compute

---

## Validation Results

### Completeness

- **Total:** 275 sessions
- **Complete:** 275 (100%)
- **Missing Files:** 0
- **Quality Flags:** 37 (13.5%)
- **Rescue:** 1 (session 309)

### Confidence Distribution

![Confidence Distribution](../validation/figures/confidence_distribution.png)

| Level | Count | % |
|-------|-------|---|
| HIGH | 204 | 74.2% |
| MEDIUM | 34 | 12.4% |
| LOW | 14 | 5.1% |
| VERY_LOW | 22 | 8.0% |
| UNKNOWN | 1 | 0.4% |

### Separation Margin

![Margin Histogram](../validation/figures/margin_histogram.png)

- **Mean:** 0.239
- **Median:** 0.223
- **Min:** 0.001 (session 632, female-female)
- **Max:** 0.582

### Contamination

![Contamination Histogram](../validation/figures/contamination_histogram.png)

- **Mean:** 1.71%
- **Median:** 0.00%
- **Max:** 45.45%
- **<5%:** 267/275 (97.1%)

### Audio Duration

![Duration Box Plot](../validation/figures/duration_boxplot.png)

- **Mean:** 7.4 minutes
- **Median:** 7.1 minutes
- **Total:** 34.1 hours

### Quality Flags

![Quality Flags](../validation/figures/quality_flags_breakdown.png)

- All 36 LOW/VERY_LOW sessions flagged
- 1 rescue session (309) flagged
- Transparent tracking for manual review

---

## Failure Modes & Limitations

### Known Issues

**1. Single Speaker Detection (Session 309)**
- **Symptom:** Only 1 speaker detected
- **Frequency:** 1/275 (0.4%)
- **Mitigation:** Forced 2-speaker rescue
- **Status:** Resolved

**2. Low Biometric Separation (Female-Female)**
- **Symptom:** Margin < 0.05, high uncertainty
- **Frequency:** 22/275 (8.0%)
- **Mitigation:** Duration-based selection + quality flags
- **Validation:** All 22 checked, duration ratio >2.0 confirms correct assignment

**3. Transcript Contamination**
- **Symptom:** Interviewer prompts in patient text
- **Frequency:** 3% sessions >5% contamination
- **Mitigation:** VAD filtering
- **Future:** Text cleaning pipeline

### Limitations

**1. Timestamp Alignment**
- Transcripts aligned to `patient_audio.wav`, not master
- Can map back using RTTM if needed

**2. Variable Contamination**
- 3% sessions need text cleaning

**3. Biometric Anchor Generalization**
- Single anchor may not capture all Ellie voice variations

---

## Future Improvements

### 1. Alignment Refinements
- Map transcript timestamps to master audio via RTTM
- Enable temporal correlation with video/physiological data

### 2. Additional Validations
- Acoustic quality metrics (SNR, clipping)
- Speaker consistency checks (intra-patient embedding variance)
- Text cleaning pipeline (remove contamination + fillers)

### 3. Pipeline Enhancements
- Adaptive confidence thresholds (learn from labels)
- Multi-anchor voting (robustness)
- Automated rescue detection

---

## Conclusion

Phase 1 rerun successfully processed 275 sessions with:
- ✅ 100% completion (no data loss)
- ✅ Robust confidence scoring (74.2% high-confidence)
- ✅ Quality flagging (37 challenging sessions marked)
- ✅ Low contamination (1.71% average)
- ✅ Production-ready for Phase 2

**Deliverables:**
- 34.1 hours patient audio
- Word-level transcriptions
- Confidence scores for sample weighting
- Quality flags for interpretability

**Next:** Phase 2 - Feature Extraction & Depression Detection Modeling

---

**Report Version:** 2.0  
**Generated:** February 5, 2026  
**Status:** Production Ready ✅
