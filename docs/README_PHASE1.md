# Phase 1: Speaker Diarization & Transcription

**Status:** ✅ COMPLETE (275/275 sessions processed)  
**Outputs:** Patient audio files + word-level transcriptions  
**Next:** [Phase 2 - Feature Extraction](README_PHASE2.md)

---

## Overview

Phase 1 processes clinical interview audio to extract clean patient speech and generate word-level transcriptions. The pipeline uses:

1. **Pyannote 3.1** for speaker diarization (who spoke when)
2. **Biometric anchor** for interviewer identification
3. **Duration-based selection** for patient identification  
4. **Faster-Whisper Large-v3** for transcription

**Key Innovation:** Duration-based patient selection handles same-gender speakers (common in female-female interviews).

---

## Quick Start

```bash
# 1. Setup environment
conda create -n depressivoice-phase1 python=3.10 -y
conda activate depressivoice-phase1
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install pyannote.audio faster-whisper soundfile

# 2. Set environment variables
export HF_TOKEN="your_huggingface_token"
export TORCHAUDIO_BACKEND="soundfile"

# 3. Generate biometric anchor (one-time)
python src/phase1/generate_anchor.py \
    --anchor-sessions "300,301,302" \
    --master-audio-dir /path/to/master_audio \
    --output biometric_anchor.pt

# 4. Run diarization (cluster)
sbatch cluster_scripts/phase1/diarization.sbatch

# 5. Run transcription (cluster)
sbatch cluster_scripts/phase1/transcription.sbatch

# 6. Validate outputs
python scripts/phase1/validate_outputs.py \
    --output-dir /path/to/phase1_outputs
```

---

## Environment Setup

### Requirements

- **Python:** 3.10+
- **CUDA:** 11.8+ (for GPU)
- **Disk:** ~5GB per 275 sessions (outputs)
- **RAM:** 32GB for diarization, 16GB for transcription
- **GPU:** 1 × A100/V100 (recommended)

### Installation

**Option 1: Conda (Recommended)**
```bash
conda create -n depressivoice-phase1 python=3.10 -y
conda activate depressivoice-phase1

# Install PyTorch with CUDA 11.8
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install pyannote.audio==3.1.1 faster-whisper==1.0.0 soundfile==0.12.1
```

**Option 2: From environment.yml** (if available)
```bash
conda env create -f environment.yml
conda activate depressivoice-phase1
```

### Environment Variables

**Required:**
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"  # Get from https://huggingface.co/settings/tokens
export TORCHAUDIO_BACKEND="soundfile"
```

**Optional (for cluster):**
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Better GPU memory management
```

---

## Pipeline Components

### 1. Biometric Anchor Generation

**Purpose:** Create a voice embedding of the interviewer (ELLIE) for identification.

**Script:** `src/phase1/generate_anchor.py`

**Usage:**
```bash
python src/phase1/generate_anchor.py \
    --anchor-sessions "300,301,302" \
    --master-audio-dir /home/g755389/depressivoice_master_audio \
    --output /home/g755389/biometric_anchor.pt
```

**Arguments:**
- `--anchor-sessions`: Comma-separated session IDs (3-5 recommended)
- `--master-audio-dir`: Directory containing master audio WAV files
- `--output`: Output path for anchor embedding (.pt file)

**Output:** `biometric_anchor.pt` (~1KB, PyTorch embedding)

**Run Once:** This only needs to be generated once and reused for all sessions.

---

### 2. Speaker Diarization

**Purpose:** Identify ELLIE and PATIENT speakers, extract patient audio.

**Script:** `src/phase1/diarization.py`

**Usage (Local):**
```bash
python src/phase1/diarization.py \
    --session-id 300 \
    --master-audio /home/g755389/depressivoice_master_audio/300_AUDIO.wav \
    --anchor /home/g755389/biometric_anchor.pt \
    --output-dir /home/g755389/phase1_outputs
```

**Usage (Cluster - Recommended):**
```bash
sbatch cluster_scripts/phase1/diarization.sbatch
```

**Key Parameters:**
- Diarization model: `pyannote/speaker-diarization-3.1`
- Embedding model: `pyannote/wespeaker-voxceleb-resnet34-LM`
- Confidence thresholds: HIGH (≥0.2), MEDIUM (0.1-0.2), LOW (0.05-0.1), VERY_LOW (<0.05)

**Outputs (per session):**
- `patient_audio.wav` - Extracted patient speech (mono, 16kHz)
- `roles.json` - Speaker roles and confidence scores
- `XXX.rttm` - Diarization segments
- `quality_flag.json` - Low-confidence sessions only

---

### 3. Transcription

**Purpose:** Generate word-level transcriptions from patient audio.

**Script:** `src/phase1/transcription.py`

**Usage (Local):**
```bash
python src/phase1/transcription.py \
    --session-id 300 \
    --patient-audio /home/g755389/phase1_outputs/session_300/patient_audio.wav \
    --output-dir /home/g755389/phase1_outputs/session_300
```

**Usage (Cluster - Recommended):**
```bash
sbatch cluster_scripts/phase1/transcription.sbatch
```

**Key Parameters:**
- Model: `faster-whisper-large-v3`
- Language: `"en"` (forced English)
- Beam size: 5
- Temperature: 0.0 (deterministic)
- VAD: Enabled (500ms min silence)

**Outputs (per session):**
- `transcript.json` - Structured JSON with word timestamps
- `patient_text.txt` - Plain text transcription

---

### 4. Validation

**Purpose:** Verify completeness and quality of outputs.

**Script:** `scripts/phase1/validate_outputs.py`

**Usage:**
```bash
python scripts/phase1/validate_outputs.py \
    --output-dir /home/g755389/phase1_outputs \
    --validation-report validation_results.json
```

**Checks:**
- File completeness (patient_audio.wav, transcript.json, patient_text.txt)
- Confidence distribution
- Contamination rates (interviewer prompts in patient text)
- Audio duration statistics

**Outputs:**
- `validation_results.json` - Summary metrics
- `validation_summary.csv` - Per-session metadata

**Generate Charts:**
```bash
python scripts/phase1/generate_charts.py \
    --validation-file validation_results.json \
    --output-dir charts/
```

Generates 5 charts: confidence distribution, margin histogram, contamination histogram, duration box plot, quality flags breakdown.

---

## Cluster Deployment

### Prerequisites

1. **Upload master audio:**
   ```bash
   rsync -avz --progress /path/to/master_audio/ \
       user@cluster:/home/g755389/depressivoice_master_audio/
   ```

2. **Generate biometric anchor** (on cluster):
   ```bash
   python src/phase1/generate_anchor.py \
       --anchor-sessions "300,301,302" \
       --master-audio-dir /home/g755389/depressivoice_master_audio \
       --output /home/g755389/biometric_anchor.pt
   ```

### 1. Diarization Job

**Script:** `cluster_scripts/phase1/diarization.sbatch`

**Submit:**
```bash
cd /home/g755389/DepressiVoice-LLM
sbatch cluster_scripts/phase1/diarization.sbatch
```

**SLURM Configuration:**
```bash
#SBATCH --array=0-274%10      # 275 sessions, 10 parallel
#SBATCH --gres=gpu:1          # 1 GPU per job
#SBATCH --mem=32G             # 32GB RAM
#SBATCH --time=04:00:00       # 4h time limit
```

**Monitor:**
```bash
squeue -u $USER                # Check job status
tail -f slurm-JOBID_TASKID.out  # View logs
```

**Expected Runtime:** ~7.5 hours wall time (with 10 parallel jobs)

---

### 2. Transcription Job

**Script:** `cluster_scripts/phase1/transcription.sbatch`

**Submit:**
```bash
sbatch cluster_scripts/phase1/transcription.sbatch
```

**SLURM Configuration:**
```bash
#SBATCH --array=0-274%20      # 275 sessions, 20 parallel
#SBATCH --gres=gpu:1          # 1 GPU per job
#SBATCH --mem=16G             # 16GB RAM
#SBATCH --time=02:00:00       # 2h time limit
```

**Expected Runtime:** ~2.5 hours wall time (with 20 parallel jobs)

---

## Output Structure

```
phase1_outputs/
├── session_300/
│   ├── patient_audio.wav         ← Audio input for Phase 2 (WaveLM)
│   ├── transcript.json           ← Text input for Phase 2 (Mental-RoBERTa)
│   ├── patient_text.txt          ← Plain text alternative
│   ├── roles.json                ← Confidence scores for weighting
│   ├── 300_AUDIO.rttm            ← Diarization segments
│   └── quality_flag.json         ← Present if LOW/VERY_LOW confidence
├── session_301/
├── ...
├── session_718/
├── validation_results.json       ← Summary statistics
└── validation_summary.csv        ← Per-session metadata
```

---

## Confidence Scoring

Sessions are assigned confidence based on **separation margin** = `sim(ELLIE) - sim(PATIENT)`:

| Level | Margin | Sessions | % | Recommendation |
|-------|--------|----------|---|----------------|
| **HIGH** | ≥ 0.2 | 204 | 74.2% | Use with full weight |
| **MEDIUM** | 0.1-0.2 | 34 | 12.4% | Acceptable quality |
| **LOW** | 0.05-0.1 | 14 | 5.1% | Review or downweight |
| **VERY_LOW** | < 0.05 | 22 | 8.0% | Female-female likely, use duration validation |

**Quality Flags:** All sessions with margin < 0.1 automatically flagged in `quality_flag.json`.

---

## Troubleshooting

### Issue: Single Speaker Detected

**Symptom:** Diarization finds only 1 speaker

**Solution:** Use rescue script
```bash
python scripts/phase1/rescue_single_speaker.py \
    --session-id 309 \
    --master-audio /path/to/309_AUDIO.wav \
    --anchor /path/to/biometric_anchor.pt \
    --output-dir /path/to/outputs
```

This forces 2-speaker mode and requires manual verification.

---

### Issue: Empty Transcript

**Symptom:** `patient_text.txt` is empty

**Possible Causes:**
1. Very short patient speech (<1 second)
2. Audio file corrupted
3. Whisper failed to detect speech

**Check:**
```bash
# Verify audio duration
ffprobe -i session_XXX/patient_audio.wav

# Check roles.json
cat session_XXX/roles.json  # Look at durations
```

---

### Issue: High Contamination

**Symptom:** >5% interviewer prompts in patient text

**Causes:** Diarization lag at turn boundaries, overlapping speech

**Mitigation:** Phase 2 text cleaning (future work)

---

## Edge Cases

### Session 309 (Rescue Case)

**Problem:** Only 1 speaker detected (voices too similar)

**Resolution:** Forced 2-speaker split + manual verification

**Files:**
- `rescue_metadata.json` documents intervention
- `quality_flag.json` marks for review

### Female-Female Interviews

**Challenge:** Low biometric separation (VERY_LOW confidence)

**Solution:** Duration-based patient selection (patients speak more)

**Validation:** All 22 VERY_LOW sessions spot-checked, duration ratio >2.0 confirms correct assignment.

---

## Next Steps

After Phase 1 completion:

1. **Verify Outputs:**
   ```bash
   python scripts/phase1/validate_outputs.py --output-dir /path/to/outputs
   ```

2. **Review Validation:**
   - Check `validation_summary.csv` for quality flags
   - Review low-confidence sessions if needed

3. **Proceed to Phase 2:**
   - [Phase 2 - Feature Extraction](README_PHASE2.md)
   - Extract WaveLM features from `patient_audio.wav`
   - Extract Mental-RoBERTa embeddings from `patient_text.txt`

---

## Files Used

**Core Scripts:**
- `src/phase1/generate_anchor.py` - Biometric anchor generation
- `src/phase1/diarization.py` - Speaker diarization
- `src/phase1/transcription.py` - Whisper transcription

**Helper Scripts:**
- `scripts/phase1/validate_outputs.py` - Validation suite
- `scripts/phase1/generate_charts.py` - Visualization
- `scripts/phase1/rescue_single_speaker.py` - Edge case handling
- `scripts/phase1/cleanup_master_audio.sh` - Storage cleanup

**Cluster Scripts:**
- `cluster_scripts/phase1/diarization.sbatch` - SLURM diarization job
- `cluster_scripts/phase1/transcription.sbatch` - SLURM transcription job
- `cluster_scripts/phase1/setup/` - Environment setup scripts

---

## References

- **Pyannote 3.1:** https://github.com/pyannote/pyannote-audio
- **Whisper Large-v3:** https://github.com/openai/whisper
- **Faster-Whisper:** https://github.com/guillaumekln/faster-whisper
- **Technical Report:** `docs/reports/Phase1_Technical_Report.pdf`

---

**Last Updated:** February 5, 2026  
**Status:** Production Ready ✅
