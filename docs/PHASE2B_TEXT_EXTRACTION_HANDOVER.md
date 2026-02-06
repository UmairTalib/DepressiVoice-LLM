# Phase 2B: Text Embedding Extraction - Handover Document

**Date:** February 6, 2026  
**Status:** 📋 Planned (Awaiting Implementation)  
**Prerequisite:** Phase 2A (Audio) Complete ✅

---

## Executive Summary

This document provides complete specifications for Phase 2B: extracting Mental-RoBERTa text embeddings from patient transcripts. **Critical:** Text extraction must use the same 8-second windowing strategy as audio to ensure proper alignment for Phase 3 multimodal fusion.

---

## 1. Data Paths (Cluster)

### Input Data: Phase 1 Rerun Outputs

**Location:** `/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/`

**Structure per Session:**
```
phase1_rerun_outputs/session_{ID}/
├── cleaned_patient_text_v2.json   ← PRIMARY INPUT (filler-removed, patient-only)
├── patient_audio.wav              ← Already processed in Phase 2A
├── roles.json                     ← Confidence scores & metadata
├── transcript.json                ← Original transcript (with fillers)
└── diarization.rttm               ← Timing reference
```

**Data Characteristics:**
- **Sessions:** 275 (IDs: 300-718, non-continuous)
- **Text Format:** JSON with word-level timestamps
- **Preprocessing:** Filler words removed, patient speech only
- **Average Length:** ~500-800 words per session

### Output Data: Text Embeddings

**Location:** `/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_raw/`

**Expected Structure:**
```
phase2_clean/
├── text_embeddings_raw/
│   ├── 300_embeddings.pt   ← [Windows, 768] tensor
│   ├── 301_embeddings.pt
│   └── ...                 (275 files total)
│
└── text_embeddings_normalized/  ← Future: normalized embeddings
    └── (same structure)
```

---

## 2. Critical Alignment Requirements

### 🚨 MUST MATCH AUDIO WINDOWING

**Audio Segmentation (Phase 2A Applied):**
- **Window Size:** 8.0 seconds
- **Stride:** 8.0 seconds (non-overlapping)
- **Output Shape:** `[num_windows, 399, 1024]`
  - `num_windows` = variable per session (e.g., Session 300 has 16 windows)
  - `399` = time frames per window
  - `1024` = WavLM feature dimension

**Text Segmentation (Phase 2B MUST Apply):**
- **Window Size:** 8.0 seconds (EXACT MATCH)
- **Stride:** 8.0 seconds (EXACT MATCH)
- **Expected Output Shape:** `[num_windows, 768]`
  - `num_windows` = **SAME** as audio (verified via audio file duration)
  - `768` = Mental-RoBERTa feature dimension

### Alignment Strategy

**Step-by-Step:**

1. **Calculate Session Duration** (from audio file):
   ```python
   import librosa
   audio_file = f"phase1_rerun_outputs/session_{sid}/patient_audio.wav"
   audio, sr = librosa.load(audio_file, sr=16000)
   duration_sec = len(audio) / sr
   num_windows = int(duration_sec / 8.0)  # e.g., 130s / 8.0 = 16 windows
   ```

2. **Segment Text by Timestamps**:
   ```python
   # Load cleaned text with timestamps
   with open(f"session_{sid}/cleaned_patient_text_v2.json") as f:
       words = json.load(f)
   
   # Group words into 8-second windows
   windows = []
   for i in range(num_windows):
       start_time = i * 8.0
       end_time = (i + 1) * 8.0
       window_text = [w for w in words if start_time <= w['start'] < end_time]
       windows.append(window_text)
   ```

3. **Extract Embeddings per Window**:
   ```python
   embeddings_list = []
   for window_words in windows:
       # Concatenate words into text
       text = " ".join([w['word'] for w in window_words])
       
       # Empty window handling
       if not text.strip():
           text = "[SILENCE]"  # Placeholder for silent windows
       
       # Extract Mental-RoBERTa embedding
       embedding = model.encode(text)  # Shape: [768]
       embeddings_list.append(embedding)
   
   # Stack into tensor: [num_windows, 768]
   final_embeddings = torch.stack(embeddings_list)
   ```

4. **Validate Alignment**:
   ```python
   audio_features = torch.load(f"audio_embeddings_raw/{sid}_features.pt")
   text_embeddings = torch.load(f"text_embeddings_raw/{sid}_embeddings.pt")
   
   assert audio_features.shape[0] == text_embeddings.shape[0], \
       f"Window mismatch: audio {audio_features.shape[0]} vs text {text_embeddings.shape[0]}"
   ```

---

## 3. Mental-RoBERTa Configuration

### Model Specifications

- **Model:** `mental/mental-roberta-base`
- **Hugging Face:** https://huggingface.co/mental/mental-roberta-base
- **Base Architecture:** RoBERTa-base (125M parameters)
- **Specialization:** Mental health text analysis
- **Output Dimension:** 768

### Recommended Installation

```bash
pip install sentence-transformers==2.2.2
# OR
pip install transformers torch
```

### Extraction Code Template

```python
from sentence_transformers import SentenceTransformer

# Load model (cache on first use)
model = SentenceTransformer('mental/mental-roberta-base')

# Extract embedding for text window
text = "I feel tired all the time and don't enjoy things anymore"
embedding = model.encode(text, convert_to_tensor=True)  # Shape: [768]
```

### Alternative (Transformers Library):

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base")
model = AutoModel.from_pretrained("mental/mental-roberta-base")

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling: [1, 768]
```

---

## 4. Session Mapping & Metadata

### Complete Session List

**Total:** 275 sessions  
**ID Range:** 300-718 (non-continuous)

**Session IDs (First 20):**
```
300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
310, 311, 312, 313, 314, 315, 316, 317, 318, 319
```

**Full List Available:**
```bash
# On cluster
ls /home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/ | grep 'session_' | sed 's/session_//' | sort -n
```

### Metadata Integration

**Use Confidence Scores for Weighting:**

```python
import json

# Load confidence score
with open(f"phase1_rerun_outputs/session_{sid}/roles.json") as f:
    metadata = json.load(f)
    confidence = metadata['confidence_score']  # 0.0-1.0
    confidence_label = metadata['confidence_label']  # HIGH/MEDIUM/LOW/VERY_LOW

# Recommendation: Weight samples during training
sample_weight = confidence ** 2  # Quadratic weighting
```

**Confidence Distribution:**

| Level | Sessions | Percentage | Recommendation |
|-------|----------|------------|----------------|
| HIGH | 204 | 74.2% | Full weight (1.0) |
| MEDIUM | 34 | 12.4% | Moderate weight (0.64-0.81) |
| LOW | 14 | 5.1% | Low weight (0.25-0.36) |
| VERY_LOW | 22 | 8.0% | Very low weight (0.01-0.09) |
| Session 309 | 1 | 0.4% | **Exclude** (confidence=0.0) |

---

## 5. Text Data Format

### `cleaned_patient_text_v2.json` Structure

```json
[
  {
    "word": "sometimes",
    "start": 0.52,
    "end": 0.88,
    "score": 0.98
  },
  {
    "word": "i",
    "start": 0.92,
    "end": 1.04,
    "score": 0.99
  },
  {
    "word": "feel",
    "start": 1.04,
    "end": 1.28,
    "score": 0.97
  }
]
```

**Fields:**
- `word`: Transcribed word (lowercase, filler-removed)
- `start`: Start time in seconds (relative to patient_audio.wav)
- `end`: End time in seconds
- `score`: Whisper confidence score (typically >0.9)

**Preprocessing Already Applied:**
- ✅ Filler words removed ("um", "uh", "like", etc.)
- ✅ Patient speech only (interviewer removed)
- ✅ Concatenated from original segments
- ✅ Timestamps adjusted to patient_audio.wav timeline

---

## 6. Edge Cases & Handling

### Silent Windows

**Problem:** Some 8-second windows may have no speech (pauses, silence).

**Solution:**
```python
if not window_text or len(window_text.strip()) == 0:
    window_text = "[SILENCE]"  # Use placeholder token
    # OR
    window_text = "no speech"  # Simple description
```

**Rationale:** Mental-RoBERTa needs text input; cannot embed empty strings.

### Very Long Windows

**Problem:** Rare cases where a window has >512 tokens (RoBERTa limit).

**Solution:**
```python
# Truncation is automatic in tokenizer
inputs = tokenizer(text, max_length=512, truncation=True)
```

**Frequency:** Unlikely (8 seconds ~= 80-120 words, well below limit).

### Session 309 (Rescue Case)

**Issue:** Forced 2-speaker split from 1-speaker detection, confidence=0.0.

**Recommendation:**
```python
if session_id == 309:
    print(f"Skipping session 309 (rescue case, unreliable diarization)")
    continue  # Or apply very low weight (0.01)
```

---

## 7. Validation Checklist

### Post-Extraction Validation

1. **File Count Check:**
   ```bash
   ls data/phase2_clean/text_embeddings_raw/*.pt | wc -l
   # Expected: 275 (or 274 if excluding session 309)
   ```

2. **Shape Consistency:**
   ```python
   for session_id in all_sessions:
       audio_feat = torch.load(f"audio_embeddings_raw/{session_id}_features.pt")
       text_emb = torch.load(f"text_embeddings_raw/{session_id}_embeddings.pt")
       
       assert audio_feat.shape[0] == text_emb.shape[0], \
           f"Window count mismatch: {session_id}"
       assert text_emb.shape[1] == 768, \
           f"Embedding dimension wrong: {text_emb.shape}"
   ```

3. **Value Range Check:**
   ```python
   embeddings = torch.load("text_embeddings_raw/300_embeddings.pt")
   print(f"Max: {embeddings.max():.4f}")
   print(f"Min: {embeddings.min():.4f}")
   print(f"Mean: {embeddings.mean():.4f}")
   # Expected: Max ~5-10, Min ~-5 to -10, Mean close to 0
   ```

4. **NaN/Inf Check:**
   ```python
   assert not torch.isnan(embeddings).any()
   assert not torch.isinf(embeddings).any()
   ```

---

## 8. Recommended Implementation

### Script: `src/phase2_text/extract_embeddings.py`

```python
#!/usr/bin/env python3
"""
Phase 2B: Mental-RoBERTa Text Embedding Extraction
Extracts 768-dim embeddings with 8-second windowing to match audio
"""

import argparse
import json
import torch
import librosa
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def extract_text_embeddings(session_id, input_dir, output_dir, model):
    # 1. Load audio to get duration/num_windows
    audio_file = f"{input_dir}/session_{session_id}/patient_audio.wav"
    audio, sr = librosa.load(audio_file, sr=16000)
    duration = len(audio) / sr
    num_windows = int(duration / 8.0)
    
    # 2. Load cleaned text
    text_file = f"{input_dir}/session_{session_id}/cleaned_patient_text_v2.json"
    with open(text_file) as f:
        words = json.load(f)
    
    # 3. Segment into 8-second windows
    embeddings = []
    for i in range(num_windows):
        start_time = i * 8.0
        end_time = (i + 1) * 8.0
        
        # Get words in this window
        window_words = [w for w in words if start_time <= w['start'] < end_time]
        text = " ".join([w['word'] for w in window_words])
        
        # Handle silence
        if not text.strip():
            text = "[SILENCE]"
        
        # Extract embedding
        embedding = model.encode(text, convert_to_tensor=True)
        embeddings.append(embedding)
    
    # 4. Save
    embeddings_tensor = torch.stack(embeddings)  # [num_windows, 768]
    output_file = f"{output_dir}/{session_id}_embeddings.pt"
    torch.save(embeddings_tensor, output_file)
    
    return embeddings_tensor.shape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--input_dir", default="/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs")
    parser.add_argument("--output_dir", default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_raw")
    args = parser.parse_args()
    
    # Load model
    print("Loading Mental-RoBERTa model...")
    model = SentenceTransformer('mental/mental-roberta-base')
    
    # Extract
    shape = extract_text_embeddings(args.session_id, args.input_dir, args.output_dir, model)
    print(f"Session {args.session_id}: {shape}")

if __name__ == "__main__":
    main()
```

### SLURM Script: `cluster_scripts/phase2/extract_text_embeddings.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=MentalRoBERTa_Phase2
#SBATCH --output=logs/phase2_text_%A_%a.out
#SBATCH --error=logs/phase2_text_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-274%20

# Phase 2B: Mental-RoBERTa Text Embedding Extraction
# Extracts 768-dim embeddings with 8-second windowing (matches audio)

module load python/3.11.2

PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
INPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
OUTPUT_DIR="$PROJECT_ROOT/data/phase2_clean/text_embeddings_raw"
SCRIPT_PATH="$PROJECT_ROOT/src/phase2_text/extract_embeddings.py"

mkdir -p logs
mkdir -p $OUTPUT_DIR

# Get session ID
mapfile -t SESSIONS < <(ls $INPUT_DIR | grep 'session_' | sed 's/session_//' | sort -n)
SESSION_ID=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "Processing Session: $SESSION_ID"
echo "Input text: $INPUT_DIR/session_$SESSION_ID/cleaned_patient_text_v2.json"
echo "Output: $OUTPUT_DIR/${SESSION_ID}_embeddings.pt"

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

python -u $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Extraction complete for session $SESSION_ID"
```

---

## 9. Phase 3 Handover Preview

### Expected Multimodal Input

**Per Session:**
```python
# Audio features
audio = torch.load(f"audio_embeddings_normalized/{sid}_features.pt")
# Shape: [num_windows, 399, 1024]

# Text embeddings
text = torch.load(f"text_embeddings_normalized/{sid}_embeddings.pt")
# Shape: [num_windows, 768]

# Verify alignment
assert audio.shape[0] == text.shape[0], "Window count must match!"

# Fusion strategy (example)
audio_pooled = audio.mean(dim=1)  # [num_windows, 1024]
fused = torch.cat([audio_pooled, text], dim=-1)  # [num_windows, 1792]
```

### Confidence Weighting

```python
# Load confidence
with open(f"phase1_rerun_outputs/session_{sid}/roles.json") as f:
    confidence = json.load(f)['confidence_score']

# Apply to loss
sample_weight = confidence ** 2
weighted_loss = loss * sample_weight
```

---

## 10. Critical Success Factors

### ✅ DO:
1. Use **exactly 8.0-second windows** (matching audio)
2. Validate **window count** matches audio for each session
3. Handle **silent/empty windows** with placeholder text
4. Use **Mental-RoBERTa** model (not generic RoBERTa)
5. Save as **FP16 tensors** for efficiency
6. Apply **confidence weighting** (from roles.json)
7. Test on **3-5 sessions** before full batch run

### ❌ DON'T:
1. Use sentence-level or full-session embeddings (breaks alignment)
2. Ignore timestamp information from JSON
3. Use `transcript.json` (has fillers + interviewer speech)
4. Assume all windows have text (silence must be handled)
5. Skip validation against audio window counts
6. Process session 309 without special handling
7. Forget to set `HF_HOME` environment variable

---

## 11. Quick Start Commands

### On Cluster:

```bash
# 1. Install dependencies
pip install --user sentence-transformers librosa soundfile

# 2. Test single session
python src/phase2_text/extract_embeddings.py \
    --session_id 300 \
    --input_dir ~/DepressiVoice-LLM/phase1_rerun_outputs \
    --output_dir ~/DepressiVoice-LLM/data/phase2_clean/text_embeddings_raw

# 3. Validate alignment
python -c "
import torch
audio = torch.load('data/phase2_clean/audio_embeddings_normalized/300_features.pt')
text = torch.load('data/phase2_clean/text_embeddings_raw/300_embeddings.pt')
print(f'Audio windows: {audio.shape[0]}')
print(f'Text windows: {text.shape[0]}')
print(f'Match: {audio.shape[0] == text.shape[0]}')
"

# 4. Submit batch job
sbatch cluster_scripts/phase2/extract_text_embeddings.sbatch
```

---

## 12. Contact & Support

**Questions?** Refer to:
- Phase 2A Technical Report: `docs/reports/Phase2_Technical_Report.pdf`
- Audio extraction script: `src/phase2_audio/extract_features.py`
- Validation utilities: `src/phase2_audio/validate.py`

**Key Alignment Rule:**  
> If audio has N windows, text MUST have N windows. Period.

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026  
**Status:** Ready for Phase 2B implementation
