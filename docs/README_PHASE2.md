# Phase 2: Feature Extraction (Audio + Text)

**Status:** ✅ COMPLETE
**Inputs:** Phase 1 outputs (patient audio + transcripts)  
**Outputs:** WaveLM audio features + Mental-RoBERTa text embeddings  
**Next:** [Phase 3 - Multimodal Fusion](README_PHASE3.md)

---

## Overview

Phase 2 extracts features from Phase 1 outputs for depression detection modeling:

1. **WaveLM** - Extract acoustic features from patient audio
2. **Mental-RoBERTa** - Extract semantic embeddings from patient text

These features are later fused in Phase 3 for multimodal classification.

---

## Phase 1 Output Paths (CRITICAL)

All Phase 2 scripts read from Phase 1 outputs on the cluster:

```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/
├── session_300/
│   ├── patient_audio.wav          ← INPUT for WaveLM
│   ├── transcript.json             ← INPUT for Mental-RoBERTa (structured)
│   ├── patient_text.txt            ← INPUT for Mental-RoBERTa (simple)
│   ├── roles.json                  ← Confidence scores for weighting
│   └── quality_flag.json           ← Quality marker (37 sessions)
├── session_301/
├── ...
└── validation_summary.csv          ← Per-session metadata
```

**Dataset:**
- **Total Sessions:** 275 (session 300-718, non-continuous)
- **Total Audio:** 34.1 hours
- **Average Duration:** 7.4 minutes per session
- **High Confidence:** 204 sessions (74.2%)
- **Quality Flags:** 37 sessions (13.5%)

---

## Part 1: WaveLM Feature Extraction (Audio)

### Overview

Extract acoustic features from patient audio using WaveLM Large (SOTA for speech representation).

**Model:** `microsoft/wavlm-large`  
**Input:** 16kHz mono WAV files  
**Output:** 1024-dim embeddings per session

### Location

**Source Code:** `src/phase2_audio/`
```
src/phase2_audio/
├── __init__.py
├── dataset.py          # Audio dataset loader
├── audiomodel.py       # WaveLM model wrapper
├── train_audio.py      # Training script
└── test_local.py       # Local testing
```

**Cluster Scripts:** `cluster_scripts/phase2/`
```
cluster_scripts/phase2/
├── extract_audio_features.sbatch          # Batch feature extraction
├── extract_audio_features_sota.sbatch     # SOTA config
└── train_audio.sbatch                     # Model training
```

### Feature Extraction

**Script:** `cluster_scripts/phase2/extract_audio_features.sbatch`

**Submit:**
```bash
cd /home/g755389/DepressiVoice-LLM
sbatch cluster_scripts/phase2/extract_audio_features.sbatch
```

**Configuration:**
- Model: `microsoft/wavlm-large`
- Layer: 12 (middle layer, best for downstream tasks)
- Pooling: Mean pooling over time
- Normalization: L2 normalization

**Input Path Pattern:**
```python
audio_path = f"/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/patient_audio.wav"
```

**Output Path Pattern:**
```python
features_path = f"/home/g755389/DepressiVoice-LLM/data/03_features/audio/session_{session_id}_wavlm.npy"
```

**Expected Output:**
- **Format:** NumPy array (.npy)
- **Shape:** `(1024,)` per session
- **Total Files:** 275 .npy files
- **Total Size:** ~300 MB

### Confidence Weighting

Use confidence scores from Phase 1 for sample weighting during training:

```python
import json
import pandas as pd

# Load validation summary
df = pd.read_csv('phase1_rerun_outputs/validation_summary.csv')

# Map confidence to weights
weights = df['confidence'].map({
    'HIGH': 1.0,
    'MEDIUM': 0.7,
    'LOW': 0.5,
    'VERY_LOW': 0.3
})

# Or load per-session confidence
with open(f'session_{session_id}/roles.json') as f:
    confidence_score = json.load(f)['confidence_score']  # 0.3 to 1.0
```

---

## Part 2: Mental-RoBERTa Embedding Extraction (Text)

### Overview

Extract semantic embeddings from patient transcripts using Mental-RoBERTa (fine-tuned for mental health text).

**Model:** `mental/mental-roberta-base` (Hugging Face)  
**Input:** Plain text transcripts  
**Output:** 768-dim embeddings per session

### Location

**Source Code:** `src/phase2_text/`
```
src/phase2_text/
├── extract_embeddings.py     # Mental-RoBERTa extraction (8s windowing)
└── validate_alignment.py     # Feature alignment verification
```

**Cluster Scripts:** `cluster_scripts/phase2/`
```
cluster_scripts/phase2/
└── extract_text_embeddings.sbatch           # Batch extraction (with --skip_filtering for robustness)
```

### Text Input Options

**Option 1: Plain Text** (Recommended for simplicity)
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/patient_text.txt
```

**Option 2: Structured JSON** (Recommended for temporal analysis)
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/transcript.json
```

**Example Usage:**
```python
# Simple approach
with open(f"session_{session_id}/patient_text.txt", 'r') as f:
    text = f.read()
embeddings = mental_roberta.encode(text)

# Segment-level approach
import json
with open(f"session_{session_id}/transcript.json", 'r') as f:
    data = json.load(f)
segments = [seg['text'] for seg in data['segments']]
embeddings = [mental_roberta.encode(seg) for seg in segments]
```

### Feature Extraction

**Script:** `cluster_scripts/phase2/extract_text_embeddings.sbatch`

**Submit:**
```bash
sbatch cluster_scripts/phase2/extract_text_embeddings.sbatch
```

**Configuration:**
- Model: `mental/mental-roberta-base`
- Max length: 512 tokens
- Pooling: CLS token or mean pooling

**Output Path Pattern:**
```python
embeddings_path = f"/home/g755389/DepressiVoice-LLM/data/03_features/text/session_{session_id}_roberta.npy"
```

**Expected Output:**
- **Format:** NumPy array (.npy)
- **Shape:** `(768,)` per session
- **Total Files:** 275 .npy files
- **Total Size:** ~200 MB

### Contamination Filtering (Optional)

Average contamination is 1.71% (acceptable), but you can filter further:

```python
# Load contamination rates
df = pd.read_csv('validation_summary.csv')

# Filter sessions with >5% contamination
clean_sessions = df[df['contamination_rate'] < 5.0]['session_id'].tolist()

# Or remove interviewer phrases manually
ellie_phrases = ["tell me", "how are you", "what do you", "can you describe"]
cleaned_text = text
for phrase in ellie_phrases:
    cleaned_text = cleaned_text.replace(phrase, "")
```

---

## Feature Organization

### Directory Structure

```
data/03_features/
├── audio/
│   ├── session_300_wavlm.npy
│   ├── session_301_wavlm.npy
│   └── ...
│
└── text/
    ├── session_300_roberta.npy
    ├── session_301_roberta.npy
    └── ...
```

### Alignment Verification

Ensure audio and text features are aligned by session ID:

```python
import numpy as np
from pathlib import Path

audio_dir = Path('/home/g755389/DepressiVoice-LLM/data/03_features/audio')
text_dir = Path('/home/g755389/DepressiVoice-LLM/data/03_features/text')

# Get session IDs
audio_sessions = sorted([int(f.stem.split('_')[1]) for f in audio_dir.glob('*.npy')])
text_sessions = sorted([int(f.stem.split('_')[1]) for f in text_dir.glob('*.npy')])

assert audio_sessions == text_sessions, "Session mismatch!"
print(f"✅ {len(audio_sessions)} sessions aligned")
```

---

## Filtering Recommendations

### By Confidence

**High Quality Only** (238 sessions):
```python
df_clean = df[df['confidence'].isin(['HIGH', 'MEDIUM'])]
session_ids = df_clean['session_id'].tolist()
```

**Exclude Quality Flagged** (238 sessions):
```python
df_clean = df[df['quality_flag'] == False]
session_ids = df_clean['session_id'].tolist()
```

### By Audio Duration

**Minimum Duration** (e.g., 1 minute):
```python
df_clean = df[df['audio_duration'] > 60]  # seconds
session_ids = df_clean['session_id'].tolist()
```

### By Contamination

**Low Contamination Only**:
```python
df_clean = df[df['contamination_rate'] < 5.0]
session_ids = df_clean['session_id'].tolist()
```

---

## Complete Workflow Example

```python
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load validation metadata
df = pd.read_csv('/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/validation_summary.csv')

# 2. Filter sessions
df_clean = df[
    (df['confidence'].isin(['HIGH', 'MEDIUM'])) &
    (df['contamination_rate'] < 5.0) &
    (df['audio_duration'] > 60)
]
session_ids = sorted(df_clean['session_id'].tolist())
print(f"Processing {len(session_ids)} sessions")

# 3. Load features
audio_features = []
text_features = []

for session_id in session_ids:
    # Load audio features
    audio_path = f'/home/g755389/DepressiVoice-LLM/data/03_features/audio/session_{session_id}_wavlm.npy'
    audio_feat = np.load(audio_path)
    audio_features.append(audio_feat)
    
    # Load text features
    text_path = f'/home/g755389/DepressiVoice-LLM/data/03_features/text/session_{session_id}_roberta.npy'
    text_feat = np.load(text_path)
    text_features.append(text_feat)

audio_features = np.array(audio_features)  # Shape: (N, 1024)
text_features = np.array(text_features)    # Shape: (N, 768)

print(f"Audio features: {audio_features.shape}")
print(f"Text features: {text_features.shape'}")

#4. Apply confidence weighting
weights = df_clean['confidence'].map({
    'HIGH': 1.0,
    'MEDIUM': 0.7,
    'LOW': 0.5,
    'VERY_LOW': 0.3
}).values

# Ready for Phase 3 fusion!
```

---

## Next Steps

After Phase 2 completion:

1. **Verify Features:**
   - Check feature shape consistency
   - Verify session alignment
   - Validate feature statistics

2. **Proceed to Phase 3:**
   - [Phase 3 - Multimodal Fusion](README_PHASE3.md)
   - Concatenate or fuse audio + text features
   - Train depression detection classifier

---

## Files Used

**Audio Feature Extraction:**
- `src/phase2_audio/dataset.py` - Audio dataset loader
- `src/phase2_audio/audiomodel.py` - WaveLM wrapper
- `cluster_scripts/phase2/extract_audio_features.sbatch` - Batch extraction

**Text Embedding Extraction:**
- `src/phase2_text/extract_embeddings.py` - Mental-RoBERTa extraction
- `cluster_scripts/phase2/extract_text_embeddings.sbatch` - Batch extraction

**Phase 1 Reference:**
- [Phase 1 - Diarization & Transcription](README_PHASE1.md)
- `docs/PHASE2_HANDOVER_PATHS.md` - Complete path reference

---

**Last Updated:** February 5, 2026  
**Status:** Phase 1 Complete ✅, Phase 2 Audio Complete ✅, Phase 2 Text Complete ✅
