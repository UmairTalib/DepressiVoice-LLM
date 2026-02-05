# Phase 1 → Phase 2 Handover: Critical Paths & Information

**For WaveLM (Audio) and Mental-RoBERTa (Text) Feature Extraction Agents**  
**Date:** February 5, 2026  
**Phase 1 Status:** ✅ COMPLETE (275/275 sessions)

---

## 🎯 Quick Reference

### Cluster Paths (OMNI HPC)

**Phase 1 Outputs:**
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/
├── session_300/
│   ├── patient_audio.wav          ← AUDIO INPUT for WaveLM
│   ├── transcript.json             ← TRANSCRIPTION for Mental-RoBERTa
│   ├── patient_text.txt            ← PLAIN TEXT for Mental-RoBERTa
│   ├── roles.json                  ← CONFIDENCE SCORES (for weighting)
│   └── quality_flag.json           ← LOW-CONFIDENCE MARKER (if present)
├── session_301/
└── ... (275 sessions total)
```

**Validation Data:**
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/
├── validation_results.json         ← SUMMARY METRICS
└── validation_summary.csv          ← PER-SESSION CONFIDENCE
```

---

## 📊 Dataset Overview

### Sessions Available

| Metric | Value |
|--------|-------|
| **Total Sessions** | 275 |
| **Complete Sessions** | 275 (100%) |
| **Missing/Failed** | 0 |
| **Total Audio Duration** | 34.1 hours |
| **Average Duration** | 7.4 minutes/session |

### Session ID Range

- **First Session:** 300
- **Last Session:** 718
- **Gaps:** Yes (not continuous, e.g., 300, 301, 302, ..., 718)
- **Special Case:** Session 309 (rescue mode, flagged)

---

## 🎵 Audio Paths for WaveLM

### Primary Input

**File:** `patient_audio.wav` (per session)

**Location Pattern:**
```python
audio_path = f"/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/patient_audio.wav"
```

**Example:**
```bash
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_632/patient_audio.wav
```

### Audio Specifications

| Property | Value |
|----------|-------|
| **Format** | WAV (uncompressed) |
| **Sample Rate** | 16,000 Hz |
| **Channels** | 1 (mono) |
| **Bit Depth** | 16-bit PCM |
| **Speaker** | Patient only (interviewer removed) |
| **Processing** | Concatenated segments (silence removed) |

### Audio Characteristics

- **Content:** Continuous patient speech only
- **Segments:** Multiple patient utterances concatenated
- **Timestamps:** Available in `transcript.json` (relative to this audio)
- **Quality:** VAD-filtered, interviewer-free

### Important Notes

⚠️ **Timestamp Alignment:**
- Word timestamps in `transcript.json` are relative to `patient_audio.wav`
- NOT aligned to original master audio (master audio deleted)
- If you need segment boundaries, use `XXX.rttm` file

⚠️ **Contamination:**
- Average 1.71% interviewer prompts leaked into transcript
- 97% of sessions have <5% contamination (acceptable)
- Check `contamination_rate` in `validation_summary.csv` if needed

---

## 📝 Text Paths for Mental-RoBERTa

### Primary Input Options

**Option 1: Plain Text** (Recommended for simplicity)
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/patient_text.txt
```

**Option 2: Structured JSON** (Recommended for timestamps)
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/transcript.json
```

### File Format Details

#### `patient_text.txt`

**Format:** Plain UTF-8 text

**Content:** Patient speech only (no timestamps)

**Example:**
```
I've been feeling really down lately. It's hard to get out of bed in the morning. 
I don't enjoy things I used to like. My sleep has been terrible too.
```

**Pros:**
- Simple to parse
- Clean text ready for RoBERTa
- Lowercase, normalized

**Cons:**
- No timestamps
- No segment boundaries

---

#### `transcript.json`

**Format:** JSON with nested structure

**Content:** Segments with word-level timestamps

**Schema:**
```json
{
  "language": "en",
  "duration": 446.8,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "I've been feeling really down lately",
      "words": [
        {"start": 0.0, "end": 0.3, "word": "I've"},
        {"start": 0.3, "end": 0.6, "word": "been"},
        ...
      ],
      "avg_logprob": -0.25,
      "no_speech_prob": 0.01
    },
    ...
  ]
}
```

**Key Fields:**
- `segments`: List of utterance segments
- `start/end`: Timestamps (seconds, relative to patient_audio.wav)
- `text`: Transcribed text for this segment
- `words`: Word-level timestamps (optional, for fine-grained analysis)
- `avg_logprob`: Transcription confidence (lower = more confident)
- `no_speech_prob`: Probability of silence (lower = more confident speech)

**Pros:**
- Timestamps for temporal features
- Word-level granularity
- Confidence scores

**Cons:**
- More complex parsing
- Need to concatenate segments for full text

---

### Text Processing Recommendations

**For Mental-RoBERTa:**

1. **Basic Approach** (Use `patient_text.txt`):
   ```python
   with open(f"session_{session_id}/patient_text.txt", 'r') as f:
       text = f.read()
   embeddings = mental_roberta.encode(text)
   ```

2. **Segment-Level Approach** (Use `transcript.json`):
   ```python
   import json
   with open(f"session_{session_id}/transcript.json", 'r') as f:
       data = json.load(f)
   
   segments = [seg['text'] for seg in data['segments']]
   embeddings = [mental_roberta.encode(seg) for seg in segments]
   ```

3. **Filtering Contamination** (Optional):
   ```python
   # Remove known interviewer phrases
   ellie_phrases = ["tell me", "how are you", "what do you", "can you describe"]
   
   cleaned_text = text
   for phrase in ellie_phrases:
       cleaned_text = cleaned_text.replace(phrase, "")
   ```

---

## 🎯 Confidence Scores & Quality Flags

### Confidence Metadata

**File:** `roles.json` (per session)

**Location:**
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/roles.json
```

**Schema:**
```json
{
  "roles": {
    "SPEAKER_00": "ELLIE",
    "SPEAKER_01": "PATIENT"
  },
  "similarities": {
    "SPEAKER_00": 0.8543,
    "SPEAKER_01": 0.6321
  },
  "separation_margin": 0.2222,
  "confidence": "HIGH",
  "confidence_score": 1.0,
  "duration_ratio": 5.91,
  "durations": {
    "ellie": 21.5,
    "patient": 127.2
  }
}
```

**Key Fields for Phase 2:**

| Field | Type | Range | Use Case |
|-------|------|-------|----------|
| `confidence` | String | HIGH/MEDIUM/LOW/VERY_LOW | Filter or weight samples |
| `confidence_score` | Float | 0.3 - 1.0 | Sample weighting in training |
| `separation_margin` | Float | 0.001 - 0.6 | Diarization quality indicator |
| `duration_ratio` | Float | 0.5 - 25.0 | Patient engagement level |

---

### Quality Flags

**File:** `quality_flag.json` (only for 37 sessions with LOW/VERY_LOW confidence)

**Location:**
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/quality_flag.json
```

**Schema:**
```json
{
  "session_id": "632",
  "flag_type": "LOW_SEPARATION_MARGIN",
  "confidence": "VERY_LOW",
  "margin": 0.0012,
  "reason": "Low biometric separation - likely same-gender speakers (female-female)",
  "review_recommended": true,
  "duration_ratio": 25.58
}
```

**Action for Phase 2:**
- **If file exists:** Consider downweighting or excluding session
- **If file missing:** Session has acceptable confidence (MEDIUM or HIGH)

---

## 📋 Validation Summary

### Per-Session Metadata

**File:** `validation_summary.csv`

**Location:**
```
/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/validation_summary.csv
```

**Columns:**
```csv
session_id,complete,audio_duration,segments,contamination_rate,confidence,margin,quality_flag,rescue,language
300,True,130.25,20,0.0,MEDIUM,0.197,False,False,en
632,True,580.73,182,0.55,VERY_LOW,0.0012,True,False,en
...
```

**Key Columns for Phase 2:**

| Column | Description | Use Case |
|--------|-------------|----------|
| `session_id` | Session ID (300-718) | Index sessions |
| `audio_duration` | Duration (seconds) | Filter short sessions |
| `segments` | Number of utterances | Complexity indicator |
| `contamination_rate` | % interviewer prompts | Text quality filter |
| `confidence` | Diarization confidence | Sample weighting |
| `quality_flag` | True/False | Exclude if needed |

**Usage Example:**
```python
import pandas as pd
df = pd.read_csv('validation_summary.csv')

# Filter high-quality sessions only
high_quality = df[
    (df['confidence'].isin(['HIGH', 'MEDIUM'])) &
    (df['contamination_rate'] < 5.0) &
    (df['audio_duration'] > 60)
]

# Use confidence scores for weighting
weights = df['confidence'].map({
    'HIGH': 1.0,
    'MEDIUM': 0.7,
    'LOW': 0.5,
    'VERY_LOW': 0.3
})
```

---

## 🚀 Phase 2 Workflow Recommendations

### For WaveLM (Audio Features)

**Step 1: Load Session List**
```python
import pandas as pd
df = pd.read_csv('/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/validation_summary.csv')
session_ids = df['session_id'].tolist()  # [300, 301, ...]
```

**Step 2: Process Audio**
```python
for session_id in session_ids:
    audio_file = f"/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/patient_audio.wav"
    
    # Load audio (e.g., with librosa or soundfile)
    import soundfile as sf
    waveform, sr = sf.read(audio_file)
    
    # Extract WaveLM features
    features = wavlm_model.extract_features(waveform, sr)
    
    # Save features
    output_path = f"/home/g755389/phase2_features/audio/session_{session_id}_wavlm.npy"
    np.save(output_path, features)
```

**Step 3: Include Confidence Weighting** (Optional)
```python
import json
with open(f"session_{session_id}/roles.json", 'r') as f:
    roles = json.load(f)
    confidence_weight = roles['confidence_score']  # 0.3 to 1.0
```

---

### For Mental-RoBERTa (Text Features)

**Step 1: Load Session List** (same as above)

**Step 2: Process Text**
```python
for session_id in session_ids:
    text_file = f"/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}/patient_text.txt"
    
    # Load text
    with open(text_file, 'r') as f:
        text = f.read()
    
    # Extract Mental-RoBERTa embeddings
    embeddings = mental_roberta.encode(text)
    
    # Save features
    output_path = f"/home/g755389/phase2_features/text/session_{session_id}_roberta.npy"
    np.save(output_path, embeddings)
```

**Step 3: Filter Contamination** (Optional)
```python
# Load contamination rates
df = pd.read_csv('validation_summary.csv')
clean_sessions = df[df['contamination_rate'] < 5.0]['session_id'].tolist()

# Process only clean sessions
for session_id in clean_sessions:
    # ... extract features
```

---

### For Multimodal Fusion

**Align Audio & Text Features:**

Both WaveLM and Mental-RoBERTa embeddings should use the same session order:

```python
# Ensure same session order
session_ids = sorted(df['session_id'].tolist())

audio_features = []
text_features = []

for session_id in session_ids:
    audio_feat = np.load(f"audio/session_{session_id}_wavlm.npy")
    text_feat = np.load(f"text/session_{session_id}_roberta.npy")
    
    audio_features.append(audio_feat)
    text_features.append(text_feat)

# Now aligned for fusion
audio_features = np.array(audio_features)  # shape: (275, audio_dim)
text_features = np.array(text_features)    # shape: (275, text_dim)
```

---

## ⚠️ Important Considerations

### 1. Session 309 (Rescue Case)

**Status:** ✅ Included in dataset, **BUT flagged**

**Issue:** Only 1 speaker detected, forced 2-speaker split

**Files Present:**
- `patient_audio.wav` ✅
- `transcript.json` ✅
- `rescue_metadata.json` ✅ (documents rescue procedure)
- `quality_flag.json` ✅

**Recommendation:**
- **Include:** For completeness (275 vs 274)
- **Downweight:** Use confidence_score = 0.0 or exclude if strict quality needed
- **Note:** Manual verification confirmed correct patient audio

---

### 2. Low-Confidence Sessions (37 total)

**Breakdown:**
- VERY_LOW: 22 sessions (likely female-female speakers)
- LOW: 14 sessions
- MEDIUM: 1 session (edge case)

**Recommendation:**
- **HIGH/MEDIUM (238 sessions):** Use with full weight
- **LOW (14 sessions):** Use with 0.5 weight or exclude
- **VERY_LOW (22 sessions):** Use with 0.3 weight or exclude

**Filter Example:**
```python
# Exclude LOW/VERY_LOW
df_filtered = df[df['confidence'].isin(['HIGH', 'MEDIUM'])]  # 238 sessions
```

---

### 3. Contamination Handling

**Average Contamination:** 1.71% (very low)

**Sessions with >5% Contamination:** 8 sessions

**Recommendation:**
- **For most sessions:** No action needed
- **For high contamination (>10%):** Consider text cleaning or exclusion
- **Check:** `contamination_rate` column in `validation_summary.csv`

---

## 📦 File Inventory Summary

### Per Session (275 × 7 files = 1,925 files)

**Required Files:**
1. `patient_audio.wav` - Audio input
2. `transcript.json` - Structured transcription
3. `patient_text.txt` - Plain text
4. `roles.json` - Confidence metadata
5. `XXX.rttm` - Diarization segments

**Optional Files:**
6. `quality_flag.json` - Present for 37 sessions
7. `rescue_metadata.json` - Present for session 309 only

---

### Aggregate Files (at top level)

1. `validation_results.json` - Summary statistics
2. `validation_summary.csv` - Per-session metadata (USE THIS!)

---

## 🎓 Quick Start Example

**Complete Pipeline for Phase 2:**

```python
import pandas as pd
import soundfile as sf
import numpy as np

# Load session metadata
df = pd.read_csv('/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/validation_summary.csv')

# Filter high-quality sessions
df_clean = df[
    (df['confidence'].isin(['HIGH', 'MEDIUM'])) &
    (df['contamination_rate'] < 5.0) &
    (df['audio_duration'] > 60)  # At least 1 minute
]

session_ids = df_clean['session_id'].tolist()
print(f"Processing {len(session_ids)} high-quality sessions")

# Process each session
for session_id in session_ids:
    base_path = f"/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_{session_id}"
    
    # Load audio for WaveLM
    audio, sr = sf.read(f"{base_path}/patient_audio.wav")
    # audio_features = wavlm_model.extract(audio, sr)
    
    # Load text for Mental-RoBERTa
    with open(f"{base_path}/patient_text.txt", 'r') as f:
        text = f.read()
    # text_features = mental_roberta.encode(text)
    
    # Load confidence for weighting
    import json
    with open(f"{base_path}/roles.json", 'r') as f:
        weight = json.load(f)['confidence_score']
    
    # Process and save...
    print(f"Session {session_id}: Audio={audio.shape}, Weight={weight}")
```

---

## 📊 Expected Outputs for Phase 2

**WaveLM Features:**
```
/home/g755389/phase2_features/audio/
├── session_300_wavlm.npy
├── session_301_wavlm.npy
└── ... (275 files)
```

**Mental-RoBERTa Features:**
```
/home/g755389/phase2_features/text/
├── session_300_roberta.npy
├── session_301_roberta.npy
└── ... (275 files)
```

**Metadata:**
```
/home/g755389/phase2_features/
├── feature_metadata.csv  (session_id, audio_dim, text_dim, weight)
└── session_manifest.json  (list of processed sessions)
```

---

## ✅ Final Checklist for Phase 2 Agents

### Before Starting

- [ ] Cluster access verified (`ssh g755389@omni.zimt.uni-siegen.de`)
- [ ] Phase 1 outputs exist (`/home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/`)
- [ ] `validation_summary.csv` loaded successfully
- [ ] 275 sessions confirmed (or filtered subset)

### For WaveLM Agent

- [ ] Audio file path pattern understood
- [ ] Sample rate (16kHz) confirmed
- [ ] Output directory created (`phase2_features/audio/`)
- [ ] Confidence weighting strategy decided

### For Mental-RoBERTa Agent

- [ ] Text file format chosen (`patient_text.txt` vs `transcript.json`)
- [ ] Contamination filtering strategy decided
- [ ] Output directory created (`phase2_features/text/`)
- [ ] Tokenization max length set appropriately

### For Both

- [ ] Session ID alignment verified
- [ ] Feature dimension documented
- [ ] Confidence weights incorporated
- [ ] Error handling for missing files
- [ ] Progress logging implemented

---

## 🆘 Troubleshooting

### Audio File Not Found

**Check:**
```bash
ls /home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_300/patient_audio.wav
```

**Expected:** File exists (6.0 MB for session 300)

**If Missing:** Check session ID is in valid range (300-718, with gaps)

---

### Empty Transcript

**Check:**
```bash
wc -l /home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_315/patient_text.txt
```

**Expected:** >0 lines

**If Empty:** Session 315 has very low engagement (8.1 seconds), consider excluding

---

### Quality Flag Concern

**Check:**
```bash
cat /home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/session_632/quality_flag.json
```

**Action:** Review `reason` field, decide to include/exclude

---

## 📞 Contact & Support

**Phase 1 Documentation:**
- Comprehensive Report: `docs/reports/Phase1_Comprehensive_Report.md`
- Validation Charts: `docs/reports/*.png` (5 figures)
- Git Commit: 329fb25

**Questions about Outputs:**
- Check `validation_summary.csv` first
- Review session-specific `roles.json` for confidence
- Consult Phase 1 technical report

---

**Document Version:** 1.0  
**Last Updated:** February 5, 2026  
**Status:** Phase 1 Complete, Ready for Phase 2 ✅
