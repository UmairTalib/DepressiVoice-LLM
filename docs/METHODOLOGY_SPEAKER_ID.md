# Methodology: Robust Speaker Role Assignment via Biometric Anchoring

**Date:** January 28, 2026
**Project:** DepressiVoice-LLM
**Status:** Implemented (Phase 2)

---

## 1. Executive Summary
Traditional "blind" diarization (A/B labeling) fails in clinical interview settings because it lacks semantic awareness of roles. Our initial analysis revealed widespread **Label Inversion** (15-50% of sessions), where the Patient was mislabeled as the Interviewer.

To resolve this, we implemented a **Biometric Speaker Identification System** (effectively "FaceID for Voice"). Instead of guessing roles based on speaking duration or turn-taking, we mathematically verify which speaker matches the known voiceprint of the virtual interviewer ("Ellie").

This reframes role assignment as a **speaker verification problem** rather than a clustering or heuristic classification problem.

---

## 2. The Problem: "Blind" Labeling Failure
WhisperX and Pyannote separate audio into `SPEAKER_00` and `SPEAKER_01` effectively, but the assignment of these labels is **arbitrary**.

### Why Heuristics Fail
*   **"Interviewer Speaks First":** Fails if the recording starts mid-sentence or if the patient acknowledges a prompt immediately.
*   **"Patient Speaks Most":** Fails in short sessions or when patients are withdrawn/monosyllabic (common in depression).
*   **"Interviewer Asks Questions":** Fails when patients ask clarifying questions ("What do you mean?").

**Consequence:** A model trained on inverted labels learns the *Interviewer's* voice features as "Depressed/Healthy," rendering the analysis invalid.

---

## 3. The Solution: Biometric Anchoring

### Core Concept
Since the interviewer ("Ellie") is a consistent virtual agent across all 275 sessions, her voice marks a **fixed point** in the speaker embedding space. Patients, conversely, exhibit high inter-speaker variability.

We therefore treat role assignment as a **verification task**, not a **diarization task**.

### The Algorithm

#### Phase A: Anchor Generation (The "Fingerprint")
1.  **Selection:** Identify a clean, canonical sample of Ellie's speech (e.g., *Participant 300, 36s-40s*).
2.  **Extraction:** Pass this audio through a speaker embedding model (`pyannote/wespeaker-voxceleb-resnet34-LM`).
3.  **Vectorization:** The model outputs a 256-dimensional vector $\vec{A}$ representing Ellie's vocal identity.
4.  **Storage:** This vector is saved as `data/ellie_anchor.npy`.

#### Phase B: Session Processing (The "Match")
For each session $S$ with speakers $\{X, Y\}$:
1.  **Embedding:** Compute the mean embedding vector for Speaker $X$ ($\vec{v}_X$) and Speaker $Y$ ($\vec{v}_Y$).
2.  **Similarity:** Compute cosine similarity to the anchor:
    $$Sim(X) = \frac{\vec{v}_X \cdot \vec{A}}{||\vec{v}_X|| \times ||\vec{A}||}$$
3.  **Assignment:**
    *   If $Sim(X) > Sim(Y) \implies X = \text{Interviewer}, Y = \text{Patient}$
    *   Else $\implies Y = \text{Interviewer}, X = \text{Patient}$

---

## 4. Technical Implementation

### Models Used
*   **Diarization:** `pyannote/speaker-diarization-3.1` (via WhisperX)
*   **Embedding:** `pyannote/wespeaker-voxceleb-resnet34-LM`
*   **Metric:** Cosine Similarity (after L2 normalization)

### Advantages
1.  **Robustness:** Independent of session length, speaking dominance, or dialogue structure.
2.  **Scalability:** Anchor is generated once; per-session overhead is negligible (<100ms).
3.  **Objectivity:** Eliminates heuristic thresholds and rule-based assumptions.

### Code Reference
*   **Anchor Generation:** `src/generate_anchor.py`
*   **Pipeline Logic:** `src/data_processing.py` (Lines 262+)

---

## 5. Automatic Validation Without Manual Labels
Because manual labeling of hundreds of sessions is impractical, correctness is validated using biometric self-consistency and statistical invariants.

### 5.1 Similarity Margin (Confidence Gap)
For each session:
$$\Delta = | Sim(best) - Sim(second) |$$

We track:
*   Mean and minimum $\Delta$
*   Percentage of sessions with $\Delta$ below a small threshold (e.g., 0.05)

**Interpretation:** Large $\Delta$ implies unambiguous identity assignment. Only small-$\Delta$ sessions are potentially risky.

### 5.2 Score Distribution Separation
We analyze two distributions:
1.  Similarities of assigned **Interviewers** to the anchor.
2.  Similarities of assigned **Patients** to the anchor.

**Correct operation produces:**
*   A high-similarity tight cluster for Interviewers
*   A low-similarity broad cluster for Patients
*   Strong separation indicates reliable identification.

### 5.3 Identity Invariant: Interviewer Consistency
Since Ellie is the same speaker in all sessions:
*   Embeddings of all assigned Interviewers must form a **tight cluster** (low variance).
*   Patients must form a **high-variance cluster**.
We verify this by measuring intra-class cosine similarity and variance.

### 5.4 Swap Test (Adversarial Sanity Check)
We invert all assignments and recompute average similarity to the anchor.
Correct behavior requires:
$$MeanSim(original) \gg MeanSim(swapped)$$

### 5.5 Automatic Outlier Detection
A session is flagged as low confidence if:
*   $max(Sim) < T$ (e.g., 0.55-0.6), or
*   $\Delta < \delta$ (e.g., 0.05)
These sessions can be excluded or reviewed.

---

## 6. Interpretation of Reliability
Rather than claiming absolute correctness, we report:
> **"X% of sessions are assigned with high confidence (mean $\Delta$ = ...). Y% are automatically flagged as ambiguous and excluded from training."**

This is standard practice in biometric verification systems.

---

## 7. Security and Ethics
Speaker embeddings are biometric identifiers and must be:
*   Stored securely
*   Access-controlled
*   Not released with the dataset

---

## 8. Conclusion
This methodology transforms role assignment from an error-prone unsupervised labeling problem into a **biometric identity verification problem** anchored to a known speaker.

Through automatic margin analysis, cluster consistency checks, and outlier detection, the system provides:
*   **High-confidence, self-validating role labeling**
*   **Automatic detection of problematic sessions**
*   A scalable and clinically robust preprocessing pipeline.

This ensures that downstream audio biomarkers are extracted from the correct speaker channel, which is a fundamental requirement for any clinical inference.
