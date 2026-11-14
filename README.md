# DepressiVoice-LLM

**Screening for elevated depressive symptoms from clinical interview recordings, using speech and text**

A research project in the module Project Work at the Chair for Embedded Systems, University of Siegen (April 2025 to March 2026), carried out by Umair Talib with one co-author. The diarization, transcription and feature extraction pipeline and its window-level results are described in a paper at CoDIT 2026, published by IEEE (see [Publication](#publication)).

> **Research prototype, not a medical device.** The code was written for a university project and is not clinically validated. The target label is a PHQ-8 score of 10 or more, a common screening cut-off for elevated depressive symptoms. It is not a diagnosis, and the model must not be used to diagnose anyone.

---

## Results at a glance

| What was measured | Result | Source |
| --- | --- | --- |
| Interviewer contamination in participant transcripts, before and after the text filter | 14.18% → 1.09% | Paper |
| Diarization error rate (DER) and word error rate (WER) | 4.2% and 12.0% | Paper |
| Window-level ROC-AUC on the validation set in the Phase 3 ablation: fusion, audio only, text only | 0.659, 0.633, 0.503 | Paper, [Phase 3 report](docs/reports/PHASE3_MULTIMODAL_FUSION_REPORT.pdf) |
| Session-level ROC-AUC on the held-out test set (56 participants: 17 with PHQ-8 of 10 or more, 39 below) | 0.614 (95% CI 0.454 to 0.771) | [Phase 4 report](docs/reports/PHASE4_TECHNICAL_REPORT.pdf) |
| F1 on the same test set | 0.566 (95% CI 0.385 to 0.712) | Phase 4 report |
| Recall, precision and specificity on the same test set, at a threshold of 0.160 chosen on the validation set | 88.2% (15 of 17), 41.7% (15 of 36), 46.2% (18 of 39) | Phase 4 report |

How to read these numbers:

- The text filter lowered interviewer contamination to 1.09%. It did not remove interviewer speech completely.
- The 95% confidence interval of the test ROC-AUC includes 0.5. With 56 test participants, the test cannot rule out chance-level ranking.
- The model flags most participants above the cut-off, but also 21 of the 39 below it. Any screening use would need a follow-up assessment by a clinician.
- Text alone scores near chance at the window level (0.503), probably because an 8-second window holds only a few words.

---

## Pipeline

| Phase | What it does | Documentation |
| --- | --- | --- |
| 1 | Speaker diarization with pyannote.audio 3.1, interviewer identification with a voiceprint (biometric anchor), participant selection by speaking time, transcription with Faster-Whisper large-v3, and a text filter that drops participant segments containing known interviewer prompts | [Phase 1 guide](docs/README_PHASE1.md), [Phase 1 report](docs/reports/Phase1_Technical_Report.pdf) |
| 2 | Audio embeddings with WavLM-Large and text embeddings with Mental-RoBERTa, on 8-second windows | [Phase 2 guide](docs/README_PHASE2.md), [Phase 2 report](docs/reports/Phase2_Technical_Report.pdf) |
| 3 | Window-level gated multimodal fusion (GMU) with multi-task learning on the PHQ-8 items | [Phase 3 report](docs/reports/PHASE3_MULTIMODAL_FUSION_REPORT.pdf) |
| 4 | Session-level cross-modal Transformer that aggregates the windows of one interview | [Phase 4 report](docs/reports/PHASE4_TECHNICAL_REPORT.pdf) |
| 5 | Text explanations generated with the Gemini API from the attention-based evidence of the Phase 4 model | [Phase 5 report](docs/reports/PHASE5_REPORT_FINAL.pdf) |
| 6 | Streamlit research dashboard to browse predictions, gate values and explanations | [`src/phase6_dashboard/`](src/phase6_dashboard/) |

Transcribing the 34.1 hours of interview audio took about 2.5 hours on 10 GPU nodes of the University of Siegen HPC cluster, scheduled with SLURM.

---

## Repository layout

```text
src/
  phase1/             diarization and transcription
  phase2_audio/       WavLM audio features
  phase2_text/        Mental-RoBERTa text embeddings
  phase3/             window-level gated fusion (final version)
  phase4_temporal/    session-level Transformer
  phase5_llm/         LLM text explanations
  phase6_dashboard/   Streamlit dashboard
  validation/         validation scripts
scripts/, cluster_scripts/   SLURM jobs and helper scripts
docs/                 phase guides, handover notes and technical reports (docs/reports/)
```

Other folders under `src/` and `scripts/`, such as `phase3_legacy/` and `phase4_sota/`, hold earlier iterations.

---

## Getting started

```bash
git clone https://github.com/UmairTalib/DepressiVoice-LLM.git
cd DepressiVoice-LLM
conda create -n depressivoice python=3.10 -y
conda activate depressivoice
pip install -r requirements.txt
```

Phase 1 reads a Hugging Face access token from the `HF_TOKEN` environment variable, and Phase 5 reads a Gemini API key from `GEMINI_API_KEY`. Keep keys out of the code. The phase guides list the exact package versions and the SLURM jobs: [Phase 1](docs/README_PHASE1.md) and [Phase 2](docs/README_PHASE2.md).

---

## Data

The project uses the Extended Distress Analysis Interview Corpus (E-DAIC), which USC ICT distributes to academic and non-profit researchers under an agreement. The audio and transcripts are not part of this repository. To rerun the pipeline, you need your own access to the corpus.

---

## Publication

U. Oleh, U. Talib, M. A. Alam, and R. Obermaisser, "Leakage-Resilient Diarization and Feature Extraction Framework for Robust Multimodal Mental Health Corpora," in *Proc. 12th Int. Conf. Control, Decision and Information Technologies (CoDIT)*, Bari, Italy, Jul. 2026, pp. 2508–2512, doi: [10.1109/CoDIT70676.2026.11630977](https://doi.org/10.1109/CoDIT70676.2026.11630977).

---

**Status:** all six phases were completed within the project (April 2025 to March 2026). README updated in November 2026.
