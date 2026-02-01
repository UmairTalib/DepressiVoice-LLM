# Phase 1: Integration & Cleanup Documentation

## Overview
This document details the final directory structure after the "Phase 1 Integration & Cleanup" event.
The project has been reorganized to support Phase 2 development while preserving Phase 1 reproducibility.

## Directory Structure
- **`src/phase1_pipeline/`**: Core scientific logic for Phase 1 (e.g., `scientific_validation.py`, `acoustic_verification.py`).
- **`src/phase1_scripts/`**: One-off execution scripts used for cleaning, rescuing, and auditing (e.g., `micro_cleanup_robust.py`, `ultra_rescue_309.py`).
- **`src/reporting/`**: Scripts used to generate the PDFs and HTML dossiers.
- **`src/phase2_scaffold/`**: New architecture for Multimodal Depression Detection (MentalRoBERTa + WavLM).
- **`data/phase1_outputs/`**: Synced copy of the final Phase 1 dataset (Metadata/Transcripts/Reports). NOTE: Large WAV files remain on the cluster.

## Cluster Management
- **`scripts/cluster_cleanup_phase1.sh`**: Run this on the cluster to archive old logs and cleanup the root.
- **`scripts/sync_phase1_data.sh`**: Run this locally to download the latest Phase 1 dataset from the cluster.

## Environment Setup
- **`setup_project.sh`**: ONE-CLICK script to create the local `depressivoice-env` conda environment using the updated `requirements.txt`.

## Phase 2 Usage
To start working on Phase 2, verify the environment:
```bash
./setup_project.sh
conda activate depressivoice-env
```
Then proceed to `src/phase2_scaffold/` implementation.
