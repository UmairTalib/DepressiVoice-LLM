# Phase 3: Text Features & Multimodal Fusion

This phase focuses on extracting text embeddings from the transcripts and building the fusion model that combines audio (wavlm) and text (roberta) features to predict depression.

## 1. Text Feature Extraction
- [ ] Create `src/features/extract_text_features.py`
    - [ ] Load `transcripts_whisperx_labeled.csv`
    - [ ] Filter for 'Patient' speech per session
    - [ ] Concatenate diverse patient utterances (or keep sequence?) -> Let's keep sequence to match audio windows or just document-level.
    - [ ] *Plan*: Since audio is time-series, text should ideally be aligned or we pool both. Common approach: Extract sentence-level embeddings first.
    - [ ] Model: `roberta-base` or `mental-bert`? Let's start with `roberta-base`.
    - [ ] Save to `data/03_features/text_embeddings/` as `.pt` files.

## 2. Dataset & Dataloader
- [ ] Create `src/data/multimodal_dataset.py`
    - [ ] Custom PyTorch Dataset that loads:
        - Audio .pt (from Phase 2)
        - Text .pt (from Step 1)
        - Labels (PHQ-8 score and binary label) using `metadata_mapped.csv`
    - [ ] Implement padding/collate function to handle variable lengths.

## 3. Fusion Model Architecture
- [ ] Create `src/models/fusion_network.py`
    - [ ] **Audio Branch**: LSTM or Transformer Encoder on top of WavLM features.
    - [ ] **Text Branch**: Transformer Encoder on text embeddings.
    - [ ] **Fusion Layer**: Concatenate outputs + Dense Layers.
    - [ ] **Heads**:
        - Regression Head (PHQ Score)
        - Classification Head (Depressed vs Healthy)

## 4. Training Loop
- [ ] Create `src/train_fusion.py`
    - [ ] Training loop with Validation
    - [ ] Loss function (MSE for score, BCE for binary)
    - [ ] Logging (WandB or simple CSV logs)
    - [ ] Model Checkpointing

## 5. Cluster Execution
- [ ] Create `scripts/train_fusion.sb`
    - [ ] GPU job script for training on the cluster.
