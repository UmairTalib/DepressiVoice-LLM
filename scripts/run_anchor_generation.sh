#!/bin/bash
# Biometric Anchor Generation - Cluster Execution Script

cd /home/g755389/DepressiVoice-LLM

# Load conda from correct path
source /cm/shared/omni/apps/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate depressivoice-phase1

# Export HF token
export HF_TOKEN="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"

# Run anchor generation
python src/generate_biometric_anchor.py \
    --master_audio_dir /home/g755389/depressivoice_master_audio \
    --output_file ellie_anchor.json \
    --num_reference 10

echo "Anchor generation complete. Exit code: $?"
