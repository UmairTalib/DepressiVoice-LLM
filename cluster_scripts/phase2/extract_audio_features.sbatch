#!/bin/bash
#SBATCH --job-name=WavLM_Phase2
#SBATCH --output=logs/phase2_audio_%A_%a.out
#SBATCH --error=logs/phase2_audio_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-274%20

# Phase 2: WavLM-Large Feature Extraction (Clean Data)
# Extracts 1024-dim audio embeddings from patient-only speech

# Load Environment
module load python/3.11.2

# Paths
PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
INPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
OUTPUT_DIR="$PROJECT_ROOT/data/phase2_clean/audio_embeddings_raw"
SCRIPT_PATH="$PROJECT_ROOT/src/phase2_audio/extract_features.py"

mkdir -p logs
mkdir -p $OUTPUT_DIR

# Get session ID from directory list
mapfile -t SESSIONS < <(ls $INPUT_DIR | grep 'session_' | sed 's/session_//' | sort -n)
SESSION_ID=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "Processing Session: $SESSION_ID"
echo "Input: $INPUT_DIR/session_$SESSION_ID/patient_audio.wav"
echo "Output: $OUTPUT_DIR/${SESSION_ID}_features.pt"

# Export environment variables
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Run Extraction (WavLM-Large with weighted layer sum)
python -u $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "microsoft/wavlm-large"

echo "Extraction complete for session $SESSION_ID"
