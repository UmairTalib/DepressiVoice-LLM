#!/bin/bash
#SBATCH --job-name=WavLM_Large_SOTA
#SBATCH --output=logs/extract_sota_%A_%a.out
#SBATCH --error=logs/extract_sota_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-274%20

# Load Environment
module load python/3.11.2

# Paths
PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
INPUT_DIR="$PROJECT_ROOT/data/Phase1_results"
OUTPUT_DIR="$PROJECT_ROOT/data/03_features/audio_embeddings_sota"
SCRIPT_PATH="$PROJECT_ROOT/src/features/extract_audio_features_sota.py"

mkdir -p logs
mkdir -p $OUTPUT_DIR

# Get session ID from directory list
mapfile -t SESSIONS < <(ls $INPUT_DIR | sort)
SESSION_ID=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "Processing Session: $SESSION_ID"

# Export vars for robustness
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# Run Extraction (WavLM-Large with Weighted Sum & Float16)
python -u $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "microsoft/wavlm-large"
