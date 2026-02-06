#!/bin/bash
#SBATCH --job-name=MentalRoBERTa_V2
#SBATCH --output=logs/text_v2_%A_%a.out
#SBATCH --error=logs/text_v2_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-274%20

# Phase 2B V2: Mental-RoBERTa Text Embedding Extraction (SKIP FILTERING)
# Since transcripts are patient-only, we skip RTTM/Role checks for maximum data integrity.

module load python/3.11.2

PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
INPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
OUTPUT_DIR="$PROJECT_ROOT/data/phase2_clean/text_embeddings_clean"
SCRIPT_PATH="$PROJECT_ROOT/src/phase2_text/extract_embeddings.py"

export HF_TOKEN="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"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# Get session ID from array
mapfile -t SESSIONS < <(ls $INPUT_DIR | grep 'session_' | sed 's/session_//' | sort -n)
SESSION_ID=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "Processing Session: $SESSION_ID"

# Run extraction with --skip_filtering
python -u $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --skip_filtering \
    --verbose

exit $?
