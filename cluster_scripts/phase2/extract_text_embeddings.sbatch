#!/bin/bash
#SBATCH --job-name=MentalRoBERTa_Phase2B
#SBATCH --output=logs/phase2b_text_%A_%a.out
#SBATCH --error=logs/phase2b_text_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-274%20

# Phase 2B: Mental-RoBERTa Text Embedding Extraction
# Extracts 768-dim embeddings with 8-second windowing (matches audio)

module load python/3.11.2

PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
INPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
OUTPUT_DIR="$PROJECT_ROOT/data/phase2_clean/text_embeddings_clean"
SCRIPT_PATH="$PROJECT_ROOT/src/phase2_text/extract_embeddings.py"

# Valid HF Token for MentalRoBERTa (gated model)
export HF_TOKEN="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"

mkdir -p logs
mkdir -p $OUTPUT_DIR

# Get session ID from array
mapfile -t SESSIONS < <(ls $INPUT_DIR | grep 'session_' | sed 's/session_//' | sort -n)
SESSION_ID=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Phase 2B Text Extraction"
echo "========================================"
echo "Session ID: $SESSION_ID"
echo "Input: $INPUT_DIR/session_$SESSION_ID/cleaned_patient_text_v2.json"
echo "Output: $OUTPUT_DIR/${SESSION_ID}_embeddings.pt"
echo "========================================"

# Set Hugging Face cache to avoid home directory quota issues
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence_transformers"

# Run extraction
python -u $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --verbose

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Extraction complete for session $SESSION_ID"
else
    echo "✗ Extraction failed for session $SESSION_ID (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
