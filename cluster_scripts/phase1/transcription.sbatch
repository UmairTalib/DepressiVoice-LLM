#!/bin/bash
#SBATCH --job-name=Phase1_Transcription
#SBATCH --output=/home/g755389/DepressiVoice-LLM/logs/transcription_%A_%a.out
#SBATCH --error=/home/g755389/DepressiVoice-LLM/logs/transcription_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-274%20

# Load environment
module load miniconda3/4.9.2 2>/dev/null || true
source /cm/shared/omni/apps/miniconda3/etc/profile.d/conda.sh
conda activate depressivoice-phase1

# Environment Variables
export TORCHAUDIO_BACKEND="soundfile"

# Directories
PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
OUTPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
LOG_DIR="$PROJECT_ROOT/logs"
SCRIPT_PATH="$PROJECT_ROOT/src/phase1_transcription_rerun.py"

# Get session ID from array index (match diarization order)
MASTER_AUDIO_DIR="/home/g755389/depressivoice_master_audio"
mapfile -t SESSION_LIST < <(ls $MASTER_AUDIO_DIR/*_AUDIO.wav | xargs -n 1 basename | sed 's/_AUDIO.wav//' | sort)
SESSION_ID=${SESSION_LIST[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Phase 1 Transcription"
echo "Session ID: $SESSION_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID / 274"
echo "Output Dir: $OUTPUT_DIR/session_$SESSION_ID"
echo "========================================="

# Run transcription python script
python $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --model "large-v3"

echo "Session $SESSION_ID transcription completed with exit code: $?"
