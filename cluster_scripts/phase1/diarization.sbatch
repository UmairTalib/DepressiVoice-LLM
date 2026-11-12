#!/bin/bash
#SBATCH --job-name=Phase1_Diarization_Rerun
#SBATCH --output=/home/g755389/DepressiVoice-LLM/logs/diarization_%A_%a.out
#SBATCH --error=/home/g755389/DepressiVoice-LLM/logs/diarization_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-274%10

# Load environment
module load miniconda3/4.9.2 2>/dev/null || true
source /cm/shared/omni/apps/miniconda3/etc/profile.d/conda.sh
conda activate depressivoice-phase1

# Environment Variables
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"
export TORCHAUDIO_BACKEND="soundfile"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NEW Directories for Phase 1 Rerun
MASTER_AUDIO_DIR="/home/g755389/depressivoice_master_audio"
PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
OUTPUT_DIR="$PROJECT_ROOT/phase1_rerun_outputs"
LOG_DIR="$PROJECT_ROOT/logs"
SCRIPT_PATH="$PROJECT_ROOT/src/phase1_pipeline_rerun.py"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Get session ID from array index
mapfile -t SESSION_LIST < <(ls $MASTER_AUDIO_DIR/*_AUDIO.wav | xargs -n 1 basename | sed 's/_AUDIO.wav//' | sort)
SESSION_ID=${SESSION_LIST[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "Phase 1 Diarization Rerun"
echo "Session ID: $SESSION_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID / 274"
echo "Master Audio: $MASTER_AUDIO_DIR/${SESSION_ID}_AUDIO.wav"
echo "Output Dir: $OUTPUT_DIR/session_$SESSION_ID"
echo "========================================="

# Run diarization python script
python $SCRIPT_PATH \
    --session_id "$SESSION_ID" \
    --audio_file "$MASTER_AUDIO_DIR/${SESSION_ID}_AUDIO.wav" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR"

echo "Session $SESSION_ID completed with exit code: $?"
