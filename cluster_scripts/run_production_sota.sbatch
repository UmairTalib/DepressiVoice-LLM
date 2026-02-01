#!/bin/bash
#SBATCH --job-name=DepressiPhase2_SOTA_FINAL
#SBATCH --output=/home/g755389/depressivoice_phase2_outputs/logs/sota_%A_%a.out
#SBATCH --error=/home/g755389/depressivoice_phase2_outputs/logs/sota_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-274%10

# Load environment logic - ROBUST FIX
module load miniconda3/4.9.2
# Use explicit source instead of conda activate to avoid shell init issues
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $CONDA_PREFIX/etc/profile.d/conda.sh 2>/dev/null
conda activate depressivoice-phase1

# Environment Variables
export HF_TOKEN="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"
export TORCHAUDIO_BACKEND="soundfile"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Directories
PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
OUTPUT_DIR="/home/g755389/depressivoice_phase2_outputs"
SCRIPT_PATH="$OUTPUT_DIR/phase2_pyannote_sota.py"

# Get session ID from array index
mapfile -t UPDATED_SESSIONS < <(ls $PROJECT_ROOT/data/01_raw/*_AUDIO.wav | xargs -n 1 basename | sed 's/_AUDIO.wav//' | sort)
SESSION_ID=${UPDATED_SESSIONS[$SLURM_ARRAY_TASK_ID]}

echo "Running Session: $SESSION_ID"
python $SCRIPT_PATH --session_id "$SESSION_ID" --base_dir "$PROJECT_ROOT" --output_dir "$OUTPUT_DIR"
