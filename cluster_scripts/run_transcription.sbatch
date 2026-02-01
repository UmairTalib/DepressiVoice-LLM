#!/bin/bash
#SBATCH --job-name=Phase1c_Transcription
#SBATCH --output=/home/g755389/depressivoice_phase2_outputs/logs/transcription_%A_%a.out
#SBATCH --error=/home/g755389/depressivoice_phase2_outputs/logs/transcription_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-9

# 1. Activate Environment
source /cm/shared/omni/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/g755389/.conda/envs/depressivoice-phase1

# 2. Ensure Dependencies (Robustness)
pip install faster-whisper --quiet

# 3. Define Paths
SCRIPT_PATH="/home/g755389/depressivoice_phase2_outputs/phase1c_transcription.py"
INPUT_DIR="/home/g755389/depressivoice_phase2_outputs/session_outputs"

# 4. Run Transcription with Sharding
# Passes SLURM_ARRAY_TASK_ID as shard_id (0..9)
# Passes 10 as num_shards (since array=0-9)

echo "Starting Transcription Shard ${SLURM_ARRAY_TASK_ID} of 10..."
python $SCRIPT_PATH \
  --input_dir $INPUT_DIR \
  --shard_id ${SLURM_ARRAY_TASK_ID} \
  --num_shards 10

echo "Shard ${SLURM_ARRAY_TASK_ID} Complete."
