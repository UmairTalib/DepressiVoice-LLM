#!/bin/bash
#SBATCH --job-name=whisperx_parallel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/parallel_%A_%a.log
#SBATCH --array=0-9

# ==============================================================================
# PARALLEL EXECUTION CONFIG
# ==============================================================================
# We use --array=0-9 above, which means 10 jobs will run.
# This fits well with the 10 GPU nodes (some have more, but 10 is a solid start).
NUM_SHARDS=10

# Hugging Face Token
export HF_TOKEN="<YOUR_HF_TOKEN>"
export HF_HOME=$(pwd)/.cache

echo "======================================================"
echo "Job Cluster ID: $SLURM_ARRAY_JOB_ID"
echo "Job Array Index: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Processing Shard: $((SLURM_ARRAY_TASK_ID + 1)) / $NUM_SHARDS"
echo "======================================================"

# Load Python
module load python/3.11.2
# FFmpeg is now installed in venv, no module load needed

# Activate Environment
source venv_gpu_test/bin/activate

# Fix: Ensure libs are found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/venv_gpu_test/lib/python3.11/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/venv_gpu_test/lib/python3.11/site-packages/nvidia/cublas/lib

# Run the Python script with Sharding arguments
python src/data_processing.py \
    --shard-index $SLURM_ARRAY_TASK_ID \
    --num-shards $NUM_SHARDS

echo "======================================================"
echo "Shard Complete."
