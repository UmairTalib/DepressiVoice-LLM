#!/bin/bash
#SBATCH --job-name=test_load
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_load_%j.log

module load python/3.11.2
# FFmpeg is in venv
source venv_gpu_test/bin/activate

echo "Running Model Load Test..."
python src/test_model_load.py
