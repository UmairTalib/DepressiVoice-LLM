#!/bin/bash
#SBATCH --job-name=FusionV10
#SBATCH --output=/home/g755389/DepressiVoice-LLM/logs/fusion_v10_%j.log
#SBATCH --error=/home/g755389/DepressiVoice-LLM/logs/fusion_v10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules
module load miniconda3/4.9.2
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $CONDA_PREFIX/etc/profile.d/conda.sh 2>/dev/null

# Activate Environment
conda activate depressivoice-phase1

cd ~/DepressiVoice-LLM/

# Ensure log dir exists
mkdir -p logs

# Run Training
echo "Starting Fusion V10 Training..."
python3 src/phase3_fusion_v10/train.py --root_dir ~/DepressiVoice-LLM/ --save_dir ./models/fusion_v10
