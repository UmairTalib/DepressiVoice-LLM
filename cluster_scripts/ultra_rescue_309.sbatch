#!/bin/bash
#SBATCH --job-name=UltraRescue_309
#SBATCH --output=/home/g755389/depressivoice_phase2_outputs/logs/ultra_rescue_309.out
#SBATCH --error=/home/g755389/depressivoice_phase2_outputs/logs/ultra_rescue_309.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# 1. Activate Environment
source /cm/shared/omni/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/g755389/.conda/envs/depressivoice-phase1

# 2. Run the Ultra Rescue Script
echo "Starting Force-Split Rescue for Session 309..."
python /home/g755389/depressivoice_phase2_outputs/ultra_rescue_309.py

echo "Job Finished."
