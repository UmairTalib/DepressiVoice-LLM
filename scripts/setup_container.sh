#!/bin/bash
#SBATCH --job-name=setup_container
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=logs/setup_container_%j.out

mkdir -p logs

echo "======================================================"
echo "Pulling PyTorch Singularity Container"
echo "======================================================"

module load singularity/3.7.1

# Define Image Path
IMAGE_NAME="depressivoice_torch.sif"
EXPORT_PATH="$HOME/$IMAGE_NAME"

# Pull from Docker Hub
# PyTorch 2.1.0 with CUDA 11.8 (Matches cluster driver)
singularity pull --force $EXPORT_PATH docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

echo "Container pulled to: $EXPORT_PATH"
