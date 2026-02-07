#!/bin/bash
#SBATCH --job-name=phase4_mtl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --error=phase4_mtl_%j.err
#SBATCH --output=phase4_mtl_%j.out

# Setup Environment
module load miniconda3/4.9.2
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate depressivoice-phase1

# Environment Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🚀 Starting Phase 4.1 MTL Training"
echo "📅 Date: $(date)"

# Phase 4 MTL Training
python -m src.phase4_temporal.train \
    --data_dir data/phase4_features \
    --output_dir experiments/phase4_mtl \
    --epochs 50 \
    --model_type transformer

if [ $? -ne 0 ]; then
    echo "❌ MTL Training Failed!"
    exit 1
fi

echo "✅ Phase 4.1 MTL Pipeline Complete."
