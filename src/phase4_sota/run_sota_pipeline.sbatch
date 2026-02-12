#!/bin/bash
#SBATCH --job-name=phase4_sota
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --error=sota_%j.err
#SBATCH --output=sota_%j.out

# Setup Environment
module load miniconda3/4.9.2
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate depressivoice-phase1

# Environment Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "🚀 Starting Phase 4 SOTA Pipeline (v11-PRO)"
echo "📅 Date: $(date)"
echo "🖥️ Node: $SLURMD_NODENAME"

# 1. SOTA Feature Extraction
echo "-----------------------------------"
echo "🛠 Step 1: SOTA Feature Extraction"
echo "Using FusionModelMax (Phenotype-Anchored)..."

python -m src.phase4_sota.extract_features \
    --checkpoint experiments/v11_PRO/best_model_max.pt \
    --data_root . \
    --output_dir data/phase4_sota_features

if [ $? -ne 0 ]; then
    echo "❌ SOTA Extraction Failed!"
    exit 1
fi

# 2. Phase 4 SOTA Training
echo "-----------------------------------"
echo "🧠 Step 2: Training SOTA Temporal Model"

python -m src.phase4_sota.train \
    --data_dir data/phase4_sota_features \
    --output_dir experiments/phase4_v11_PRO_SOTA \
    --epochs 50 \
    --model_type transformer

if [ $? -ne 0 ]; then
    echo "❌ SOTA Training Failed!"
    exit 1
fi

echo "✅ Phase 4 SOTA Pipeline Complete."
