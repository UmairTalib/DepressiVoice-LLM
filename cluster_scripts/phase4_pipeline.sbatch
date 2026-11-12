#!/bin/bash
#SBATCH --job-name=phase4_temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --error=phase4_%j.err
#SBATCH --output=phase4_%j.out

# Setup Environment
module load miniconda3/4.9.2
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate depressivoice-phase1

# Environment Variables
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"
export TORCHAUDIO_BACKEND="soundfile"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "🚀 Starting Phase 4 Pipeline"
echo "📅 Date: $(date)"
echo "🖥️ Node: $SLURMD_NODENAME"

# 1. Feature Extraction (The Bridge)
echo "-----------------------------------"
echo "🛠 Step 1: Feature Extraction"
echo "Extracting fused features from Phase 3.5 model..."

python -m src.phase4_temporal.extract_features \
    --checkpoint experiments/fusion_v11_handover/best_fusion_v11.pth \
    --data_root . \
    --output_dir data/phase4_features

if [ $? -ne 0 ]; then
    echo "❌ Feature Extraction Failed!"
    exit 1
fi

# 2. Phase 4 Training
echo "-----------------------------------"
echo "🧠 Step 2: Training Temporal Model"

python -m src.phase4_temporal.train \
    --data_dir data/phase4_features \
    --output_dir experiments/phase4_temporal \
    --epochs 50

if [ $? -ne 0 ]; then
    echo "❌ Training Failed!"
    exit 1
fi

echo "✅ Phase 4 Pipeline Complete."
