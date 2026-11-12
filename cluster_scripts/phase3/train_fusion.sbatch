#!/bin/bash
#SBATCH --job-name=fusion_v2_reg
#SBATCH --output=experiments/fusion_v2/slurm_%j.out
#SBATCH --error=experiments/fusion_v2/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Create experiment directory
mkdir -p experiments/fusion_v2/

# Load environment logic
module load miniconda3/4.9.2
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $CONDA_PREFIX/etc/profile.d/conda.sh 2>/dev/null
conda activate depressivoice-phase1

# Environment Variables
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"
export TORCHAUDIO_BACKEND="soundfile"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "🚀 Starting Phase 3 Fusion Training: fusion_v2 (Aggressive Regularization)"
echo "📅 Date: $(date)"
echo "🖥️ Node: $SLURMD_NODENAME"

# Configuration:
# - Dropout: 0.5 (Default in code)
# - Weight Decay: 0.05 (Default in code)
# - Grad Clipping: 1.0 (Fixed in code)
# - Truncation: 5000/1000 (Default in code)
# - Batch Size: 16
python src/phase3_fusion/train.py \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir experiments/fusion_v2/ \
    --audio_dir /home/g755389/DepressiVoice-LLM/data/03_features/audio_embeddings_clean/ \
    --text_dir /home/g755389/DepressiVoice-LLM/data/03_features/text_embeddings_v2/ \
    --labels_csv /home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv

echo "✅ Training Job Complete."
