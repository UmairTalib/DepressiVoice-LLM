#!/bin/bash
#SBATCH --job-name=eval_fusion_v2
#SBATCH --output=experiments/fusion_v2/eval_%j.out
#SBATCH --error=experiments/fusion_v2/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Load environment
module load miniconda3/4.9.2
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $CONDA_PREFIX/etc/profile.d/conda.sh 2>/dev/null
conda activate depressivoice-phase1

echo "🚀 Starting Evaluation"
echo "📅 Date: $(date)"

# Install missing dependencies (if internet is available)
pip install scikit-learn seaborn matplotlib || echo "Pip install failed, hoping packages exist..."

# Run Evaluation
python src/phase3_fusion/evaluate_detailed.py \
    --model_path experiments/fusion_v2/best_fusion_model.pt \
    --output_dir experiments/fusion_v2/evaluation_results

echo "✅ Evaluation Complete"
