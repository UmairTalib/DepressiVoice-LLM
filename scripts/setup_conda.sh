#!/bin/bash
#SBATCH --job-name=setup_conda
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=logs/setup_conda_%j.out

# Ensure logs directory exists
mkdir -p logs

echo "======================================================"
echo "Cluster Environment: Setting up Conda"
echo "======================================================"

# 1. Load Conda
module purge
module load miniconda3/4.9.2

# 2. Init Conda (for shell interaction)
# We can't run 'conda init' effectively in batch. 
# We use direct path or source the startup script.
# Assuming standard path for this module.
# Try sourcing the func.
source /cm/shared/omni/apps/miniconda3/etc/profile.d/conda.sh

# 3. Create Environment (in home to save space? No, use project or home if quota allows)
# We cleared space. Let's put it in $HOME/conda_env to be safe path-wise.
ENV_PATH="$HOME/conda_env"

if [ -d "$ENV_PATH" ]; then
    echo "Environment exists. Updating..."
else
    echo "Creating NEW Conda Env at $ENV_PATH"
    conda create -p "$ENV_PATH" python=3.10 -y
fi

# 4. Activate
conda activate "$ENV_PATH"

# 5. Install Dependencies
echo "Installing pip dependencies..."
# Upgrade pip
python -m pip install --upgrade pip --no-cache-dir

# Install Critical Deps
# CUDA 11.8 compatible torch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
pip install transformers librosa soundfile numpy pandas tqdm --no-cache-dir

echo "======================================================"
echo "Conda Setup COMPLETE"
echo "======================================================"
