#!/bin/bash
#SBATCH --job-name=rebuild_venv
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=logs/rebuild_venv_%j.out

# Ensure logs directory exists
mkdir -p logs

echo "======================================================"
echo "Cluster Environment Repair: Rebuilding Venv with --copies"
echo "======================================================"

# 1. Clean Slate: SKIP DELETION (Too slow). Just use new name.
# rm -rf venv_gpu_test

# 2. Load Modules (Use only GpuModules as base)
module purge
module load GpuModules

# Check python version
which python
python --version

# 3. Create NEW Environment with --copies
echo "[2/4] Creating new venv: venv_gpu_final with --copies"
/cm/shared/omni/apps/python/3.11.2/bin/python3.11 -m venv venv_gpu_final --copies

# 4. Activate & Install
echo "[3/4] Installing dependencies..."
source venv_gpu_final/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements (ensure requirements.txt exists)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Apply our specific fixes (Numba/PyTorch versions)
# These are identical to run_setup.sh
echo "[4/4] Applying compatibility fixes..."
pip install "numpy<2.4"
pip install ctranslate2==4.4.0 --force-reinstall
pip install torch torchaudio

echo "======================================================"
echo "Environment Rebuild COMPLETE"
echo "You can now run submit_extraction.sb"
echo "======================================================"
