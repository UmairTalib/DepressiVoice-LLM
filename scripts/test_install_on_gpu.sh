#!/bin/bash
#SBATCH --job-name=gpu_install_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_test_%j.log

# This script runs ON THE GPU NODE.
# It tests if we can create an environment and install packages here.

echo "======================================================"
echo "Job running on node: $(hostname)"
echo "Starting Installation Test on GPU Node"
echo "======================================================"

# 1. Load correct Python module (Adjust name if needed, e.g. python/3.10)
module load python/3.11.2

# 2. Create a fresh venv in the CURRENT folder (wherever you submitted from)
# This ensures it's in your project space, which usually sticks around.
echo "Creating virtual environment 'venv_gpu_test'..."
python3 -m venv venv_gpu_test
source venv_gpu_test/bin/activate

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install dependencies (This tests Internet Access on Compute Node)
echo "Installing PyTorch (GPU)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing WhisperX..."
pip install git+https://github.com/m-bain/whisperx.git

# 5. Verify Installation
echo "======================================================"
echo "VERIFICATION:"
echo "======================================================"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
python -c "import whisperx; print('WhisperX Success!')"

echo "======================================================"
echo "Test Complete."
