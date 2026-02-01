#!/bin/bash
#SBATCH --job-name=setup_whisperx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.log

# FAIL FAST
set -e

echo "======================================================"
echo "Starting Environment Setup on GPU Node (v3 - Robust)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================"

# 1. Load Python
module load python/3.11.2

# 2. Create Virtual Environment
if [ ! -d "venv_gpu_test" ]; then
    echo "Creating virtual environment 'venv_gpu_test'..."
    python3 -m venv venv_gpu_test
else
    echo "Virtual environment exists."
fi

# 3. Activate
source venv_gpu_test/bin/activate

# 3.5 Install FFmpeg (Static Build)
if [ ! -f "venv_gpu_test/bin/ffmpeg" ]; then
    echo "Downloading and Installing FFmpeg..."
    # Use wget with retry
    wget -q --tries=5 https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xf ffmpeg-release-amd64-static.tar.xz
    mv ffmpeg-*-amd64-static/ffmpeg venv_gpu_test/bin/
    mv ffmpeg-*-amd64-static/ffprobe venv_gpu_test/bin/
    rm -rf ffmpeg-*-amd64-static*
fi

# 4. OPTIMIZED PIP INSTALLATION
# Use --no-cache-dir to avoid NFS I/O bottlenecks causing timeouts
# Use --retries to handle transient network issues

echo "Upgrading Pip..."
pip install --no-cache-dir --upgrade pip setuptools wheel

echo "Installing PyTorch 2.4.1 (CUDA 12.1)..."
pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing WhisperX dependencies..."
pip install --no-cache-dir faster-whisper==1.0.3 ctranslate2==4.4.0

echo "Installing WhisperX..."
pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git --no-deps

echo "Installing requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo "Final fixes..."
pip install --no-cache-dir "numpy<2.4"

echo "======================================================"
echo "Verifying Installation..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo "SETUP COMPLETE."

