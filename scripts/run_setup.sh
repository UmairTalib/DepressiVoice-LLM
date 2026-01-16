#!/bin/bash
#SBATCH --job-name=setup_whisperx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup_%j.log

echo "======================================================"
echo "Starting Environment Setup on GPU Node"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "======================================================"

# 1. Load Python
module load python/3.11.2
# module load ffmpeg (Not available on cluster)

# 1.5 Install FFmpeg (Moved to Step 3.5)

# 2. Create Virtual Environment (if not exists)
if [ ! -d "venv_gpu_test" ]; then
    echo "Creating virtual environment 'venv_gpu_test'..."
    python3 -m venv venv_gpu_test
else
    echo "Virtual environment 'venv_gpu_test' already exists. Updating..."
fi

# 3. Activate
source venv_gpu_test/bin/activate

# 3.5 Install FFmpeg (Static Build) directly into venv
if [ ! -f "venv_gpu_test/bin/ffmpeg" ]; then
    echo "Downloading and Installing FFmpeg (Static)..."
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xf ffmpeg-release-amd64-static.tar.xz
    # Move strictly to the venv bin (which is in PATH)
    mv ffmpeg-*-amd64-static/ffmpeg venv_gpu_test/bin/
    mv ffmpeg-*-amd64-static/ffprobe venv_gpu_test/bin/
    # Cleanup
    rm -rf ffmpeg-*-amd64-static*
    echo "FFmpeg installed successfully."
fi

# 4. Upgrade Pip
pip install --upgrade pip


# 5. Install WhisperX (This often pulls wrong Torch)
echo "Installing WhisperX..."
pip install git+https://github.com/m-bain/whisperx.git

# 6. Install Phase 2 Dependencies (Transformers, PEFT, etc.)
echo "Installing requirements.txt..."
# Use --upgrade to ensure we get the versions specified in requirements.txt
# (especially transformers>=4.36.0 which is critical for WavLM)
pip install --upgrade -r requirements.txt

# 7. FORCE Install PyTorch 2.4.1 (CUDA 12.1) & Compatible CTranslate2
# The driver is 550.x (CUDA 12.4), so we use PyTorch 2.4.1 + cu121
echo "Overwriting with PyTorch 2.4.1 (CUDA 12.1) & Fixed CTranslate2..."
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
# Downgrade ctranslate2 to 4.4.0 to avoid weird CUDNN linking issues
pip install ctranslate2==4.4.0 --force-reinstall

# 8. Verification Check
echo "======================================================"
echo "Verifying Installation..."
nvidia-smi
python -c "import torch; import whisperx; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "======================================================"

echo "SETUP COMPLETE. You can now run 'sbatch run_parallel.sh'"
