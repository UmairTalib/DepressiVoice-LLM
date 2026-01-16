# Fast Track: Running on Omni Cluster WITHOUT Custom Docker Image

If uploading the Docker image is too slow, you can use the cluster's Python environment and install the libraries directly there. This is often faster.

## 1. Login to Cluster
SSH into the Omni cluster.

## 2. Load Python Module
Most clusters have Python pre-installed.
```bash
module load python/3.11.2
# If that fails, checking 'module avail python' gave us this version.
```

## 3. Create a Virtual Environment
Create a folder to hold your libraries so you don't need root access.
```bash
# Go to your project folder on the cluster
cd /path/to/your/project

# Create venv (Explicitly use python3 if needed)
python3 -m venv venv

# Activate it
source venv/bin/activate

# IMPORTANT: Upgrade pip immediately (System pip might be too old)
pip install --upgrade pip
```

## 4. Install Libraries
**Check your Python version first!** It must be 3.8 or newer.
```bash
python --version
```

If it says Python 3.6 or older, **STOP**. You must load a newer python module (Step 2).

Install WhisperX and dependencies directly on the cluster.
```bash
# Install PyTorch (lightweight command, checks cuda automatically usually)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX
pip install git+https://github.com/m-bain/whisperx.git

# Install other requirements
pip install -r requirements.txt
```

## 5. Run the Job
Create a new submission script `run_fast.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=transcribe_fast
#SBATCH --output=logs/output_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# Load module again
module load python/3.11.2

# Helper for HuggingFace (if cleaning up cache is needed)
export HF_HOME=$(pwd)/.cache
export HF_TOKEN="your_token_here"

# Activate environment
source venv/bin/activate

# Run script
python src/data_processing.py
```

Submit it:
```bash
sbatch run_fast.sh
```

## 6. How to Verify Installation
To check if WhisperX is successfully installed on the cluster:

1.  **Activate your environment**:
    ```bash
    source venv/bin/activate
    ```

2.  **Check with Pip**:
    ```bash
    pip list | grep whisperx
    ```

    ```bash
    python -c "import whisperx; print('WhisperX is available!')"
    ```
    If it prints "WhisperX is available!", you are good to go.

4.  **Verify GPU Support (Crucial)**:
    Since you are on a CPU login node, `torch.cuda.is_available()` will return `False` right now. That is normal.
    To check if you installed the **GPU-enabled** version correctly, run:
    ```bash
    python -c "import torch; print(f'Torch Version: {torch.__version__}')"
    ```
    It should print something like `2.1.0+cu118` (the `+cu118` implies it has CUDA support ready for the GPU node).

## 7. Submit a Test Job
Don't run the full dataset yet. Submit a test to confirm the GPU works.
Create `test_gpu.sh`:
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu_test_%j.txt

module load python/3.11.2
source venv/bin/activate

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
```
Submit with `sbatch test_gpu.sh`. Check the output file in `logs/` - it should say "True" and the name of the GPU (e.g., Tesla V100).

## Pros vs Cons
*   **Pros**: Zero upload time. Uses cluster's internet speed (extremely fast) to download libraries.
*   **Cons**: You have to run the installation commands once on the cluster terminal.
