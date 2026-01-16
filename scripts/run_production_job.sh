#!/bin/bash
#SBATCH --job-name=whisperx_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pipeline_%j.log

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. Hugging Face Token (REQUIRED for Diarization)
#    Replace 'hf_...' with your actual token OR ensure .env file exists.
export HF_TOKEN="<YOUR_HF_TOKEN>"

# Ensure cache is in a writable location
export HF_HOME=$(pwd)/.cache

echo "======================================================"
echo "Job running on node: $(hostname)"
echo "Date: $(date)"
echo "======================================================"

# 2. Load Python (Must match the one used in creation)
module load python/3.11.2

# 3. Activate the Verified Environment
#    We are reusing 'venv_gpu_test' because we KNOW it works.
if [ -d "venv_gpu_test" ]; then
    echo "Activating existing verified environment: venv_gpu_test"
    source venv_gpu_test/bin/activate
else
    echo "Error: venv_gpu_test not found! Did you run the test script?"
    exit 1
fi

# 4. Install Remaining Dependencies
#    (Librosa, DotEnv, etc. that weren't in the test script)
echo "Installing remaining project requirements..."
pip install -r requirements.txt

# 5. Run the Pipeline
echo "======================================================"
echo "Starting Transcription Pipeline..."
echo "======================================================"

# Run the python script
python src/data_processing.py

echo "======================================================"
echo "Job Complete."
