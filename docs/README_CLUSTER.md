# Running DepressiVoice-LLM on Omni Cluster (Uni Siegen)

The Omni cluster (and many HPC systems) often does not support Docker directly for security reasons. Instead, they use **Apptainer** (formerly Singularity). 

This guide explains how to convert the Docker solution to detailed steps for the cluster.

## 1. Setup on Local Machine (or where Docker is available)

First, you need to build the Docker image and save it, or push it to a registry (like Docker Hub).
**Option A: Push to Docker Hub (Easiest)**
1.  Create a Docker Hub account.
2.  Login: `docker login`
3.  Build and push:
    ```bash
    docker build -t your-username/depressivoice-llm:latest .
    docker push your-username/depressivoice-llm:latest
    ```

**Option B: Save as SIF file (Advanced)**
If you have `apptainer` or `singularity` installed locally:
```bash
apptainer build depressivoice.sif docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime --sandbox
# This is complex to do purely locally without linux. Option A is recommended for Mac users.
```

## 2. On Omni Cluster

### 2.1 Pull the Image
Login to the cluster and pull the image from Docker Hub. Apptainer/Singularity can interpret Docker images.

```bash
# Load apptainer/singularity module (check exact name with 'module avail')
module load apptainer
# OR if your cluster uses the specific singularity module:
# module load singularity

# Pull the image - this creates a .sif file
# You can use 'singularity' instead of 'apptainer' if that is the installed command
apptainer pull depressivoice.sif docker://umi002/depressivoice-llm:latest
# OR: singularity pull depressivoice.sif docker://umi002/depressivoice-llm:latest
```

### 2.2 Prepare Data
Ensure your audio data (`data/01_raw`) and config (`config.yaml`) are available on the cluster storage.

### 2.3 Run with Slurm
Create a submission script, e.g., `run_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=transcribe_audio
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Load necessary modules
module load apptainer

# Create output directories if they don't exist
mkdir -p data/02_transcribed logs

# Run the container
# --nv is crucial! It enables GPU access inside the container
# --bind mounts your cluster directories into the container
# Pass HF_TOKEN environment variable for diarization (if needed)
export HF_TOKEN="your_hugging_face_token_here"

apptainer run --nv \
    --env HF_TOKEN=$HF_TOKEN \
    --bind $(pwd)/data:/app/data \
    depressivoice.sif python src/data_processing.py

# OR if using 'singularity' command:
# singularity run --nv \
#     --env HF_TOKEN=$HF_TOKEN \
#     --bind $(pwd)/data:/app/data \
#     depressivoice.sif python src/data_processing.py
```

### 2.4 Important: Hugging Face Token for Diarization
To use **WhisperX Diarization** (Pyannote), you must:
1.  Have a Hugging Face account.
2.  Accept user conditions for `pyannote/speaker-diarization-3.1` on Hugging Face.
3.  Generate an Access Token (Read).
4.  Either put it in `config.yaml` or export it as `HF_TOKEN` before running the container.


### 2.4 Submit Job
```bash
sbatch run_job.sh
```

## Troubleshooting
- **Libraries**: If you run into "library not found" errors, the Dockerfile ensures `ffmpeg` and others are inside the image, isolated from the cluster's old libraries.
- **CUDA**: The `--nv` flag is essential for Apptainer to pass the host GPU drivers to the container.
- **Paths**: Ensure `config.yaml` points to `/app/data/...` or relative paths that match the valid structure inside the container.
