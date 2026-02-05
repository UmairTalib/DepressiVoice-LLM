#!/bin/bash
#SBATCH --job-name=WavLM_Audio_Only
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load Environment
module load python/3.11.2
# Or use the specific condo/venv you have set up previously

# Set paths
PROJECT_ROOT="/home/g755389/DepressiVoice-LLM"
DATA_DIR="$PROJECT_ROOT/data/Phase1_results"
LABELS_CSV="$PROJECT_ROOT/data/02_transcribed/transcripts_whisperx_labeled.csv"
OUTPUT_DIR="$PROJECT_ROOT/data/04_results/checkpoints_audio_only"

mkdir -p $OUTPUT_DIR

# Env variables for performance/robustness
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# Move to the scaffold directory
cd $PROJECT_ROOT/src/phase2_scaffold

# Run Training
python -u train_audio.py \
    --data_dir "$DATA_DIR" \
    --labels_csv "$LABELS_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --epochs 10 \
    --lr 5e-5 \
    --save_interval 1
