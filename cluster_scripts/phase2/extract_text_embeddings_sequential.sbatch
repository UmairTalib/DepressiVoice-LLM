#!/bin/bash
#SBATCH --job-name=mental_roberta_seq
#SBATCH --output=logs/text_seq_extract_%j.log
#SBATCH --error=logs/text_seq_extract_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

# Load Python module as per handover
module load python/3.11.2

# Ensure output directories exist
mkdir -p data/03_features/text_embeddings_seq
mkdir -p logs

# Run extraction (Sequential Segment Mode)
echo "Starting Sequential MentalRoBERTa extraction at $(date)"
python3 src/extraction/text/mental_roberta_extractor.py \
    --input_root data/Phase1_results/ \
    --output_dir data/03_features/text_embeddings_seq/ \
    --mode sequential \
    --hf_token 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
echo "Finished extraction at $(date)"
