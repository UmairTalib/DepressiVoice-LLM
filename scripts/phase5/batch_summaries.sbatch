#!/bin/bash
#SBATCH --job-name=v7_summaries
#SBATCH --output=logs/v7_summary_%A_%a.out
#SBATCH --error=logs/v7_summary_%A_%a.err
#SBATCH --array=1-113%20
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

cd ~/DepressiVoice-LLM
export PYTHONPATH=$PYTHONPATH:.
export GEMINI_API_KEY="AIzaSyBcq_pMAGs3u3JV0LEOfggMYantKqQfo6k"
export PYTHONUNBUFFERED=1

# Diagnostics
echo "📍 Node: $SLURM_NODELIST"
echo "🐍 Python: $(/home/g755389/.conda/envs/depressivoice-phase1/bin/python --version)"

# Get Patient ID from list
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/output/explanations/patient_ids.txt)

echo "🚀 Starting Clinical Summary for Patient: $PID (Task $SLURM_ARRAY_TASK_ID)"

# Run Explanation (CPU Mode)
/home/g755389/.conda/envs/depressivoice-phase1/bin/python -u src/phase5_llm/extract_v7_explanations.py --patient_ids "$PID"

echo "✅ Finished Patient: $PID"
