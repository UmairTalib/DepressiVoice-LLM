#!/bin/bash
set -e

# DepressiVoice-LLM Setup Script
# Usage: ./setup_project.sh

ENV_NAME="depressivoice-env"
PYTHON_VERSION="3.10"

echo "==========================================="
echo "   DepressiVoice-LLM Environment Setup"
echo "==========================================="

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH."
    echo "   Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✅ Conda detected."

# 2. Check if Env Exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "   Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        conda remove --name $ENV_NAME --all -y
    else
        echo "   Skipping creation. Updating dependencies..."
    fi
else
    # Create Env
    echo "🚀 Creating Conda environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
fi

# 3. Activate and Install
echo "📦 Installing Dependencies..."
# Note: We use 'eval' to initialize conda in script
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install foundational packages via conda (faster/safer for ds)
conda install -y numpy pandas scikit-learn matplotlib seaborn

# Install pip packages
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

echo "==========================================="
echo "✅ Setup Complete!"
echo "   To start working, run:"
echo "   conda activate $ENV_NAME"
echo "==========================================="
