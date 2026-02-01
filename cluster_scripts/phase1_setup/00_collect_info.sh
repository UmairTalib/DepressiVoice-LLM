#!/bin/bash
################################################################################
# 00_collect_info.sh
# Interactive script to collect HPC cluster configuration information
# Run this ON THE CLUSTER to generate cluster_config.env
################################################################################

set -euo pipefail

echo "========================================"
echo "HPC Cluster Configuration Collector"
echo "========================================"
echo ""

# Output file
CONFIG_FILE="cluster_config.env"

# Collect information
echo "[1/7] Detecting hostname..."
HOSTNAME=$(hostname)
echo "  → $HOSTNAME"

echo ""
echo "[2/7] Detecting home directory..."
HOME_DIR="$HOME"
echo "  → $HOME_DIR"

echo ""
echo "[3/7] Searching for DepressiVoice-LLM repository..."
echo "  Searching in common locations..."
REPO_CANDIDATES=$(find "$HOME" /work/$USER /scratch/$USER /data/$USER \
  -maxdepth 4 -type d -name "DepressiVoice-LLM" 2>/dev/null || true)

if [ -z "$REPO_CANDIDATES" ]; then
  echo "  ⚠️  Repository not found automatically."
  read -p "  Enter full path to DepressiVoice-LLM: " REPO_ROOT
else
  REPO_COUNT=$(echo "$REPO_CANDIDATES" | wc -l)
  if [ "$REPO_COUNT" -eq 1 ]; then
    REPO_ROOT="$REPO_CANDIDATES"
    echo "  ✓ Found: $REPO_ROOT"
  else
    echo "  Found multiple candidates:"
    echo "$REPO_CANDIDATES" | nl
    read -p "  Enter the number of the correct repository: " REPO_NUM
    REPO_ROOT=$(echo "$REPO_CANDIDATES" | sed -n "${REPO_NUM}p")
  fi
fi

# Validate repo
if [ ! -d "$REPO_ROOT/.git" ]; then
  echo "  ⚠️  Warning: $REPO_ROOT does not appear to be a git repository"
fi

echo ""
echo "[4/7] Determining output directory..."
echo "  Checking available workspaces..."
if [ -n "${WORK:-}" ]; then
  echo "  WORK directory available: $WORK"
  OUT_ROOT_DEFAULT="$WORK/depressivoice_phase1_outputs"
elif [ -n "${SCRATCH:-}" ]; then
  echo "  SCRATCH directory available: $SCRATCH"
  OUT_ROOT_DEFAULT="$SCRATCH/depressivoice_phase1_outputs"
else
  echo "  No standard workspace found"
  OUT_ROOT_DEFAULT="$HOME/depressivoice_phase1_outputs"
fi

read -p "  Output directory [$OUT_ROOT_DEFAULT]: " OUT_ROOT
OUT_ROOT="${OUT_ROOT:-$OUT_ROOT_DEFAULT}"
echo "  → $OUT_ROOT"

echo ""
echo "[5/7] Checking SLURM partitions..."
echo "  Available partitions:"
sinfo -o "%P %a %l %D %N" | head -20
echo ""
read -p "  Which partition for GPU jobs? [gpu]: " SLURM_PARTITION
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
read -p "  Time limit for environment check job? [01:00:00]: " SLURM_TIME
SLURM_TIME="${SLURM_TIME:-01:00:00}"
read -p "  Memory for environment check job? [16G]: " SLURM_MEM
SLURM_MEM="${SLURM_MEM:-16G}"

echo ""
echo "[6/7] Checking conda availability..."
if command -v conda &>/dev/null; then
  CONDA_PATH=$(which conda)
  CONDA_VERSION=$(conda --version 2>&1)
  echo "  ✓ Conda found: $CONDA_PATH"
  echo "    Version: $CONDA_VERSION"
  CONDA_MODULE=""
else
  echo "  ⚠️  Conda not found in PATH"
  echo "  Checking modules..."
  CONDA_MODULES=$(module avail 2>&1 | grep -i conda || echo "none")
  if [ "$CONDA_MODULES" != "none" ]; then
    echo "  Available conda modules:"
    echo "$CONDA_MODULES"
    read -p "  Enter module to load (e.g., 'conda' or 'anaconda3'): " CONDA_MODULE
  else
    echo "  No conda modules found"
    read -p "  Enter full path to conda (or press Enter to install Miniconda): " CONDA_PATH
    CONDA_MODULE=""
  fi
fi

echo ""
echo "[7/7] Setting conda environment name..."
read -p "  Conda environment name [depressivoice-phase1]: " CONDA_ENV
CONDA_ENV="${CONDA_ENV:-depressivoice-phase1}"

# Generate config file
echo ""
echo "========================================"
echo "Generating $CONFIG_FILE..."
echo "========================================"

cat > "$CONFIG_FILE" <<EOF
# HPC Cluster Configuration for Phase-1 Orchestration
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Hostname: $HOSTNAME

# Repository and Output Paths
export REPO_ROOT="$REPO_ROOT"
export OUT_ROOT="$OUT_ROOT"

# SLURM Configuration
export SLURM_PARTITION="$SLURM_PARTITION"
export SLURM_TIME="$SLURM_TIME"
export SLURM_MEM="$SLURM_MEM"

# Conda Configuration
export CONDA_ENV="$CONDA_ENV"
export CONDA_MODULE="$CONDA_MODULE"

# Git Branch for Worktree
export BRANCH_NAME="phase1-gpu-env-setup"

# Orchestration Metadata
export ORCHESTRATION_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export TIMESTAMP="\$(date -u +"%Y%m%d_%H%M%S")"
EOF

echo ""
echo "✓ Configuration saved to: $CONFIG_FILE"
echo ""
echo "========================================"
echo "Configuration Summary"
echo "========================================"
cat "$CONFIG_FILE"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review and edit $CONFIG_FILE if needed"
echo "  2. Run: source $CONFIG_FILE"
echo "  3. Run: ./01_precheck.sh"
echo ""
