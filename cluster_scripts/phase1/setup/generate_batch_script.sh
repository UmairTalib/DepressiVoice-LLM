#!/bin/bash
################################################################################
# generate_batch_script.sh
# Generates 03_env_check.sbatch from template using cluster_config.env
################################################################################

set -euo pipefail

if [ ! -f "cluster_config.env" ]; then
  echo "❌ ERROR: cluster_config.env not found"
  echo "Run ./00_collect_info.sh first"
  exit 1
fi

source cluster_config.env

TEMPLATE="03_env_check.sbatch.template"
OUTPUT="03_env_check.sbatch"

if [ ! -f "$TEMPLATE" ]; then
  echo "❌ ERROR: Template file not found: $TEMPLATE"
  exit 1
fi

echo "Generating $OUTPUT from template..."
echo "  SLURM_PARTITION: $SLURM_PARTITION"
echo "  SLURM_TIME: $SLURM_TIME"
echo "  SLURM_MEM: $SLURM_MEM"
echo "  OUT_ROOT: $OUT_ROOT"
echo "  CONDA_ENV: $CONDA_ENV"
echo "  CONDA_MODULE: $CONDA_MODULE"
echo "  REPO_ROOT: $REPO_ROOT"

sed -e "s|__SLURM_PARTITION__|$SLURM_PARTITION|g" \
    -e "s|__SLURM_TIME__|$SLURM_TIME|g" \
    -e "s|__SLURM_MEM__|$SLURM_MEM|g" \
    -e "s|__OUT_ROOT__|$OUT_ROOT|g" \
    -e "s|__CONDA_ENV__|$CONDA_ENV|g" \
    -e "s|__CONDA_MODULE__|$CONDA_MODULE|g" \
    -e "s|__REPO_ROOT__|$REPO_ROOT|g" \
    "$TEMPLATE" > "$OUTPUT"

chmod +x "$OUTPUT"

echo "✓ Generated: $OUTPUT"
echo ""
echo "Next step: sbatch $OUTPUT"
