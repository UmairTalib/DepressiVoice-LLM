#!/bin/bash
################################################################################
# 05_finalize.sh
# Finalizes environment setup based on validation results
# Creates env_ready.json on PASS/WARN or failure_report.zip on FAIL
################################################################################

set -euo pipefail

source cluster_config.env

LOG_DIR="$OUT_ROOT/env_install_logs"
ACTION_LOG="$LOG_DIR/actions.log"

log_action() {
  local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
  echo "[$timestamp] $1" | tee -a "$ACTION_LOG"
}

log_action "=========================================="
log_action "FINALIZE: Processing validation results"
log_action "=========================================="

# Check if compute_result.json exists
if [ ! -f "$LOG_DIR/compute_result.json" ]; then
  log_action "❌ ERROR: compute_result.json not found"
  log_action "Environment validation did not complete"
  exit 1
fi

# Read status from compute_result.json
RESULT_STATUS=$(grep -o '"status": "[^"]*"' "$LOG_DIR/compute_result.json" | cut -d'"' -f4)

log_action "Validation status: $RESULT_STATUS"

if [ "$RESULT_STATUS" == "FAIL" ]; then
  log_action "=========================================="
  log_action "FAIL: Creating failure report"
  log_action "=========================================="
  
  FAILURE_ZIP="$LOG_DIR/failure_report_$(date -u +%Y%m%d_%H%M%S).zip"
  
  log_action "Bundling failure diagnostics..."
  
  # Create failure report bundle
  cd "$LOG_DIR"
  zip -r "$FAILURE_ZIP" \
    *.json \
    *.log \
    slurm-*.out \
    slurm-*.err \
    phase1_env_check_*.out \
    phase1_env_check_*.err \
    2>&1 | tee -a "$ACTION_LOG" || true
  
  log_action "✓ Created: $FAILURE_ZIP"
  log_action ""
  log_action "=========================================="
  log_action "ENVIRONMENT SETUP FAILED"
  log_action "=========================================="
  log_action "Please review the failure report and logs:"
  log_action "  - $FAILURE_ZIP"
  log_action "  - $LOG_DIR/compute_result.json"
  log_action ""
  log_action "Common issues:"
  log_action "  - Missing CUDA libraries (if GPU required)"
  log_action "  - Network issues during pip install"
  log_action "  - Incompatible package versions"
  log_action ""
  log_action "❌ Human intervention required"
  log_action "=========================================="
  
  exit 1
fi

# Status is PASS or WARN - create env_ready.json
log_action "=========================================="
log_action "SUCCESS: Creating environment ready marker"
log_action "=========================================="

# Gather summary information
TOTAL_PACKAGES=$(cat "$LOG_DIR/pip_list_compute.json" | grep -c '"name"' || echo "unknown")
TORCH_VERSION=$(grep -o '"torch_version": "[^"]*"' "$LOG_DIR/torch_cuda.json" | cut -d'"' -f4 || echo "unknown")
CUDA_AVAILABLE=$(grep -o '"cuda_available": [^,]*' "$LOG_DIR/torch_cuda.json" | cut -d' ' -f2 || echo "false")

cat > "$LOG_DIR/env_ready.json" <<EOF
{
  "status": "READY",
  "validation_status": "$RESULT_STATUS",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": {
    "conda_env": "$CONDA_ENV",
    "total_packages": $TOTAL_PACKAGES,
    "torch_version": "$TORCH_VERSION",
    "cuda_available": $CUDA_AVAILABLE
  },
  "output_locations": {
    "worktree": "$OUT_ROOT/phase1_worktree",
    "logs": "$LOG_DIR",
    "manifests": {
      "worktree": "$LOG_DIR/worktree_manifest.json",
      "imports": "$LOG_DIR/compute_imports.json",
      "torch_cuda": "$LOG_DIR/torch_cuda.json",
      "pip_list": "$LOG_DIR/pip_list_compute.json"
    }
  },
  "next_steps": {
    "1": "Review env_ready.json and all validation reports",
    "2": "If satisfied, create human approval: echo '{\"approved\": true}' > $LOG_DIR/human_ok.json",
    "3": "Once approved, Phase-1 processing can begin"
  }
}
EOF

log_action "✓ Created: $LOG_DIR/env_ready.json"

# Create summary report
cat > "$LOG_DIR/env_install_summary.txt" <<EOF
========================================
Phase-1 Environment Setup Summary
========================================
Status: $RESULT_STATUS
Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

PATHS:
  REPO_ROOT (read-only): $REPO_ROOT
  OUT_ROOT: $OUT_ROOT
  Worktree: $OUT_ROOT/phase1_worktree
  Logs: $LOG_DIR

ENVIRONMENT:
  Conda environment: $CONDA_ENV
  Total packages: $TOTAL_PACKAGES
  PyTorch version: $TORCH_VERSION
  CUDA available: $CUDA_AVAILABLE

VALIDATION FILES:
  - compute_result.json     (overall validation)
  - compute_imports.json    (package import checks)
  - torch_cuda.json         (torch/CUDA details)
  - pip_list_compute.json   (all installed packages)
  - worktree_manifest.json  (files copied to worktree)

NEXT STEPS:
  1. Review all validation reports (see files above)
  2. Verify REPO_ROOT unchanged: cd $REPO_ROOT && git status
  3. Check for warnings in compute_result.json
  4. If satisfied, approve for Phase-1:
     echo '{"approved_by": "$USER", "timestamp": "$(date -u +%Y-%m-%d)"}' > $LOG_DIR/human_ok.json

========================================
EOF

log_action "✓ Created: $LOG_DIR/env_install_summary.txt"
log_action ""
log_action "=========================================="
log_action "ENVIRONMENT SETUP COMPLETED: $RESULT_STATUS"
log_action "=========================================="
log_action ""

cat "$LOG_DIR/env_install_summary.txt" | tee -a "$ACTION_LOG"

log_action ""
log_action "=========================================="
log_action "⏸️  WAITING FOR HUMAN APPROVAL"
log_action "=========================================="
log_action ""
log_action "To approve and proceed with Phase-1:"
log_action "  echo '{\"approved_by\": \"$USER\", \"timestamp\": \"$(date -u +%Y-%m-%d)\"}' > $LOG_DIR/human_ok.json"
log_action ""
log_action "To review:"
log_action "  cat $LOG_DIR/env_install_summary.txt"
log_action "  cat $LOG_DIR/compute_result.json"
log_action ""

exit 0
