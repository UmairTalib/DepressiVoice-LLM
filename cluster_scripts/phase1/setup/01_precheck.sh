#!/bin/bash
################################################################################
# 01_precheck.sh
# Validates execution environment before running orchestration
# Checks: hostname (not on login node), SLURM context, creates log directory
################################################################################

set -euo pipefail

# Source configuration
if [ ! -f "cluster_config.env" ]; then
  echo "❌ ERROR: cluster_config.env not found"
  echo "Run ./00_collect_info.sh first to generate configuration"
  exit 1
fi

source cluster_config.env

# Initialize logging
LOG_DIR="$OUT_ROOT/env_install_logs"
mkdir -p "$LOG_DIR"
ACTION_LOG="$LOG_DIR/actions.log"

# Logging function
log_action() {
  local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
  echo "[$timestamp] $1" | tee -a "$ACTION_LOG"
}

log_action "=========================================="
log_action "PRECHECK: Starting environment validation"
log_action "=========================================="
log_action "Hostname: $(hostname)"
log_action "User: $USER"
log_action "PWD: $PWD"
log_action "REPO_ROOT: $REPO_ROOT"
log_action "OUT_ROOT: $OUT_ROOT"

# Check 1: Verify we're NOT on login node
HOSTNAME=$(hostname)
log_action "Checking execution node type..."

IS_LOGIN_NODE=false
if [[ "$HOSTNAME" == *"login"* ]]; then
  IS_LOGIN_NODE=true
  log_action "⚠️  WARNING: Detected login node (hostname contains 'login')"
fi

# Check if we're in a SLURM job
if [ -z "${SLURM_JOB_ID:-}" ]; then
  log_action "⚠️  WARNING: Not running in SLURM job (SLURM_JOB_ID is empty)"
  IN_SLURM=false
else
  log_action "✓ Running in SLURM job: $SLURM_JOB_ID"
  IN_SLURM=true
fi

# Decision: Are we safe to proceed with heavy operations?
if [ "$IS_LOGIN_NODE" = true ] || [ "$IN_SLURM" = false ]; then
  log_action "❌ BLOCKING: Cannot run installation/validation on login node without SLURM"
  
  # Create need_compute_node.json
  cat > "$LOG_DIR/need_compute_node.json" <<EOF
{
  "status": "BLOCKED",
  "reason": "Execution on login node detected without SLURM context",
  "hostname": "$HOSTNAME",
  "slurm_job_id": "${SLURM_JOB_ID:-null}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "guidance": {
    "message": "Environment validation and installation must be performed on a compute node via SLURM",
    "next_step": "Submit the environment check job using: sbatch 03_env_check.sbatch",
    "alternative": "Run this script interactively on compute node: srun --partition=$SLURM_PARTITION --time=00:15:00 --mem=4G ./01_precheck.sh"
  }
}
EOF
  
  log_action "Created: $LOG_DIR/need_compute_node.json"
  log_action ""
  log_action "=========================================="
  log_action "NEXT STEPS:"
  log_action "=========================================="
  log_action "1. Review cluster_config.env"
  log_action "2. Proceed to create worktree: ./02_create_worktree.sh"
  log_action "3. Submit environment check: sbatch 03_env_check.sbatch"
  log_action "=========================================="
  
  # This is actually OK - precheck can run on login, heavy stuff runs via SLURM
  exit 0
fi

# If we reach here, we're on compute node in SLURM job
log_action "✓ Precheck passed: Running on compute node in SLURM context"

# Check 2: Verify REPO_ROOT exists
log_action "Verifying REPO_ROOT..."
if [ ! -d "$REPO_ROOT" ]; then
  log_action "❌ ERROR: REPO_ROOT does not exist: $REPO_ROOT"
  exit 1
fi

if [ ! -d "$REPO_ROOT/.git" ]; then
  log_action "⚠️  WARNING: REPO_ROOT is not a git repository: $REPO_ROOT"
fi

log_action "✓ REPO_ROOT exists: $REPO_ROOT"

# Check 3: Create OUT_ROOT if needed
log_action "Setting up OUT_ROOT..."
mkdir -p "$OUT_ROOT"
mkdir -p "$LOG_DIR"
log_action "✓ Created output directories"

# Check 4: Verify git is available
if command -v git &>/dev/null; then
  GIT_VERSION=$(git --version)
  log_action "✓ Git available: $GIT_VERSION"
else
  log_action "⚠️  WARNING: git command not found"
fi

# Write precheck result
cat > "$LOG_DIR/precheck_result.json" <<EOF
{
  "status": "PASS",
  "hostname": "$HOSTNAME",
  "slurm_job_id": "${SLURM_JOB_ID:-null}",
  "user": "$USER",
  "repo_root": "$REPO_ROOT",
  "repo_exists": true,
  "out_root": "$OUT_ROOT",
  "out_root_writable": true,
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

log_action "✓ Precheck completed successfully"
log_action "Created: $LOG_DIR/precheck_result.json"
log_action "=========================================="
log_action ""

exit 0
