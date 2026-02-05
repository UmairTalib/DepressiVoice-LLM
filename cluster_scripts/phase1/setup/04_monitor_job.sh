#!/bin/bash
################################################################################
# 04_monitor_job.sh
# Monitors SLURM job progress and displays logs
################################################################################

set -euo pipefail

source cluster_config.env

LOG_DIR="$OUT_ROOT/env_install_logs"

# Function to get the latest job ID
get_latest_job() {
  squeue -u $USER -n phase1_env_check -h -o "%i" | head -1
}

# Check if job ID provided as argument
if [ $# -eq 1 ]; then
  JOB_ID="$1"
else
  JOB_ID=$(get_latest_job)
  
  if [ -z "$JOB_ID" ]; then
    echo "No running job found. Checking recent jobs..."
    echo ""
    sacct -u $USER -n phase1_env_check --format=JobID,State,ExitCode,Elapsed -P | head -10
    echo ""
    read -p "Enter job ID to monitor: " JOB_ID
  fi
fi

echo "========================================"
echo "Monitoring SLURM Job: $JOB_ID"
echo "========================================"
echo ""

# Function to check job status
check_status() {
  squeue -j $JOB_ID -h -o "%T" 2>/dev/null || echo "COMPLETED_OR_FAILED"
}

# Function to show job info
show_info() {
  echo "Job Info:"
  scontrol show job $JOB_ID 2>/dev/null | grep -E "(JobId|JobName|JobState|NodeList|StartTime|EndTime|WorkDir)" || \
    sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList -P
  echo ""
}

show_info

STATUS=$(check_status)
echo "Current Status: $STATUS"
echo ""

if [ "$STATUS" == "COMPLETED_OR_FAILED" ]; then
  echo "Job has finished. Checking results..."
  echo ""
  
  # Check for SLURM output files
  SLURM_OUT=$(find . -name "phase1_env_check_${JOB_ID}.out" -o -name "slurm-${JOB_ID}.out" 2>/dev/null | head -1)
  SLURM_ERR=$(find . -name "phase1_env_check_${JOB_ID}.err" -o -name "slurm-${JOB_ID}.err" 2>/dev/null | head -1)
  
  if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
    echo "========== SLURM Output =========="
    tail -50 "$SLURM_OUT"
    echo "=================================="
    echo ""
  fi
  
  if [ -n "$SLURM_ERR" ] && [ -f "$SLURM_ERR" ] && [ -s "$SLURM_ERR" ]; then
    echo "========== SLURM Errors =========="
    tail -50 "$SLURM_ERR"
    echo "=================================="
    echo ""
  fi
  
  # Check result files
  if [ -f "$LOG_DIR/compute_result.json" ]; then
    echo "========== Validation Result =========="
    cat "$LOG_DIR/compute_result.json" | python3 -m json.tool 2>/dev/null || cat "$LOG_DIR/compute_result.json"
    echo "========================================"
    echo ""
    
    # Check status
    RESULT_STATUS=$(grep -o '"status": "[^"]*"' "$LOG_DIR/compute_result.json" | cut -d'"' -f4)
    
    if [ "$RESULT_STATUS" == "PASS" ]; then
      echo "✓ Environment validation PASSED"
      echo "  Next step: ./05_finalize.sh"
    elif [ "$RESULT_STATUS" == "WARN" ]; then
      echo "⚠️  Environment validation PASSED with warnings"
      echo "  Review warnings before proceeding"
      echo "  Next step: ./05_finalize.sh"
    else
      echo "❌ Environment validation FAILED"
      echo "  Review errors in compute_result.json"
      echo "  Check logs in: $LOG_DIR/"
    fi
  else
    echo "⚠️  No compute_result.json found"
    echo "Job may have failed before validation completed"
  fi
  
else
  # Job still running - tail the logs
  echo "Job is running. Press Ctrl+C to stop monitoring."
  echo ""
  
  SLURM_LOG="$LOG_DIR/slurm_job_${JOB_ID}.log"
  
  if [ -f "$SLURM_LOG" ]; then
    echo "Tailing log: $SLURM_LOG"
    echo "========================================"
    tail -f "$SLURM_LOG"
  else
    echo "Waiting for log file: $SLURM_LOG"
    sleep 5
    
    # Try SLURM output file
    SLURM_OUT=$(find . -name "phase1_env_check_${JOB_ID}.out" -o -name "slurm-${JOB_ID}.out" 2>/dev/null | head -1)
    if [ -n "$SLURM_OUT" ] && [ -f "$SLURM_OUT" ]; then
      echo "Tailing SLURM output: $SLURM_OUT"
      echo "========================================"
      tail -f "$SLURM_OUT"
    else
      echo "No log files found yet. Job may not have started."
    fi
  fi
fi
