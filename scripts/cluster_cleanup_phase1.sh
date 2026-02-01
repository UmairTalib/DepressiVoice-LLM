#!/bin/bash
# scripts/cluster_cleanup_phase1.sh
# Run this ON THE CLUSTER to clean up after Phase 1

BASE_DIR="/home/g755389/DepressiVoice-LLM"

echo "=== Phase 1 Cluster Cleanup ==="

# 1. Archive Logs
LOG_DIR="$BASE_DIR/logs"
ARCHIVE_DIR="$BASE_DIR/logs_archive/phase1_complete"
mkdir -p "$ARCHIVE_DIR"

if [ -d "$LOG_DIR" ]; then
    echo "📦 Archiving logs to $ARCHIVE_DIR..."
    mv "$LOG_DIR"/* "$ARCHIVE_DIR/" 2>/dev/null
else
    echo "⚠️ No logs directory found."
fi

# 2. Remove Temporary Scripts (The ones uploaded to root)
echo "🧹 Removing temporary root scripts..."
rm -f "$BASE_DIR"/micro_cleanup.py
rm -f "$BASE_DIR"/micro_cleanup_robust.py
rm -f "$BASE_DIR"/audit_cleanup.py
rm -f "$BASE_DIR"/final_audit.py
rm -f "$BASE_DIR"/identify_residuals.py
rm -f "$BASE_DIR"/clean_transcripts_*.py
rm -f "$BASE_DIR"/generate_*.py
rm -f "$BASE_DIR"/*_rescue_*.py

# 3. Secure Session Outputs
echo "🔒 Verifying Session Outputs..."
OUTPUT_COUNT=$(find "$BASE_DIR/session_outputs" -maxdepth 1 -type d | wc -l)
echo "   Found $OUTPUT_COUNT sessions in session_outputs."

echo "✅ Cleanup Complete. 'session_outputs' is ready for Phase 2."
