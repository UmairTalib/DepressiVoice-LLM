#!/bin/bash
# Delete master audio files - Phase 1 rerun complete, patient audio extracted

echo "============================================================"
echo "DELETING MASTER AUDIO FILES"
echo "============================================================"
echo ""

MASTER_DIR="/home/g755389/depressivoice_master_audio"
LOG_FILE="/home/g755389/DepressiVoice-LLM/logs/master_audio_deletion_$(date +%Y%m%d_%H%M%S).log"

echo "Master Audio Directory: $MASTER_DIR" | tee -a "$LOG_FILE"
echo "Deletion Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check directory exists
if [ ! -d "$MASTER_DIR" ]; then
    echo "ERROR: Directory not found: $MASTER_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# Count files
file_count=$(ls "$MASTER_DIR"/*.wav 2>/dev/null | wc -l)
total_size=$(du -sh "$MASTER_DIR" | awk '{print $1}')

echo "Files to delete: $file_count WAV files" | tee -a "$LOG_FILE"
echo "Space to free: $total_size" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List all files to log
echo "File list:" | tee -a "$LOG_FILE"
ls "$MASTER_DIR"/*.wav 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Confirmation
echo "============================================================"
echo "READY TO DELETE"
echo "This action is IRREVERSIBLE"
echo "Master audio files are NO LONGER NEEDED because:"
echo "  - Diarization complete"
echo "  - Patient audio extracted to phase1_rerun_outputs"
echo "  - All future processing uses patient_audio.wav"
echo "============================================================"
echo ""
read -p "Type 'DELETE' to confirm: " confirmation

if [ "$confirmation" != "DELETE" ]; then
    echo "Deletion cancelled." | tee -a "$LOG_FILE"
    exit 0
fi

# Delete files
echo "" | tee -a "$LOG_FILE"
echo "Deleting files..." | tee -a "$LOG_FILE"
rm -rfv "$MASTER_DIR" 2>&1 | tee -a "$LOG_FILE"

# Verify deletion
if [ ! -d "$MASTER_DIR" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "SUCCESS: Master audio directory deleted" | tee -a "$LOG_FILE"
    echo "Space freed: $total_size" | tee -a "$LOG_FILE"
    echo "Files deleted: $file_count" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "ERROR: Deletion may have failed" | tee -a "$LOG_FILE"
    exit 1
fi
