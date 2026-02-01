#!/bin/bash
# scripts/sync_phase1_data.sh
# Syncs Phase 1 outputs (Metadata/Text/Reports) from Cluster to Local
# Excludes heavy audio files to save bandwidth (Audio remains on Cluster for Phase 2 training)

CLUSTER_USER="g755389"
CLUSTER_HOST="omni.zimt.uni-siegen.de"
REMOTE_BASE="/home/g755389/depressivoice_phase2_outputs"
LOCAL_DATA_DIR="data/phase1_outputs"

mkdir -p "$LOCAL_DATA_DIR"

echo "=== Syncing Phase 1 Data ==="

# 1. Package Metadata on Cluster (JSON/TXT only)
echo "📦 Packaging session metadata on cluster..."
ssh $CLUSTER_USER@$CLUSTER_HOST "cd $REMOTE_BASE && zip -q -r session_metadata.zip session_outputs -i '*.json' -i '*.txt' -i '*.csv'"

# 2. Download Metadata Zip
echo "⬇️ Downloading session metadata..."
scp $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_BASE/session_metadata.zip "$LOCAL_DATA_DIR/"

# 3. Unzip
echo "📂 Unzipping to $LOCAL_DATA_DIR..."
unzip -q -o "$LOCAL_DATA_DIR/session_metadata.zip" -d "$LOCAL_DATA_DIR"
rm "$LOCAL_DATA_DIR/session_metadata.zip"

# 4. Download Validation Bundle
echo "⬇️ Downloading Final Validation Bundle..."
scp -r $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_BASE/phase2_worktree/final_lock_bundle "$LOCAL_DATA_DIR/"

echo "✅ Sync Complete. Data is in $LOCAL_DATA_DIR"
echo "   Note: Heavy WAV files remain on cluster at $REMOTE_BASE/session_outputs"
