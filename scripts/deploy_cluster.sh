#!/bin/bash

# Configuration
CLUSTER_USER="g755389"
CLUSTER_HOST="omni.zimt.uni-siegen.de"
REMOTE_PATH="~/"  # Deploy to home directory

echo "================================================="
echo "Deploying to $CLUSTER_USER@$CLUSTER_HOST"
echo "================================================="

# 1. Clean up old files (Preserve DATA input and VENV)
echo "[1/4] Cleaning up remote files..."
# Remove old code, temp zips, and PARTIAL output features to ensure a clean run
ssh $CLUSTER_USER@$CLUSTER_HOST "rm -rf src scripts config.yaml requirements.txt logs *.zip data/session_ids.txt DepressiVoice-LLM/data/03_features/*.pt"
# Explicitly delete the zip if it's stuck
ssh $CLUSTER_USER@$CLUSTER_HOST "rm -f depressivoice_cluster_deploy.zip"

# 2. CREATE BUNDLE (Ensure we have the latest)
echo "[2/4] Creating local deployment package..."
./scripts/create_cluster_bundle.sh > /dev/null

# 3. UPLOAD
echo "[3/4] Uploading new deployment package..."
scp depressivoice_cluster_deploy.zip $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_PATH

# 4. EXTRACT & SETUP
echo "[4/4] Extracting and setting up on cluster..."
ssh $CLUSTER_USER@$CLUSTER_HOST "unzip -o -q depressivoice_cluster_deploy.zip && chmod +x scripts/*.sb scripts/*.sh && echo 'Deployment Complete!'"

echo "================================================="
echo "Ready! Log in and run:"
echo "ssh $CLUSTER_USER@$CLUSTER_HOST"
echo "sbatch scripts/submit_extraction.sb"
echo "================================================="
