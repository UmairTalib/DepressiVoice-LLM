#!/bin/bash
# scripts/deploy_cluster.sh
# Deploys the current codebase to the Omni Cluster for Phase 2

CLUSTER_USER="g755389"
CLUSTER_HOST="omni.zimt.uni-siegen.de"
REMOTE_BASE="/home/g755389/DepressiVoice-LLM"

echo "=== Deploying Code to Cluster ==="

# 1. Zip the src folder
echo "📦 Packaging source code..."
zip -q -r depressivoice_cluster_deploy.zip src config.yaml requirements.txt setup_project.sh

# 2. Upload
echo "⬆️ Uploading deployment bundle..."
scp depressivoice_cluster_deploy.zip $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_BASE/

# 3. Unzip on Cluster
echo "📂 Unpacking on remote..."
ssh $CLUSTER_USER@$CLUSTER_HOST "cd $REMOTE_BASE && unzip -o -q depressivoice_cluster_deploy.zip && rm depressivoice_cluster_deploy.zip"

# 4. Clean up local
rm depressivoice_cluster_deploy.zip

echo "✅ Deployment Complete!"
echo "   Remote: $REMOTE_BASE/src"
