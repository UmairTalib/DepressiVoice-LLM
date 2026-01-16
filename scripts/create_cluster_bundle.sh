#!/bin/bash

# Define the zip filename
ZIP_NAME="depressivoice_cluster_deploy.zip"

echo "Creating deployment package: $ZIP_NAME"

# Create the zip file excluding unnecessary items like venv, git, local cache
# We include:
# - src/ (Code)
# - config.yaml (Configuration)
# - requirements.txt (Dependencies)
# - run_parallel.sh (Job Script)
# - README_DIRECT_INSTALL.md (Docs)
# - README_CLUSTER.md (Docs)


zip -r $ZIP_NAME \
    src/ \
    config.yaml \
    requirements.txt \
    run_parallel.sh \
    run_features_parallel.sh \
    run_setup.sh \
    run_test_model.sh \
    README_DIRECT_INSTALL.md \
    -x "src/__pycache__/*" \
    -x "*.DS_Store"

echo ""
echo "========================================================"
echo "SUCCESS!"
echo "Created: $ZIP_NAME"
echo "========================================================"
echo "NEXT STEPS:"
echo "1. Upload this zip file to the cluster:"
echo "   scp $ZIP_NAME <username>@<cluster_address>:~/"
echo ""
echo "2. Upload your DATA folder separately (it might be large):"
echo "   scp -r data/01_raw <username>@<cluster_address>:~/data/01_raw"
echo "========================================================"
