#!/bin/bash
set -e

echo "Starting Docker Build for umi002/depressivoice-llm:latest..."

# Fix for "docker-credential-desktop not found"
export PATH=$PATH:/Applications/Docker.app/Contents/Resources/bin

# Build the image using the full path to docker found on system
/usr/local/bin/docker build -t umi002/depressivoice-llm:latest .

echo "Build complete. Pushing to Docker Hub..."
/usr/local/bin/docker push umi002/depressivoice-llm:latest

echo "Done! You can now pull this image on the cluster."
