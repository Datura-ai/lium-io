#!/bin/bash
set -e

# Build the watchtower Docker image
echo "Building watchtower Docker image..."
docker build -t daturaai/lium-watchtower:latest .

