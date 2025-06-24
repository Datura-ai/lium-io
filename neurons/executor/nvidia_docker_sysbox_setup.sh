#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install -y jq nvidia-container-toolkit

# Copy configuration files
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Copy daemon.json if it exists
if [ -f "$SCRIPT_DIR/daemon.json" ]; then
    sudo mkdir -p /etc/docker
    sudo cp "$SCRIPT_DIR/daemon.json" /etc/docker/daemon.json
fi

# Copy config.toml if it exists
if [ -f "$SCRIPT_DIR/config.toml" ]; then
    sudo mkdir -p /etc/nvidia-container-runtime
    sudo cp "$SCRIPT_DIR/config.toml" /etc/nvidia-container-runtime/config.toml
fi


# Install Sysbox v0.6.6
sudo apt-get install -y ./sysbox-ce_0.6.6-0.linux_amd64.deb
