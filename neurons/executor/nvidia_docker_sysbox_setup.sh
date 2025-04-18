#!/bin/bash

# Sysbox v0.6.6 Installation Script
# This script installs Sysbox v0.6.6 on Ubuntu/Debian systems and sets up NVIDIA Container Toolkit

# Exit on any error
set -e

echo "Starting Sysbox v0.6.6 installation with NVIDIA Container Toolkit setup..."

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    exit 1
fi

# Step 1: Check if nvidia-container-toolkit is installed
echo "Step 1: Checking NVIDIA Container Toolkit installation..."
if command -v nvidia-container-cli &> /dev/null; then
    echo "NVIDIA Container CLI is installed:"
    nvidia-container-cli --version
else
    echo "NVIDIA Container CLI is not installed. Installing it now..."
    apt-get update
    apt-get install -y nvidia-container-toolkit
fi

# Step 2: Update NVIDIA container toolkit to latest version
echo "Step 2: Updating NVIDIA Container Toolkit to latest version..."
apt-get update && apt-get install --only-upgrade nvidia-container-toolkit

# Step 3: Enable cgroups for docker
echo "Step 3: Enabling cgroups for Docker in NVIDIA Container Runtime config..."
NVIDIA_CONFIG="/etc/nvidia-container-runtime/config.toml"
if [ -f "$NVIDIA_CONFIG" ]; then
    # Check if no-cgroups is present but commented out
    if grep -q "^#no-cgroups" "$NVIDIA_CONFIG"; then
        # Uncomment and ensure it's set to false
        sed -i 's/^#no-cgroups.*$/no-cgroups = false/' "$NVIDIA_CONFIG"
        echo "Uncommented and set no-cgroups = false in $NVIDIA_CONFIG"
    # Check if no-cgroups is present and set to true
    elif grep -q "^no-cgroups = true" "$NVIDIA_CONFIG"; then
        # Change true to false
        sed -i 's/^no-cgroups = true/no-cgroups = false/' "$NVIDIA_CONFIG"
        echo "Changed no-cgroups from true to false in $NVIDIA_CONFIG"
    # Check if no-cgroups is present and already set to false
    elif grep -q "^no-cgroups = false" "$NVIDIA_CONFIG"; then
        echo "no-cgroups is already set to false in $NVIDIA_CONFIG"
    else
        # Add the setting if it's not present at all
        # Find the [nvidia-container-cli] section and add the setting after it
        sed -i '/\[nvidia-container-cli\]/a no-cgroups = false' "$NVIDIA_CONFIG"
        echo "Added no-cgroups = false to [nvidia-container-cli] section in $NVIDIA_CONFIG"
    fi
else
    echo "Warning: NVIDIA Container Runtime config file not found at $NVIDIA_CONFIG"
fi

# Step 4: Configure Docker daemon.json
echo "Step 4: Configuring Docker daemon.json..."
DOCKER_DAEMON_CONFIG='/etc/docker/daemon.json'

# Create Docker config directory if it doesn't exist
mkdir -p /etc/docker

# Create temporary configuration with our required settings
TMP_FILE="/tmp/daemon.json.new"
cat > "$TMP_FILE" << EOF
{
    "runtimes": {
        "sysbox-runc": {
            "path": "/usr/bin/sysbox-runc"
        },
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "exec-opts": ["native.cgroupdriver=cgroupfs"],
    "bip": "172.24.0.1/16",
    "default-address-pools": [
        {
            "base": "172.31.0.0/16",
            "size": 24
        }
    ]
}
EOF

# If daemon.json exists, merge with it, otherwise create baseline configuration
if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
    echo "Existing Docker daemon configuration found. Merging configurations..."
    # Merge with existing configuration
    jq -s '.[0] * .[1]' "$DOCKER_DAEMON_CONFIG" "$TMP_FILE" > "$TMP_FILE.2"
else
    echo "No existing Docker daemon configuration found. Creating baseline configuration..."
    # Create baseline configuration with network settings
    BASE_CONFIG="/tmp/base.json"
    cat > "$BASE_CONFIG" << EOF
{
    "bip": "172.20.0.1/16",
    "default-address-pools": [
        {
            "base": "172.25.0.0/16",
            "size": 24
        }
    ]
}
EOF
    # Merge baseline with our required settings
    jq -s '.[0] * .[1]' "$BASE_CONFIG" "$TMP_FILE" > "$TMP_FILE.2"
    # Clean up base config
    rm -f "$BASE_CONFIG"
fi

# Move the final result to daemon.json
mv "$TMP_FILE.2" "$DOCKER_DAEMON_CONFIG"

# Clean up temporary files
rm -f "$TMP_FILE" 2>/dev/null || true

# Install jq dependency if not already installed
echo "Installing dependencies..."
apt-get update
apt-get install -y jq

# Download Sysbox v0.6.6
echo "Downloading Sysbox v0.6.6..."
wget https://downloads.nestybox.com/sysbox/releases/v0.6.6/sysbox-ce_0.6.6-0.linux_amd64.deb

# Install Sysbox
echo "Installing Sysbox..."
apt-get install -y ./sysbox-ce_0.6.6-0.linux_amd64.deb

# Restart Docker to apply changes
echo "Restarting Docker service..."
if systemctl is-active --quiet docker; then
    systemctl restart docker
else
    echo "Warning: Docker service not running. Please start it manually."
fi

# Verify installation
if systemctl is-active --quiet sysbox; then
    echo "Sysbox v0.6.6 installed and running successfully."
    echo "Docker configured with custom network settings."
    echo "You can now create system containers with Docker using: docker run --runtime=sysbox-runc -it <image>"
else
    echo "Sysbox installation completed but service is not running."
    echo "Try starting it manually with: systemctl start sysbox"
fi

# Clean up downloaded deb file
echo "Cleaning up..."
rm -f sysbox-ce_0.6.6-0.linux_amd64.deb

echo "Installation complete!"