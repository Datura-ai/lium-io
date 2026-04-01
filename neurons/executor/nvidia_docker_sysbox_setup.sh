#!/bin/bash
set -e

# ============================================================
# Lium Executor — Sysbox Setup Script
# Installs sysbox runtime with NVIDIA GPU support.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Datura-ai/compute-subnet/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo bash
#   or: cd compute-subnet/neurons/executor && sudo bash nvidia_docker_sysbox_setup.sh
# ============================================================

SYSBOX_VERSION="0.6.6"
SYSBOX_DEB_URL="https://github.com/nestybox/sysbox/releases/download/v${SYSBOX_VERSION}/sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
SYSBOX_DEB_SHA256="87cfa5cad97dc5dc1a243d6d88be1393be75b93a517dc1580ecd8a2801c2777a"
DAEMON_JSON="/etc/docker/daemon.json"
VERIFY_IMAGE="daturaai/compute-subnet-executor:latest"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 1. Pre-flight checks ──────────────────────────────────

if [ "$(id -u)" -ne 0 ]; then
    error "This script must be run as root (use sudo)."
    exit 1
fi

if [ "$(uname -m)" != "x86_64" ]; then
    error "Sysbox requires x86_64 architecture. Detected: $(uname -m)"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install Docker first: https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

if ! docker ps &>/dev/null; then
    error "Docker daemon is not running or not accessible."
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. NVIDIA drivers must be installed first."
    exit 1
fi

if ! ls /proc/driver/nvidia &>/dev/null; then
    error "NVIDIA driver not loaded (/proc/driver/nvidia missing)."
    error "Try: sudo nvidia-smi  (to initialize the driver)"
    exit 1
fi

info "Pre-flight checks passed."

# ── 2. Already installed? ─────────────────────────────────

if command -v sysbox-runc &>/dev/null; then
    installed_version=$(sysbox-runc --version 2>/dev/null | head -1 || echo "unknown")
    info "Sysbox is already installed: $installed_version"
    info "Skipping installation, jumping to configuration and verification..."
    SKIP_INSTALL=true
else
    SKIP_INSTALL=false
fi

# ── 3. Check running rented pods ──────────────────────────

rented_pods=$(docker ps --filter "name=pod_" --format "{{.Names}}" 2>/dev/null || true)

if [ -n "$rented_pods" ]; then
    warn "Found running rented pods:"
    echo "$rented_pods" | sed 's/^/  - /'
    echo ""
    warn "Docker restart is required and will STOP these pods."
    warn "Active tenants will be disconnected."
    echo ""

    if [ ! -t 0 ]; then
        error "Cannot proceed non-interactively with active rented pods."
        error "Run this script manually when pods are idle:"
        error "  sudo bash nvidia_docker_sysbox_setup.sh"
        exit 1
    fi

    echo "Options:"
    echo "  1) Abort — wait until pods finish, then run again"
    echo "  2) Continue — install sysbox and restart Docker now"
    echo ""
    read -rp "Choose [1/2]: " choice
    if [ "$choice" != "2" ]; then
        info "Aborted. Run this script again when there are no active rentals."
        exit 0
    fi
    warn "Continuing with active pods. They will be stopped."
fi

# ── 4. Install packages ──────────────────────────────────

info "Installing nvidia-container-toolkit and jq..."
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit jq > /dev/null

if [ "$SKIP_INSTALL" = false ]; then
    LOCAL_DEB="./sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
    if [ -f "$LOCAL_DEB" ]; then
        info "Installing sysbox v${SYSBOX_VERSION} from local package..."
        SYSBOX_DEB="$LOCAL_DEB"
    else
        info "Downloading sysbox v${SYSBOX_VERSION}..."
        SYSBOX_DEB=$(mktemp /tmp/sysbox-ce.XXXXXX.deb)
        wget -q -O "$SYSBOX_DEB" "$SYSBOX_DEB_URL"
        ACTUAL_SHA=$(sha256sum "$SYSBOX_DEB" | cut -d' ' -f1)
        if [ "$ACTUAL_SHA" != "$SYSBOX_DEB_SHA256" ]; then
            error "Checksum mismatch! Expected: $SYSBOX_DEB_SHA256"
            error "Got: $ACTUAL_SHA"
            rm -f "$SYSBOX_DEB"
            exit 1
        fi
        info "Checksum verified."
    fi
    apt-get install -y -qq "$SYSBOX_DEB" > /dev/null
fi

# ── 5. Configure Docker daemon ────────────────────────────

info "Configuring Docker daemon..."

SYSBOX_CONFIG='{
    "runtimes": {
        "sysbox-runc": {"path": "/usr/bin/sysbox-runc"},
        "nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}
    },
    "exec-opts": ["native.cgroupdriver=cgroupfs"],
    "features": {"cdi": false}
}'

mkdir -p /etc/docker

if [ -f "$DAEMON_JSON" ]; then
    info "Merging sysbox config into existing $DAEMON_JSON..."
    jq --argjson patch "$SYSBOX_CONFIG" '. * $patch' "$DAEMON_JSON" > /tmp/daemon.json.tmp
    mv /tmp/daemon.json.tmp "$DAEMON_JSON"
else
    info "Creating $DAEMON_JSON..."
    echo "$SYSBOX_CONFIG" | jq '.' > "$DAEMON_JSON"
fi

# ── 6. CDI cleanup (Docker >= 29.2.0 compatibility) ──────

DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
DOCKER_MINOR=$(echo "$DOCKER_VERSION" | cut -d. -f2)

if [ "$DOCKER_MINOR" -ge 2 ] 2>/dev/null; then
    info "Docker $DOCKER_VERSION detected (>= 29.2.0). Cleaning up CDI specs..."
    rm -f /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml
    if systemctl list-units --all | grep -q nvidia-cdi-refresh; then
        systemctl disable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service 2>/dev/null || true
        info "Disabled nvidia-cdi-refresh service."
    fi
fi

# ── 7. Restart Docker ─────────────────────────────────────

info "Restarting Docker daemon..."
systemctl restart docker

for i in $(seq 1 30); do
    if docker ps &>/dev/null; then
        break
    fi
    sleep 1
done

if ! docker ps &>/dev/null; then
    error "Docker failed to restart within 30 seconds."
    error "Check: journalctl -u docker.service"
    exit 1
fi

info "Docker is running."

# ── 8. Verify ─────────────────────────────────────────────

info "Verifying sysbox + GPU (this may pull the image on first run)..."

if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi; then
    echo ""
    info "============================================"
    info " SUCCESS: Sysbox is working with GPU support"
    info "============================================"
    echo ""
    info "Your executor will receive full incentive score (no 20% sysbox penalty)."
else
    echo ""
    error "Sysbox verification FAILED. Diagnostics:"
    echo ""
    echo "  Docker version:     $(docker --version 2>/dev/null || echo 'unknown')"
    echo "  nvidia-smi:         $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'FAILED')"
    echo "  /proc/driver/nvidia: $(ls /proc/driver/nvidia &>/dev/null && echo 'exists' || echo 'MISSING')"
    echo "  sysbox-runc:        $(sysbox-runc --version 2>/dev/null | head -1 || echo 'not found')"
    echo "  CDI specs:          $(ls /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml 2>/dev/null || echo 'none')"
    echo "  daemon.json cdi:    $(jq -r '.features.cdi // "not set"' /etc/docker/daemon.json 2>/dev/null)"
    echo ""
    error "Check logs:"
    error "  journalctl -u docker.service --no-pager -n 20"
    error "  journalctl -u sysbox-mgr.service --no-pager -n 20"
    exit 1
fi
