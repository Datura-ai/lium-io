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
DOWNLOADED_DEB=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

info()  { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}!${NC} $1"; }
error() { echo -e "  ${RED}✗${NC} $1"; }
step()  { echo -e "\n${BLUE}[$1/$2]${NC} $3"; }

cleanup() {
    if [ -n "$DOWNLOADED_DEB" ] && [ -f "$DOWNLOADED_DEB" ]; then
        rm -f "$DOWNLOADED_DEB"
    fi
}
trap cleanup EXIT

# ── 1. Pre-flight checks ──────────────────────────────────

step 1 6 "Pre-flight checks"

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

info "All checks passed."

echo ""
echo "  This script will:"
echo "    • Install sysbox runtime (for Docker-in-Docker support)"
echo "    • Configure Docker daemon (daemon.json)"
echo "    • Restart Docker"
echo "    • Verify sysbox + GPU works"
echo ""

if [ -t 0 ]; then
    read -rp "  Continue? [Y/n]: " confirm
    if [ "$confirm" = "n" ] || [ "$confirm" = "N" ]; then
        info "Aborted."
        exit 0
    fi
fi

# ── 2. Quick check: already working? ─────────────────────

if command -v sysbox-runc &>/dev/null; then
    # Only check if runtime is registered in Docker and image is cached
    if docker info 2>/dev/null | grep -q sysbox-runc; then
        if docker image inspect "$VERIFY_IMAGE" &>/dev/null; then
            if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
                echo ""
                info "Sysbox is already working with GPU support. Nothing to do."
                exit 0
            fi
        fi
    fi
    warn "Sysbox is installed but not working. Reconfiguring..."
    SKIP_INSTALL=true
else
    SKIP_INSTALL=false
fi

# ── 3. Check running rented pods ──────────────────────────

step 2 6 "Checking for active rentals"

rented_pods=$(docker ps --filter "name=pod_" --format "{{.Names}}" 2>/dev/null || true)

if [ -n "$rented_pods" ]; then
    warn "Found running rented pods:"
    echo "$rented_pods" | sed 's/^/    - /'
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

    echo "  Options:"
    echo "    1) Abort — wait until pods finish, then run again"
    echo "    2) Continue — install sysbox and restart Docker now"
    echo ""
    read -rp "  Choose [1/2]: " choice
    if [ "$choice" != "2" ]; then
        info "Aborted. Run this script again when there are no active rentals."
        exit 0
    fi
    warn "Continuing with active pods. They will be stopped."
else
    info "No active rentals."
fi

# ── 4. Install packages ──────────────────────────────────

step 3 6 "Installing packages"

info "Installing nvidia-container-toolkit and jq..."
apt-get update -qq 2>/dev/null
apt-get install -y -qq nvidia-container-toolkit jq > /dev/null 2>&1

if [ "$SKIP_INSTALL" = false ]; then
    LOCAL_DEB="./sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
    if [ -f "$LOCAL_DEB" ]; then
        info "Installing sysbox v${SYSBOX_VERSION} from local package..."
        SYSBOX_DEB="$LOCAL_DEB"
    else
        info "Downloading sysbox v${SYSBOX_VERSION}..."
        SYSBOX_DEB=$(mktemp /tmp/sysbox-ce.XXXXXX.deb)
        DOWNLOADED_DEB="$SYSBOX_DEB"
        wget -q -O "$SYSBOX_DEB" "$SYSBOX_DEB_URL"
        ACTUAL_SHA=$(sha256sum "$SYSBOX_DEB" | cut -d' ' -f1)
        if [ "$ACTUAL_SHA" != "$SYSBOX_DEB_SHA256" ]; then
            error "Checksum mismatch! Expected: $SYSBOX_DEB_SHA256"
            error "Got: $ACTUAL_SHA"
            exit 1
        fi
        info "Checksum verified."
    fi
    apt-get install -y -qq "$SYSBOX_DEB" > /dev/null 2>&1
    info "Sysbox v${SYSBOX_VERSION} installed."
else
    info "Sysbox already installed, skipping."
fi

# ── 5. Configure Docker ──────────────────────────────────

step 4 6 "Configuring Docker"

SYSBOX_CONFIG='{
    "runtimes": {
        "sysbox-runc": {"path": "/usr/bin/sysbox-runc"},
        "nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}
    },
    "features": {"cdi": false}
}'

mkdir -p /etc/docker

if [ -f "$DAEMON_JSON" ]; then
    cp "$DAEMON_JSON" "${DAEMON_JSON}.bak"
    jq --argjson patch "$SYSBOX_CONFIG" '. * $patch' "$DAEMON_JSON" > /tmp/daemon.json.tmp
    mv /tmp/daemon.json.tmp "$DAEMON_JSON"
    info "Merged sysbox config into existing daemon.json (backup: daemon.json.bak)."
else
    echo "$SYSBOX_CONFIG" | jq '.' > "$DAEMON_JSON"
    info "Created daemon.json."
fi

# CDI cleanup for Docker >= 29.2.0
DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
DOCKER_MAJOR=$(echo "$DOCKER_VERSION" | cut -d. -f1)
DOCKER_MINOR=$(echo "$DOCKER_VERSION" | cut -d. -f2)

if [ "$DOCKER_MAJOR" -ge 29 ] && [ "$DOCKER_MINOR" -ge 2 ] 2>/dev/null; then
    rm -f /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml
    if systemctl list-units --all | grep -q nvidia-cdi-refresh; then
        systemctl disable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service 2>/dev/null || true
    fi
    info "Cleaned up CDI specs (Docker $DOCKER_VERSION)."
fi

# ── 6. Restart Docker ─────────────────────────────────────

step 5 6 "Restarting Docker"

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

# ── 7. Verify ─────────────────────────────────────────────

step 6 6 "Verifying sysbox + GPU"

if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
    echo ""
    echo -e "  ${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "  ${GREEN}║  SUCCESS: Sysbox is working with GPUs!   ║${NC}"
    echo -e "  ${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    info "Your executor now supports Docker-in-Docker (no 20% sysbox penalty)."
else
    echo ""
    error "Verification FAILED. Diagnostics:"
    echo ""
    echo "    Docker:             $(docker --version 2>/dev/null || echo 'unknown')"
    echo "    NVIDIA driver:      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'FAILED')"
    echo "    /proc/driver/nvidia: $(ls /proc/driver/nvidia &>/dev/null && echo 'exists' || echo 'MISSING')"
    echo "    sysbox-runc:        $(sysbox-runc --version 2>/dev/null | head -1 || echo 'not found')"
    echo "    CDI specs:          $(ls /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml 2>/dev/null || echo 'none')"
    echo "    daemon.json cdi:    $(jq -r '.features.cdi // "not set"' /etc/docker/daemon.json 2>/dev/null)"
    echo ""
    error "Check logs:"
    error "  journalctl -u docker.service --no-pager -n 20"
    error "  journalctl -u sysbox-mgr.service --no-pager -n 20"
    exit 1
fi
