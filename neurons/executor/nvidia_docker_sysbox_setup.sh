#!/bin/bash
set -e

# Lium Executor — Sysbox Setup
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Datura-ai/compute-subnet/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo bash
#   or: cd compute-subnet/neurons/executor && sudo bash nvidia_docker_sysbox_setup.sh

SYSBOX_VERSION="0.6.6"
SYSBOX_DEB_URL="https://github.com/nestybox/sysbox/releases/download/v${SYSBOX_VERSION}/sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
SYSBOX_SHA="87cfa5cad97dc5dc1a243d6d88be1393be75b93a517dc1580ecd8a2801c2777a"
VERIFY_IMAGE="daturaai/compute-subnet-executor:latest"
DOWNLOADED_DEB=""

G='\033[0;32m' Y='\033[1;33m' R='\033[0;31m' B='\033[1;34m' N='\033[0m'
ok()   { echo -e "  ${G}✓${N} $1"; }
warn() { echo -e "  ${Y}!${N} $1"; }
fail() { echo -e "  ${R}✗${N} $1"; }
step() { echo -e "\n${B}[$1/$2]${N} $3"; }

cleanup() { [ -n "$DOWNLOADED_DEB" ] && rm -f "$DOWNLOADED_DEB"; }
trap cleanup EXIT

docker_version_ge() {
    local ver major minor
    ver=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    major=${ver%%.*} minor=${ver#*.} minor=${minor%%.*}
    [ "$major" -gt "$1" ] 2>/dev/null || { [ "$major" -eq "$1" ] && [ "$minor" -ge "$2" ]; } 2>/dev/null
}

# ── 1. Pre-flight ────────────────────────────────────────

step 1 7 "Pre-flight checks"

[ "$(id -u)" -eq 0 ]                         || { fail "Run as root (use sudo)."; exit 1; }
[ "$(uname -m)" = "x86_64" ]                 || { fail "Sysbox requires x86_64."; exit 1; }
command -v docker &>/dev/null                 || { fail "Docker not installed."; exit 1; }
docker ps &>/dev/null                         || { fail "Docker daemon not running."; exit 1; }
command -v nvidia-smi &>/dev/null             || { fail "nvidia-smi not found. Install NVIDIA drivers first."; exit 1; }
ls /proc/driver/nvidia &>/dev/null            || { fail "NVIDIA driver not loaded. Try: nvidia-smi"; exit 1; }

ok "All checks passed."

echo -e "\n  This will install sysbox, configure Docker, restart Docker, and verify."
if [ -t 0 ]; then
    read -rp "  Continue? [Y/n]: " c; [ "$c" = "n" ] || [ "$c" = "N" ] && { ok "Aborted."; exit 0; }
fi

# ── 2. Already working? ─────────────────────────────────

if command -v sysbox-runc &>/dev/null && docker info 2>/dev/null | grep -q sysbox-runc \
   && docker image inspect "$VERIFY_IMAGE" &>/dev/null \
   && docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
    ok "Sysbox is already working. Nothing to do."
    exit 0
fi

SKIP_INSTALL=false
command -v sysbox-runc &>/dev/null && SKIP_INSTALL=true && warn "Sysbox installed but not working. Reconfiguring..."

# ── 3. Check running containers ─────────────────────────

step 2 7 "Checking running containers"

EXECUTOR_COMPOSE_DIR=""
all_containers=$(docker ps --format "{{.Names}}" 2>/dev/null || true)
has_pods=false has_validator=false has_executor=false has_unknown=false
unknown_list=""

for name in $all_containers; do
    case "$name" in
        pod_*)       has_pods=true ;;
        container_*) has_validator=true ;;
        executor-*|executor_*)
            has_executor=true
            if [ -z "$EXECUTOR_COMPOSE_DIR" ]; then
                EXECUTOR_COMPOSE_DIR=$(docker inspect "$name" --format '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' 2>/dev/null || true)
            fi
            ;;
        *)           has_unknown=true; unknown_list="$unknown_list\n    - $name" ;;
    esac
done

if [ "$has_pods" = true ]; then
    fail "Active rentals found (pod_* containers). Cannot proceed."
    docker ps --filter "name=pod_" --format "    - {{.Names}}" 2>/dev/null
    exit 1
fi

if [ "$has_validator" = true ]; then
    fail "Validator check in progress (container_* containers). Wait ~30 seconds and retry."
    exit 1
fi

if [ "$has_unknown" = true ]; then
    fail "Unknown containers found — stop them manually before proceeding:"
    echo -e "$unknown_list"
    exit 1
fi

if [ "$has_executor" = true ]; then
    if [ -z "$EXECUTOR_COMPOSE_DIR" ]; then
        # Fallback: if running from repo, compose is in current dir
        [ -f ./docker-compose.yml ] && EXECUTOR_COMPOSE_DIR="$(pwd)"
    fi
    if [ -z "$EXECUTOR_COMPOSE_DIR" ]; then
        fail "Executor containers found but cannot determine compose directory."
        fail "Stop them manually: docker compose down (from executor directory)"
        exit 1
    fi
    warn "Executor containers will be stopped and restarted after setup."
    docker compose -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" down 2>/dev/null \
        || { fail "Failed to stop executor. Run 'docker compose down' manually from: $EXECUTOR_COMPOSE_DIR"; exit 1; }
    ok "Executor stopped ($EXECUTOR_COMPOSE_DIR)."
fi

# Check for stopped containers (sysbox requires zero containers)
stopped=$(docker ps -a -q 2>/dev/null || true)
if [ -n "$stopped" ]; then
    warn "Removing stopped containers (sysbox requires none)..."
    docker rm -f $stopped > /dev/null 2>&1
    ok "Stopped containers removed."
fi

ok "Docker is clear for sysbox installation."

# ── 4. Install ──────────────────────────────────────────

step 3 7 "Installing packages"

apt-get update -qq 2>/dev/null
apt-get install -y -qq nvidia-container-toolkit jq > /dev/null 2>&1
ok "nvidia-container-toolkit, jq"

if [ "$SKIP_INSTALL" = false ]; then
    LOCAL_DEB="./sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
    if [ -f "$LOCAL_DEB" ]; then
        SYSBOX_DEB="$LOCAL_DEB"
    else
        SYSBOX_DEB=$(mktemp /tmp/sysbox-ce.XXXXXX.deb)
        DOWNLOADED_DEB="$SYSBOX_DEB"
        ok "Downloading sysbox v${SYSBOX_VERSION}..."
        wget -q -O "$SYSBOX_DEB" "$SYSBOX_DEB_URL"
        actual=$(sha256sum "$SYSBOX_DEB" | cut -d' ' -f1)
        [ "$actual" = "$SYSBOX_SHA" ] || { fail "Checksum mismatch!"; exit 1; }
    fi
    apt-get install -y -qq "$SYSBOX_DEB" > /dev/null 2>&1
    ok "Sysbox v${SYSBOX_VERSION} installed."
else
    ok "Sysbox already installed, skipping."
fi

# ── 5. Configure Docker ────────────────────────────────

step 4 7 "Configuring Docker"

CONFIG='{"runtimes":{"sysbox-runc":{"path":"/usr/bin/sysbox-runc"},"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}},"features":{"cdi":false}}'

mkdir -p /etc/docker
if [ -f /etc/docker/daemon.json ]; then
    cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
    jq --argjson p "$CONFIG" '. * $p' /etc/docker/daemon.json > /tmp/daemon.json.tmp \
        || { fail "Failed to merge daemon.json (invalid JSON?). Backup: daemon.json.bak"; exit 1; }
    [ -s /tmp/daemon.json.tmp ] || { fail "Merged daemon.json is empty. Backup: daemon.json.bak"; exit 1; }
    mv /tmp/daemon.json.tmp /etc/docker/daemon.json
    ok "Merged into daemon.json (backup: daemon.json.bak)."
else
    echo "$CONFIG" | jq . > /etc/docker/daemon.json
    ok "Created daemon.json."
fi

if docker_version_ge 29 2; then
    rm -f /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml
    systemctl disable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service 2>/dev/null || true
    ok "Cleaned up CDI specs (Docker >= 29.2)."
fi

# ── 6. Restart Docker ──────────────────────────────────

step 5 7 "Restarting Docker"

systemctl restart docker
for _ in $(seq 1 30); do docker ps &>/dev/null && break; sleep 1; done
docker ps &>/dev/null || { fail "Docker failed to restart. Check: journalctl -u docker.service"; exit 1; }
ok "Docker is running."

# ── 7. Verify ──────────────────────────────────────────

step 6 7 "Verifying sysbox + GPU"

if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
    echo ""
    echo -e "  ${G}╔══════════════════════════════════════════╗${N}"
    echo -e "  ${G}║  SUCCESS: Sysbox is working with GPUs!   ║${N}"
    echo -e "  ${G}╚══════════════════════════════════════════╝${N}"
    echo ""
    ok "Your executor now supports Docker-in-Docker."

    # ── 8. Restart executor ──────────────────────────────
    if [ -n "$EXECUTOR_COMPOSE_DIR" ]; then
        step 7 7 "Restarting executor"
        docker compose -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" up -d 2>/dev/null \
            && ok "Executor restarted ($EXECUTOR_COMPOSE_DIR)." \
            || warn "Failed to restart executor. Run manually: cd $EXECUTOR_COMPOSE_DIR && docker compose up -d"
    fi
else
    echo ""
    fail "Verification FAILED. Diagnostics:"
    echo ""
    echo "    Docker:              $(docker --version 2>/dev/null || echo unknown)"
    echo "    NVIDIA driver:       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo FAILED)"
    echo "    /proc/driver/nvidia: $(ls /proc/driver/nvidia &>/dev/null && echo exists || echo MISSING)"
    echo "    sysbox-runc:         $(sysbox-runc --version 2>/dev/null | head -1 || echo 'not found')"
    echo "    CDI specs:           $(ls /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml 2>/dev/null || echo none)"
    echo "    daemon.json cdi:     $(jq -r '.features.cdi // "not set"' /etc/docker/daemon.json 2>/dev/null)"
    echo ""
    fail "Check: journalctl -u docker.service --no-pager -n 20"
    fail "Check: journalctl -u sysbox-mgr.service --no-pager -n 20"
    exit 1
fi
