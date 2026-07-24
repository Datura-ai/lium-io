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

kernel_supports_idmapped() {
    # overlayfs over ID-mapped mounts landed in 5.19; without it sysbox falls back to shiftfs
    local ver major minor
    ver=$(uname -r)
    major=${ver%%.*} minor=${ver#*.} minor=${minor%%.*}
    [ "$major" -gt 5 ] 2>/dev/null || { [ "$major" -eq 5 ] && [ "$minor" -ge 19 ]; } 2>/dev/null
}

sysbox_idmapped_report() {
    # what sysbox-mgr itself decided this boot: "yes", "no", or empty when unavailable
    journalctl -u sysbox-mgr -b --no-pager 2>/dev/null \
        | grep -o 'Overlayfs on ID-mapped mounts supported by kernel: [a-z]*' \
        | tail -1 | awk '{print $NF}'
}

nvidia_hook_symptom() {
    fail "Sysbox then falls back to shiftfs and runs the container rootfs from another path, so the NVIDIA hook fails with:"
    fail "  nvidia-container-cli: mount error: .../merged/proc/driver/nvidia: no such file or directory"
}

fail_old_kernel() {
    fail "Kernel $(uname -r) is too old for sysbox with GPUs — overlayfs gained ID-mapped mounts in 5.19 (6.x recommended)."
    nvidia_hook_symptom
    fail "Fix on Ubuntu 22.04: sudo apt-get install -y linux-generic-hwe-22.04 && sudo reboot"
    fail "  Ubuntu 20.04 tops out at 5.15 even with HWE — upgrade the distro instead."
    fail "Before rebooting: stop any rentals, and confirm the NVIDIA driver is DKMS-managed ('dkms status'),"
    fail "otherwise a .run-installed driver will not load on the new kernel and the node comes back without GPUs."
}

fail_no_idmapped() {
    fail "Sysbox reports it cannot use ID-mapped mounts, although kernel $(uname -r) supports them."
    nvidia_hook_symptom
    fail "Usual causes: Docker's data-root sits on a filesystem without ID-map support (ZFS, some btrfs setups),"
    fail "or sysbox-mgr runs with ID-mapped mounts disabled. Check:"
    fail "  docker info --format '{{.Driver}}'   (overlay2 expected)"
    fail "  journalctl -u sysbox-mgr -b | grep -i id-mapped"
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

if command -v sysbox-runc &>/dev/null && docker info 2>/dev/null | grep -q sysbox-runc; then
    # pull first: without the image the real test cannot run and the host would be judged on kernel version alone
    if ! docker image inspect "$VERIFY_IMAGE" &>/dev/null; then
        echo "  Pulling $VERIFY_IMAGE to test the current setup..."
        docker pull "$VERIFY_IMAGE" &>/dev/null || true
    fi
    if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
        ok "Sysbox is already working. Nothing to do."
        exit 0
    fi
fi

# Reached only when the real GPU test above did not pass, so a working host is never rejected here.
IDMAPPED=$(sysbox_idmapped_report)
case "$IDMAPPED" in
    yes) ;;
    no)
        if kernel_supports_idmapped; then fail_no_idmapped; else fail_old_kernel; fi
        exit 1
        ;;
    *)  # sysbox-mgr reported nothing this boot — fall back to the kernel version
        if ! kernel_supports_idmapped; then
            fail_old_kernel
            exit 1
        fi
        ;;
esac

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

find_executor_compose() {
    # 1. Label from container
    [ -n "$EXECUTOR_COMPOSE_DIR" ] && [ -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" ] && return 0
    # 2. Current directory
    [ -f ./docker-compose.yml ] && EXECUTOR_COMPOSE_DIR="$(pwd)" && return 0
    # 3. Known paths
    for d in \
        "$HOME/compute-subnet/neurons/executor" \
        /root/compute-subnet/neurons/executor \
        /home/*/compute-subnet/neurons/executor \
        "$HOME/executor" \
        /root/executor; do
        [ -f "$d/docker-compose.yml" ] && EXECUTOR_COMPOSE_DIR="$d" && return 0
    done
    return 1
}

if [ "$has_executor" = true ]; then
    find_executor_compose || EXECUTOR_COMPOSE_DIR=""
    warn "Executor containers will be stopped and restarted after setup."
    if [ -n "$EXECUTOR_COMPOSE_DIR" ] && [ -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" ]; then
        docker compose -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" down 2>/dev/null \
            || { fail "Failed to stop executor. Run 'docker compose down' manually from: $EXECUTOR_COMPOSE_DIR"; exit 1; }
        ok "Executor stopped via compose ($EXECUTOR_COMPOSE_DIR)."
    else
        # Compose file missing — stop and remove executor containers directly
        executor_ids=$(docker ps --filter "name=executor" -q 2>/dev/null || true)
        if [ -n "$executor_ids" ]; then
            docker stop $executor_ids > /dev/null 2>&1
            docker rm $executor_ids > /dev/null 2>&1
        fi
        ok "Executor containers stopped (compose file not found — restart manually after setup)."
        EXECUTOR_COMPOSE_DIR=""  # Clear so we don't try to restart
    fi
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
    echo "    Kernel:              $(uname -r)"
    echo "    ID-mapped mounts:    $(sysbox_idmapped_report | grep . || echo 'not reported by sysbox-mgr')"
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
