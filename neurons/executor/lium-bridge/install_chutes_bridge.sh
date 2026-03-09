#!/usr/bin/env bash
# install_chutes_bridge.sh — one-liner installer for lium-bridge
# Usage: curl -sfL https://install.lium.io/chutes-bridge | sh
#
# This installs the bridgectl command interface on the host.
# It does NOT install K3s or Chutes — that's done by `bridgectl setup`.

set -euo pipefail

BRIDGE_DIR="/opt/lium-bridge"
BRIDGE_USER="lium-bridge"
LOG_FILE="/var/log/lium-bridge.log"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Root check ---
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (or with sudo)"
fi

info "Installing lium-bridge..."

# --- Prerequisites ---

info "Checking prerequisites..."

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. NVIDIA drivers must be installed."
fi

if ! command -v docker &>/dev/null; then
    error "Docker not found. Docker must be installed."
fi

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Python 3 is required."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
info "Detected GPU: ${GPU_NAME}"

DISK_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ "${DISK_FREE}" -lt 100 ]]; then
    warn "Only ${DISK_FREE}GB free disk space. Recommended: 200GB+"
fi

# --- Create system user ---

if id "${BRIDGE_USER}" &>/dev/null; then
    info "User ${BRIDGE_USER} already exists, skipping"
else
    info "Creating system user: ${BRIDGE_USER}"
    useradd -r -s /bin/bash -d /var/lib/${BRIDGE_USER} -m ${BRIDGE_USER}
    # Allow SSH key auth without password (password field must not be locked '!')
    usermod -p '*' ${BRIDGE_USER}
fi

# --- Create directory structure ---

info "Creating ${BRIDGE_DIR}..."
mkdir -p "${BRIDGE_DIR}/bin"

# --- Copy scripts ---
# In production these would be downloaded from a URL.
# For now, we expect them to be in the same directory as this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(bridgectl status setup-chutes start-chutes stop-chutes uninstall-chutes)

for script in "${SCRIPTS[@]}"; do
    src="${SCRIPT_DIR}/bin/${script}"
    dst="${BRIDGE_DIR}/bin/${script}"
    if [[ -f "${src}" ]]; then
        cp "${src}" "${dst}"
        chmod 755 "${dst}"
        chown root:root "${dst}"
        info "Installed: ${dst}"
    else
        warn "Script not found (skipping): ${src}"
    fi
done

# --- Initial state ---

STATE_FILE="${BRIDGE_DIR}/state.json"
if [[ ! -f "${STATE_FILE}" ]]; then
    cat > "${STATE_FILE}" <<'STATEJSON'
{"state": "not_installed", "last_error": null, "node_name": null, "gpu_verified": false}
STATEJSON
    info "Created initial state: ${STATE_FILE}"
else
    info "State file already exists, preserving: ${STATE_FILE}"
fi

chmod 644 "${STATE_FILE}"

# --- Log file ---

touch "${LOG_FILE}"
chown ${BRIDGE_USER}:${BRIDGE_USER} "${LOG_FILE}"
chmod 644 "${LOG_FILE}"

# --- SSH configuration ---

SSH_CONF="/etc/ssh/sshd_config.d/lium-bridge.conf"
info "Configuring restricted SSH access..."

# Add our SSH public key to lium-bridge user
BRIDGE_SSH_DIR="/var/lib/${BRIDGE_USER}/.ssh"
mkdir -p "${BRIDGE_SSH_DIR}"
chmod 700 "${BRIDGE_SSH_DIR}"

# If authorized_keys doesn't exist, create an empty one (keys will be added by the platform)
if [[ ! -f "${BRIDGE_SSH_DIR}/authorized_keys" ]]; then
    touch "${BRIDGE_SSH_DIR}/authorized_keys"
fi
chmod 600 "${BRIDGE_SSH_DIR}/authorized_keys"
chown -R "${BRIDGE_USER}:${BRIDGE_USER}" "${BRIDGE_SSH_DIR}"

cat > "${SSH_CONF}" <<'SSHCONF'
Match User lium-bridge
    ForceCommand /opt/lium-bridge/bin/bridgectl
    PermitTTY no
    AllowTcpForwarding no
    X11Forwarding no
SSHCONF

# Validate sshd config before reloading
if sshd -t 2>/dev/null; then
    systemctl reload sshd 2>/dev/null || systemctl reload ssh 2>/dev/null || true
    info "SSH configured: ${SSH_CONF}"
else
    rm -f "${SSH_CONF}"
    warn "SSH config validation failed, removed ${SSH_CONF}"
fi

# --- Sudoers ---

SUDOERS_FILE="/etc/sudoers.d/lium-bridge"
info "Configuring sudoers..."

cat > "${SUDOERS_FILE}" <<SUDOERS
# lium-bridge: restricted sudo for bridge scripts only
${BRIDGE_USER} ALL=(root) NOPASSWD: ${BRIDGE_DIR}/bin/setup-chutes
${BRIDGE_USER} ALL=(root) NOPASSWD: ${BRIDGE_DIR}/bin/start-chutes
${BRIDGE_USER} ALL=(root) NOPASSWD: ${BRIDGE_DIR}/bin/stop-chutes
${BRIDGE_USER} ALL=(root) NOPASSWD: ${BRIDGE_DIR}/bin/status
${BRIDGE_USER} ALL=(root) NOPASSWD: ${BRIDGE_DIR}/bin/uninstall-chutes
SUDOERS

chmod 440 "${SUDOERS_FILE}"

# Validate sudoers
if visudo -c -f "${SUDOERS_FILE}" &>/dev/null; then
    info "Sudoers configured: ${SUDOERS_FILE}"
else
    rm -f "${SUDOERS_FILE}"
    error "Sudoers validation failed!"
fi

# --- Add shadeform/current user to docker group (for bridgectl) ---

usermod -aG docker "${BRIDGE_USER}" 2>/dev/null || true

# --- Done ---

echo ""
info "lium-bridge installed successfully!"
info "  Bridge dir:  ${BRIDGE_DIR}"
info "  Scripts:     ${BRIDGE_DIR}/bin/"
info "  State:       ${STATE_FILE}"
info "  Log:         ${LOG_FILE}"
info "  SSH config:  ${SSH_CONF}"
info ""
info "Next step: bridgectl setup --validator-hotkey <ss58> --hotkey-ss58 <ss58> --hotkey-seed <hex>"
echo ""

# Quick status check
"${BRIDGE_DIR}/bin/status"
