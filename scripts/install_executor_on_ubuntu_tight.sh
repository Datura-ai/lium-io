#!/usr/bin/env bash
# Purpose: Install minimal executor prerequisites on Ubuntu/Debian.
# Behavior: Fail fast, clear errors, refuse on non-apt systems.

set -Eeuo pipefail

# =============== UX helpers ===============
RED=$'\e[1;31m'; GRN=$'\e[1;32m'; BLU=$'\e[1;34m'; BLD=$'\e[1m'; RST=$'\e[0m'
log()   { printf "${BLU}==>${RST} %s\n" "$*"; }
ok()    { printf "${GRN}✓${RST} %s\n" "$*"; }
die()   { printf "${RED}✗ %s${RST}\n" "$*" >&2; exit 1; }

trap 'die "Failed at: ${BASH_COMMAND}"' ERR

CONFIRM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) CONFIRM=0 ;;          # legacy no-op (always non-interactive)
    -c|--confirm) CONFIRM=1 ;;       # require RETURN before actions
    *) die "Unknown flag: $1" ;;
  esac
  shift
done

pause_if_needed() {
  [[ $CONFIRM -eq 1 ]] || return 0
  read -r -p "Press RETURN to continue or Ctrl+C to abort..." _
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# =============== Guards ===============
# 1) Must have apt-get (Ubuntu/Debian). Fail early on Arch, etc.
command -v apt-get >/dev/null 2>&1 || die "This installer supports only Debian/Ubuntu (apt-get not found)."

# 2) Must have sudo or be root
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  require_cmd sudo
fi

# 3) Detect distro info for repo setup
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  DISTRO_ID="${ID:-unknown}"
  DISTRO_CODENAME="${VERSION_CODENAME:-}"
else
  DISTRO_ID="unknown"; DISTRO_CODENAME=""
fi

log "Detected: ${DISTRO_ID} ${DISTRO_CODENAME}"

export DEBIAN_FRONTEND=noninteractive
APT="sudo apt-get -y -qq"
APT_INSTALL="$APT install --no-install-recommends"

# =============== Steps ===============
apt_refresh() {
  log "Refreshing apt metadata"
  $APT update
  ok "apt updated"
}

install_base() {
  log "Installing base packages"
  $APT_INSTALL ca-certificates curl git build-essential apt-transport-https gnupg lsb-release software-properties-common
  ok "Base packages installed"
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    ok "Docker already installed: $(docker --version 2>/dev/null | head -n1)"
    return
  fi
  log "Installing Docker Engine"

  # Keyring dir
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  # Repo (use ubuntu repo for Debian derivatives with matching codename if present)
  CODENAME="${DISTRO_CODENAME:-$(lsb_release -cs 2>/dev/null || echo focal)}"
  echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
 ${CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  $APT update
  $APT install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  # Add current user to docker group if not root and user exists
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    sudo usermod -aG docker "${SUDO_USER:-$USER}" || true
    ok "Added ${SUDO_USER:-$USER} to docker group (log out/in to take effect)"
  fi
  ok "Docker installed"
}

verify_docker() {
  log "Verifying Docker daemon"
  # Try to ping the daemon (this fails if user session needs re-login after group change)
  if ! docker info >/dev/null 2>&1; then
    die "Docker daemon not reachable. If you were just added to the docker group, log out and back in, then retry."
  fi

  # Check compose v2 is available
  if ! docker compose version >/dev/null 2>&1; then
    die "docker compose plugin not available."
  fi
  ok "Docker daemon reachable & compose available"
}

optional_python_tools() {
  # Keep it minimal; don’t hard-pin 3.11 unless you must.
  if ! command -v python3 >/dev/null 2>&1; then
    log "Installing Python"
    $APT_INSTALL python3 python3-pip python3-venv
    ok "Python installed"
  else
    ok "Python present: $(python3 --version)"
  fi
}

# (Optional) redis/postgres/btcli—only if you really require them for the executor host.
# Wire these behind flags later if needed.

# =============== Run ===============
apt_refresh
pause_if_needed

install_base
pause_if_needed

install_docker
pause_if_needed

verify_docker
pause_if_needed

optional_python_tools

ok "All required executor prerequisites are installed."
