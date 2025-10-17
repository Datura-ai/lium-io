#!/usr/bin/env bash
# Purpose: Install minimal executor prerequisites on Ubuntu/Debian, 100% non-interactive.
# Behavior: Fail fast, clear errors, refuse on non-apt systems or when sudo would prompt.

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
    -c|--confirm) CONFIRM=1 ;;       # require RETURN before actions (opt-in only)
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
require_cmd apt-get

# Must have sudo or be root, and sudo must be passwordless for non-interactive runs
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  require_cmd sudo
  if ! sudo -n true 2>/dev/null; then
    die "Passwordless sudo required (or run as root). Aborting to remain non-interactive."
  fi
  SUDO="sudo -n"
else
  SUDO=""
fi

# Detect distro info for repo setup
DISTRO_ID="unknown"; DISTRO_CODENAME=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  DISTRO_ID="${ID:-unknown}"
  DISTRO_CODENAME="${VERSION_CODENAME:-}"
  UBUNTU_CODENAME="${UBUNTU_CODENAME:-${UBUNTU_CODENAME:-}}"
  ID_LIKE="${ID_LIKE:-}"
fi
log "Detected: ${DISTRO_ID} ${DISTRO_CODENAME:-?}"

export DEBIAN_FRONTEND=noninteractive
# Auto-accept service restarts during package upgrades
export NEEDRESTART_MODE=a

APT="$SUDO apt-get -y -qq -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold"
APT_INSTALL="$APT install --no-install-recommends"

# =============== Steps ===============
apt_refresh() {
  log "Refreshing apt metadata (with retries)"
  local tries=0 max=5
  until $APT update; do
    tries=$((tries+1))
    (( tries >= max )) && die "apt update failed after ${max} attempts"
    sleep $((2 * tries))
  done
  ok "apt updated"
}

install_base() {
  log "Installing base packages"
  $APT_INSTALL ca-certificates curl git build-essential apt-transport-https gnupg lsb-release software-properties-common
  ok "Base packages installed"
}

USER_ADDED_TO_DOCKER_GROUP=0

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    ok "Docker already installed: $(docker --version 2>/dev/null | head -n1)"
    return
  fi
  log "Installing Docker Engine"

  # Determine repo family and codename
  local family codename
  if [[ "$DISTRO_ID" == "ubuntu" || -n "${UBUNTU_CODENAME:-}" ]]; then
    family="ubuntu"
    codename="${UBUNTU_CODENAME:-${DISTRO_CODENAME:-$($SUDO lsb_release -cs 2>/dev/null || echo focal)}}"
  elif [[ "$DISTRO_ID" == "debian" || "$ID_LIKE" == *debian* ]]; then
    family="debian"
    codename="${DISTRO_CODENAME:-$($SUDO lsb_release -cs 2>/dev/null || echo bookworm)}"
  else
    die "Unsupported or unrecognized Debian/Ubuntu derivative (${DISTRO_ID})."
  fi

  # Keyring dir
  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${family}/gpg" | $SUDO gpg --yes --dearmor -o /etc/apt/keyrings/docker.gpg
  $SUDO chmod a+r /etc/apt/keyrings/docker.gpg

  # Repo
  echo "deb [arch=$($SUDO dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${family} ${codename} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null

  apt_refresh
  $APT install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  # Ensure daemon enabled/started on systemd hosts
  if command -v systemctl >/dev/null 2>&1; then
    $SUDO systemctl enable --now docker || true
  fi

  # Add current user to docker group if not root
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    if ! id -nG "${SUDO_USER:-$USER}" | grep -qw docker; then
      $SUDO usermod -aG docker "${SUDO_USER:-$USER}" || true
      USER_ADDED_TO_DOCKER_GROUP=1
      ok "Added ${SUDO_USER:-$USER} to docker group (activating for this session)"
    fi
  fi
  ok "Docker installed"
}

verify_docker() {
  log "Verifying Docker daemon"

  # If user was just added to docker group, use sg to activate it for verification
  if [[ $USER_ADDED_TO_DOCKER_GROUP -eq 1 ]]; then
    if ! sg docker -c "docker info" >/dev/null 2>&1; then
      die "Docker daemon not reachable. Ensure 'systemctl status docker' is healthy."
    fi
    sg docker -c "docker compose version" >/dev/null 2>&1 || die "docker compose plugin not available."
  else
    if ! docker info >/dev/null 2>&1; then
      die "Docker daemon not reachable. Ensure 'systemctl status docker' is healthy."
    fi
    docker compose version >/dev/null 2>&1 || die "docker compose plugin not available."
  fi

  ok "Docker daemon reachable & compose available"
}

# =============== Run ===============
apt_refresh
pause_if_needed

install_base
pause_if_needed

install_docker
pause_if_needed

verify_docker
pause_if_needed

ok "All required executor prerequisites are installed (non-interactive)."

# If we added user to docker group, activate it for the current session
if [[ $USER_ADDED_TO_DOCKER_GROUP -eq 1 ]]; then
  log "Activating docker group for current session..."
  exec sg docker -c "$SHELL"
fi
