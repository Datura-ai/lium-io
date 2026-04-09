#!/bin/bash
set -eo pipefail

# ── Terminal colors & symbols ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

SYM_OK="✔"
SYM_ERR="✖"
SYM_ARROW="➜"
SYM_INFO="ℹ"
SYM_WARN="⚠"
SYM_STAR="✦"
SYM_BUILD="🔨"
SYM_DONE="🎉"

# ── Logging helpers ────────────────────────────────────────────────────────────
log_header() {
  local title="$1"
  local width=60
  local line
  line=$(printf '═%.0s' $(seq 1 $width))
  echo -e "\n${BOLD}${BLUE}${line}${RESET}"
  echo -e "${BOLD}${BLUE}  ${SYM_STAR}  ${title}${RESET}"
  echo -e "${BOLD}${BLUE}${line}${RESET}"
}

log_step() {
  echo -e "\n${CYAN}${SYM_ARROW}  ${BOLD}$1${RESET}"
}

log_success() {
  echo -e "${GREEN}  ${SYM_OK}  $1${RESET}"
}

log_info() {
  echo -e "${DIM}  ${SYM_INFO}  $1${RESET}"
}

log_warn() {
  echo -e "${YELLOW}  ${SYM_WARN}  $1${RESET}"
}

log_error() {
  echo -e "${RED}  ${SYM_ERR}  ERROR: $1${RESET}" >&2
}

log_kv() {
  printf "${DIM}  %-28s${RESET}${BOLD}%s${RESET}\n" "$1" "$2"
}

# ── Resolve script directory ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_OVERRIDE_FILE="${SCRIPT_DIR}/src/core/config_override.py"

# ── Banner ─────────────────────────────────────────────────────────────────────
log_header "Lium Executor — Image Build"
echo -e "${DIM}  Script  : ${BASH_SOURCE[0]}${RESET}"
echo -e "${DIM}  Date    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')${RESET}"

# ── Validate required environment variables ────────────────────────────────────
log_step "Validating environment variables"

missing=()
[[ -z "${TAG:-}" ]] && missing+=("TAG")

if [[ ${#missing[@]} -gt 0 ]]; then
  log_error "The following required environment variables are not set:"
  for var in "${missing[@]}"; do
    echo -e "    ${RED}•  ${var}${RESET}" >&2
  done
  echo -e "\n${YELLOW}  Usage example:${RESET}"
  echo -e "  ${DIM}TAG=latest [VALIDATOR_HOTKEY_SS58=<hotkey>] bash docker_build.sh${RESET}\n"
  exit 1
fi

log_success "All required environment variables are set"
log_kv "TAG:" "${TAG}"

IMAGE_NAME="daturaai/compute-subnet-executor:${TAG}"

# ── Optionally generate config_override.py ────────────────────────────────────
log_step "Validator hotkey configuration"

if [[ -n "${VALIDATOR_HOTKEY_SS58:-}" ]]; then
  log_info "VALIDATOR_HOTKEY_SS58 is set — generating config_override.py"
  echo "_VALIDATOR_HOTKEY_SS58 = \"${VALIDATOR_HOTKEY_SS58}\"" > "${CONFIG_OVERRIDE_FILE}"
  log_success "Generated: ${CONFIG_OVERRIDE_FILE}"
  log_kv "Validator hotkey:" "${VALIDATOR_HOTKEY_SS58}"
else
  log_warn "VALIDATOR_HOTKEY_SS58 not set — skipping config_override.py generation"
  if [[ -f "${CONFIG_OVERRIDE_FILE}" ]]; then
    existing_hotkey=$(python3 -c "import ast, sys; tree=ast.parse(open('${CONFIG_OVERRIDE_FILE}').read()); [sys.stdout.write(n.value.s) for n in ast.walk(tree) if isinstance(n, ast.Assign)]" 2>/dev/null || true)
    log_info "Using existing config_override.py"
    [[ -n "$existing_hotkey" ]] && log_kv "Validator hotkey:" "${existing_hotkey}"
  else
    log_info "No config_override.py found — will use hardcoded default in config.py"
  fi
fi

# ── Build executor Docker image ────────────────────────────────────────────────
log_step "${SYM_BUILD}  Building executor image"
log_kv "Image name:"    "${IMAGE_NAME}"
log_kv "Dockerfile:"    "Dockerfile"
log_kv "Build context:" "${SCRIPT_DIR}"
echo ""

docker build \
  --build-context datura=../../datura \
  --tag "${IMAGE_NAME}" \
  "${SCRIPT_DIR}"

# ── Success summary ────────────────────────────────────────────────────────────
log_header "${SYM_DONE}  Build Complete"
echo -e "${GREEN}${BOLD}"
log_kv "Executor image:" "${IMAGE_NAME}"
echo -e "${RESET}"
