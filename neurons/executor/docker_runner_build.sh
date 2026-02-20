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
SYM_CLEAN="🧹"
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

# ── Cleanup trap ───────────────────────────────────────────────────────────────
TEMP_COMPOSE_FILE=""

cleanup() {
  local exit_code=$?
  if [[ -n "$TEMP_COMPOSE_FILE" && -f "$TEMP_COMPOSE_FILE" ]]; then
    echo -e "\n${MAGENTA}  ${SYM_CLEAN}  Cleaning up temporary compose file...${RESET}"
    rm -f "$TEMP_COMPOSE_FILE"
    log_success "Removed: $(basename "$TEMP_COMPOSE_FILE")"
  fi
  if [[ $exit_code -ne 0 ]]; then
    echo -e "\n${RED}${BOLD}  ${SYM_ERR}  Build failed with exit code ${exit_code}.${RESET}\n"
  fi
}

trap cleanup EXIT

# ── Resolve script directory ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_TEMPLATE="${SCRIPT_DIR}/docker-compose.app.yml"
TEMP_COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.app.tmp.yml"

# ── Banner ─────────────────────────────────────────────────────────────────────
log_header "Lium Executor — Runner Image Build"
echo -e "${DIM}  Script  : ${BASH_SOURCE[0]}${RESET}"
echo -e "${DIM}  Date    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')${RESET}"

# ── Validate required environment variables ────────────────────────────────────
log_step "Validating environment variables"

missing=()
[[ -z "${TAG:-}"                   ]] && missing+=("TAG")
[[ -z "${EXECUTOR_IMAGE_SHA256:-}" ]] && missing+=("EXECUTOR_IMAGE_SHA256")

if [[ ${#missing[@]} -gt 0 ]]; then
  log_error "The following required environment variables are not set:"
  for var in "${missing[@]}"; do
    echo -e "    ${RED}•  ${var}${RESET}" >&2
  done
  echo -e "\n${YELLOW}  Usage example:${RESET}"
  echo -e "  ${DIM}TAG=latest EXECUTOR_IMAGE_SHA256=<sha256> bash docker_runner_build.sh${RESET}\n"
  exit 1
fi

log_success "All required environment variables are set"
log_kv "TAG:" "${TAG}"
log_kv "EXECUTOR_IMAGE_SHA256:" "${EXECUTOR_IMAGE_SHA256}"

IMAGE_NAME="daturaai/compute-subnet-executor-runner:${TAG}"

# ── Validate template file exists ──────────────────────────────────────────────
log_step "Locating compose template"

if [[ ! -f "$COMPOSE_TEMPLATE" ]]; then
  log_error "Compose template not found: ${COMPOSE_TEMPLATE}"
  exit 1
fi

log_success "Template found"
log_info "${COMPOSE_TEMPLATE}"

# ── Generate temporary compose file ───────────────────────────────────────────
log_step "Injecting executor image sha256 into compose template"

log_info "Substituting: \${EXECUTOR_IMAGE_SHA256} → ${EXECUTOR_IMAGE_SHA256}  (includes sha256: prefix)"

EXECUTOR_IMAGE_SHA256="${EXECUTOR_IMAGE_SHA256}" \
  envsubst '${EXECUTOR_IMAGE_SHA256}' \
  < "$COMPOSE_TEMPLATE" \
  > "$TEMP_COMPOSE_FILE"

# Verify the hash was injected correctly in every expected location
injected_count=$(grep -c "${EXECUTOR_IMAGE_SHA256}" "$TEMP_COMPOSE_FILE" || true)
if [[ "$injected_count" -eq 0 ]]; then
  log_error "Hash injection verification failed — EXECUTOR_IMAGE_SHA256 not found in generated file."
  log_info "Expected : ${EXECUTOR_IMAGE_SHA256}"
  exit 1
fi

log_success "Temp compose file generated (${injected_count} service(s) updated)"
log_info "Output : $(basename "$TEMP_COMPOSE_FILE")"

# Show injected image lines for confirmation
echo -e "${DIM}"
grep --color=never "sha256:" "$TEMP_COMPOSE_FILE" | while IFS= read -r line; do
  echo "    ${line}"
done
echo -e "${RESET}"

# ── Build runner Docker image ──────────────────────────────────────────────────
log_step "${SYM_BUILD}  Building runner image"
log_kv "Image name:"    "${IMAGE_NAME}"
log_kv "Dockerfile:"    "Dockerfile.runner"
log_kv "Compose file:"  "$(basename "$TEMP_COMPOSE_FILE")"
log_kv "Build context:" "${SCRIPT_DIR}"
echo ""

docker build \
  --file "${SCRIPT_DIR}/Dockerfile.runner" \
  --build-arg "targetFile=$(basename "$TEMP_COMPOSE_FILE")" \
  --tag "${IMAGE_NAME}" \
  "${SCRIPT_DIR}"

# ── Success summary ────────────────────────────────────────────────────────────
log_header "${SYM_DONE}  Build Complete"
echo -e "${GREEN}${BOLD}"
log_kv "Runner image  :" "${IMAGE_NAME}"
log_kv "Executor SHA  :" "${EXECUTOR_IMAGE_SHA256}"
echo -e "${RESET}"
