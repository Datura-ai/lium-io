#!/bin/bash
set -eo pipefail

# ── Colors & symbols ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

SYM_ARROW="➜"
SYM_OK="✔"
SYM_STAR="✦"

log_step()    { echo -e "\n${CYAN}${SYM_ARROW}  ${BOLD}$1${RESET}"; }
log_success() { echo -e "${GREEN}  ${SYM_OK}  $1${RESET}"; }
log_kv()      { printf "${DIM}  %-28s${RESET}${BOLD}%s${RESET}\n" "$1" "$2"; }
log_header()  {
  local line
  line=$(printf '═%.0s' $(seq 1 60))
  echo -e "\n${BOLD}\033[0;34m${line}${RESET}"
  echo -e "${BOLD}\033[0;34m  ${SYM_STAR}  $1${RESET}"
  echo -e "${BOLD}\033[0;34m${line}${RESET}"
}

# ── Build ──────────────────────────────────────────────────────────────────────
log_header "Lium Executor — Publish Image"

# ── Login & push ───────────────────────────────────────────────────────────────
log_step "Logging in to Docker Hub"
echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
log_success "Authenticated as ${DOCKERHUB_USERNAME}"

log_step "Pushing image: ${IMAGE_NAME}"
PUSH_OUTPUT=$(docker push "$IMAGE_NAME" 2>&1 | tee /dev/stderr)

# ── Extract sha256 digest ──────────────────────────────────────────────────────
log_step "Extracting image digest"

# `docker push` prints a line like:
#   latest: digest: sha256:<hash> size: <n>
EXECUTOR_IMAGE_SHA256=$(echo "$PUSH_OUTPUT" \
  | grep -oP 'digest:\s*\Ksha256:[a-f0-9]+' \
  | tail -1)

if [[ -z "$EXECUTOR_IMAGE_SHA256" ]]; then
  # Fallback: inspect the pushed image via the registry
  EXECUTOR_IMAGE_SHA256=$(docker inspect --format='{{index .RepoDigests 0}}' "$IMAGE_NAME" \
    | grep -oP 'sha256:[a-f0-9]+' \
    | head -1)
fi

if [[ -z "$EXECUTOR_IMAGE_SHA256" ]]; then
  echo -e "\n\033[0;31m  ✖  ERROR: Could not determine image digest.\033[0m" >&2
  exit 1
fi

# ── Summary ────────────────────────────────────────────────────────────────────
log_header "Publish Complete"
log_kv "Image name    :" "${IMAGE_NAME}"
log_kv "Digest        :" "${EXECUTOR_IMAGE_SHA256}"

# Machine-readable marker — easy to grep in CI/CD logs
echo -e "\n${GREEN}${BOLD}EXECUTOR_IMAGE_SHA256=${EXECUTOR_IMAGE_SHA256}${RESET}"
