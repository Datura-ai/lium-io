#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Prepare this executor checkout for staging Chutes relay activation.

Usage:
  ./chutes-install.sh [options]

Options:
  --env-file PATH              Env file to update. Default: ./.env
  --stage-secrets-dir PATH     Host directory for relay key and known_hosts. Default: ./stage-secrets
  --bridge-host HOST           Host-reachable bridge address. Default: host.docker.internal
  --bridge-port PORT           Bridge SSH port. Default: 22
  --bridge-user USER           Bridge SSH user. Default: lium-bridge
  --connect-timeout SEC        SSH connect timeout. Default: 10
  --command-timeout SEC        Bridge command timeout. Default: 300
  --help, -h                   Show this help message

This script does not:
  - install lium-bridge on the GPU host
  - add the executor relay key to the host authorized_keys
  - copy secret material into stage-secrets/
  - restart docker compose automatically
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/.env"
DEFAULT_ENV_TEMPLATE="${SCRIPT_DIR}/.env.template"

ENV_FILE="${DEFAULT_ENV_FILE}"
STAGE_SECRETS_DIR="./stage-secrets"
BRIDGE_HOST="host.docker.internal"
BRIDGE_PORT="22"
BRIDGE_USER="lium-bridge"
CONNECT_TIMEOUT="10"
COMMAND_TIMEOUT="300"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --stage-secrets-dir)
            STAGE_SECRETS_DIR="$2"
            shift 2
            ;;
        --bridge-host)
            BRIDGE_HOST="$2"
            shift 2
            ;;
        --bridge-port)
            BRIDGE_PORT="$2"
            shift 2
            ;;
        --bridge-user)
            BRIDGE_USER="$2"
            shift 2
            ;;
        --connect-timeout)
            CONNECT_TIMEOUT="$2"
            shift 2
            ;;
        --command-timeout)
            COMMAND_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${BRIDGE_HOST}" ]]; then
    echo "--bridge-host must not be empty" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ ! -f "${DEFAULT_ENV_TEMPLATE}" ]]; then
        echo "Missing env template: ${DEFAULT_ENV_TEMPLATE}" >&2
        exit 1
    fi
    cp "${DEFAULT_ENV_TEMPLATE}" "${ENV_FILE}"
    echo "Created ${ENV_FILE} from .env.template"
fi

mkdir -p "${STAGE_SECRETS_DIR}"
chmod 700 "${STAGE_SECRETS_DIR}"

upsert_env() {
    local key="$1"
    local value="$2"
    local escaped

    escaped="$(printf '%s' "${value}" | sed 's/[&|]/\\&/g')"
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i.bak "s|^${key}=.*|${key}=${escaped}|" "${ENV_FILE}"
    else
        printf '\n%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
    fi
}

upsert_env "CHUTES_BRIDGE_ENABLED" "true"
upsert_env "CHUTES_BRIDGE_SSH_HOST" "${BRIDGE_HOST}"
upsert_env "CHUTES_BRIDGE_SSH_PORT" "${BRIDGE_PORT}"
upsert_env "CHUTES_BRIDGE_SSH_USER" "${BRIDGE_USER}"
upsert_env "CHUTES_BRIDGE_SSH_KEY_PATH" "/run/secrets/chutes_bridge_key"
upsert_env "CHUTES_BRIDGE_CONNECT_TIMEOUT_SEC" "${CONNECT_TIMEOUT}"
upsert_env "CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC" "${COMMAND_TIMEOUT}"
upsert_env "CHUTES_BRIDGE_SSH_KEY_HOST_PATH" "${STAGE_SECRETS_DIR}/chutes_bridge_key"
upsert_env "CHUTES_BRIDGE_SSH_KNOWN_HOSTS_HOST_PATH" "${STAGE_SECRETS_DIR}/chutes_bridge_known_hosts"

rm -f "${ENV_FILE}.bak"

cat <<EOF

Prepared executor relay activation in:
  Env file: ${ENV_FILE}
  Stage secrets dir: ${STAGE_SECRETS_DIR}

Configured values:
  CHUTES_BRIDGE_ENABLED=true
  CHUTES_BRIDGE_SSH_HOST=${BRIDGE_HOST}
  CHUTES_BRIDGE_SSH_PORT=${BRIDGE_PORT}
  CHUTES_BRIDGE_SSH_USER=${BRIDGE_USER}
  CHUTES_BRIDGE_SSH_KEY_PATH=/run/secrets/chutes_bridge_key
  CHUTES_BRIDGE_CONNECT_TIMEOUT_SEC=${CONNECT_TIMEOUT}
  CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC=${COMMAND_TIMEOUT}

Next steps:
  1. On the GPU host, run sudo ./install_chutes_bridge.sh if lium-bridge is not installed yet.
  2. Add the executor relay public key to /var/lib/lium-bridge/.ssh/authorized_keys on the GPU host.
  3. Put the matching private key at ${STAGE_SECRETS_DIR}/chutes_bridge_key and chmod 600 it.
  4. Put the host fingerprint at ${STAGE_SECRETS_DIR}/chutes_bridge_known_hosts.
  5. Restart executor:
       docker compose -f docker-compose.app.yml -f docker-compose.chutes-stage.override.yml up -d --force-recreate executor
  6. Smoke-test GET /chutes/status, then run a signed POST /chutes/install via the Phase 1 scripts.
EOF
