#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-miner:$TAG"

# a non-prod image trusts its own validator (core/config.py); with a staging owner key it also follows messages that
# key signed (scripts/staging_trust_override.sh), else it reads no signed validator message
if [[ -n "${STAGING_TRUST_OWNER_SS58:-}" && -z "${VALIDATOR_HOTKEY_SS58:-}" ]]; then
  echo "STAGING_TRUST_OWNER_SS58 is set but VALIDATOR_HOTKEY_SS58 is not; only a non-prod build names a staging owner" >&2
  exit 1
fi
if [[ -n "${VALIDATOR_HOTKEY_SS58:-}" ]]; then
  echo "_VALIDATOR_HOTKEY_SS58 = \"${VALIDATOR_HOTKEY_SS58}\"" > src/core/config_override.py
  source ../../scripts/staging_trust_override.sh
  staging_trust_override src/core/config_override.py
fi

docker build --build-context datura=../../datura -t $IMAGE_NAME .