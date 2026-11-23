#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-miner:$TAG"

# a non-prod image trusts its own validator (core/config.py) and reads no signed validator message
if [[ -n "${VALIDATOR_HOTKEY_SS58:-}" ]]; then
  echo "_VALIDATOR_HOTKEY_SS58 = \"${VALIDATOR_HOTKEY_SS58}\"" > src/core/config_override.py
fi

docker build --build-context datura=../../datura -t $IMAGE_NAME .