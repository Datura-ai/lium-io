#!/bin/bash
set -eux -o pipefail

# the prod backend must read every message this validator sends (protocol_release_gate.py)
python3 ./protocol_release_gate.py

IMAGE_NAME="daturaai/compute-subnet-validator:$TAG"

docker build --build-context datura=../../datura -t $IMAGE_NAME .