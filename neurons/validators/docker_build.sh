#!/bin/bash
set -eux -o pipefail

# the prod release (latest) waits until the prod backend reads this validator's protocol; staging and dev builds do not
if [ "$TAG" = "latest" ]; then
  python3 protocol_release_gate.py
fi

IMAGE_NAME="daturaai/compute-subnet-validator:$TAG"

docker build --build-context datura=../../datura --build-context executor_setup=../executor -t $IMAGE_NAME .
