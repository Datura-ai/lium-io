#!/bin/bash
set -eux -o pipefail

# a prod release (TAG=latest) must speak no newer protocol than the prod backend reads; dev builds are not gated
if [ "$TAG" = "latest" ]; then
  python3 ./protocol_release_gate.py
fi

IMAGE_NAME="daturaai/compute-subnet-validator:$TAG"

docker build --build-context datura=../../datura -t $IMAGE_NAME .