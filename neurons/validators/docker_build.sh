#!/bin/bash
set -eux -o pipefail

# the Dockerfile runs the protocol release check on every build; only non-latest (dev) tags opt out
ALLOW_PROTOCOL_AHEAD=0
if [ "$TAG" != "latest" ]; then
  ALLOW_PROTOCOL_AHEAD=1
fi

IMAGE_NAME="daturaai/compute-subnet-validator:$TAG"

docker build --build-arg ALLOW_PROTOCOL_AHEAD=$ALLOW_PROTOCOL_AHEAD --build-context datura=../../datura --build-context executor_setup=../executor -t $IMAGE_NAME .
