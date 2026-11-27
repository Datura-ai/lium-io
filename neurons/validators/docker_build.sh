#!/bin/bash
set -eux -o pipefail

# the prod release (latest) builds with the protocol release gate on (Dockerfile); staging and dev builds do not
if [ "$TAG" = "latest" ]; then GATE=1; else GATE=0; fi

IMAGE_NAME="daturaai/compute-subnet-validator:$TAG"

docker build --build-arg PROTOCOL_RELEASE_GATE=$GATE --build-context datura=../../datura --build-context executor_setup=../executor -t $IMAGE_NAME .
