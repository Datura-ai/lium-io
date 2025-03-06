#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-miner-runner:$TAG"

docker build --file Dockerfile.runner --build-arg targetFile=$TARGET_FILE -t $IMAGE_NAME .