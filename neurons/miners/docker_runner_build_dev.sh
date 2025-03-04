#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-miner-runner:$TAG"

docker build --file Dockerfile.runner.dev -t $IMAGE_NAME .