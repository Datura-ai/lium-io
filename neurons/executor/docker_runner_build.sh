#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-executor-runner:$TAG"

docker build --file Dockerfile.runner -t $IMAGE_NAME .