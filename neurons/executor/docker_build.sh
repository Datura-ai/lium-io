#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-executor:$TAG"

docker build --build-context datura=../../datura -t $IMAGE_NAME .