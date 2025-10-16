#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-executor:$TAG"

docker build --build-context datura=../../datura --build-arg SSH_PUBLIC_KEY="$SSH_PUBLIC_KEY" -t $IMAGE_NAME .