#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-validator:latest"

docker build --build-context datura=../../datura -t $IMAGE_NAME .