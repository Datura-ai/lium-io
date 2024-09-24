#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-miner:latest"

docker build -t $IMAGE_NAME .