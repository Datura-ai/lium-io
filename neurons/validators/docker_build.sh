#!/bin/bash
set -eux -o pipefail

IMAGE_NAME="daturaai/compute-subnet-validator:$TAG"

docker build --build-context datura=../../datura --build-context executor_src=../executor/src --build-context executor_setup=../executor -t $IMAGE_NAME .
