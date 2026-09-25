#!/bin/bash
set -eux -o pipefail

source ./docker_build.sh

docker push "$IMAGE_NAME"

docker rmi "$IMAGE_NAME"
docker builder prune -f