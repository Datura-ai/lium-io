#!/bin/bash
set -eux -o pipefail

source ./docker_runner_build_dev.sh

echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
docker push "$IMAGE_NAME"