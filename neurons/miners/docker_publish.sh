#!/bin/bash
set -eux -o pipefail

source ./docker_build.sh

# Tracing off for the login: with `set -x` the shell prints this command expanded, token included.
{ set +x; } 2>/dev/null
echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
set -x
docker push "$IMAGE_NAME"