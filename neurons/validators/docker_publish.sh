#!/bin/bash
set -eux -o pipefail

source ./docker_build.sh

# Login only when a caller passes a token; the lium-io workflows log in with docker/login-action (OIDC).
# Tracing off first: with `set -x` the test and the login would print the token expanded.
{ set +x; } 2>/dev/null
if [ -n "${DOCKERHUB_PAT:-}" ]; then
  echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi
set -x
docker push "$IMAGE_NAME"

docker rmi "$IMAGE_NAME"
docker builder prune -f