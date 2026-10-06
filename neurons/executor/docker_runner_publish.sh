#!/bin/bash
set -eo pipefail

# Login only when a caller passes a token; the lium-io workflows log in with docker/login-action (OIDC).
if [ -n "${DOCKERHUB_PAT:-}" ]; then
  echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi
docker push "$IMAGE_NAME"