#!/bin/bash
set -eo pipefail

# Login only when a caller passes a token; the lium-io workflows log in with docker/login-action (OIDC).
if [ -n "${DOCKERHUB_PAT:-}" ]; then
  echo "$DOCKERHUB_PAT" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
fi
if [ -n "${PLATFORMS:-}" ]; then
  # the multi-arch OCI archive docker_runner_build.sh wrote
  skopeo copy --all --preserve-digests --digestfile /tmp/executor-runner.digest \
    "oci-archive:${RUNNER_OCI_ARCHIVE:-/tmp/executor-runner.oci.tar}" "docker://docker.io/${IMAGE_NAME}"
  echo "RUNNER_IMAGE_SHA256=$(cat /tmp/executor-runner.digest)"
else
  docker push "$IMAGE_NAME"
fi
