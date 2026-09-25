#!/bin/bash
set -eux -o pipefail

source ./docker_runner_build.sh

docker push "$IMAGE_NAME"