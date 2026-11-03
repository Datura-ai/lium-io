#!/bin/sh
set -eu

# The executor reserves every GPU (docker-compose.app.yml), so the NVIDIA container runtime hook
# runs when it starts. After a driver package update without a reboot that hook fails ("driver/library
# version mismatch") and a recreated executor never comes back: port 8080 stays closed and a rented
# node goes offline (12 Oct 2026, executor-v1.139). Before replacing a RUNNING executor, start a
# throwaway container from the image it already runs (local, nothing pulled) with every GPU requested
# and the entrypoint `true`, the same probe daturaai/lium-watchtower runs before it updates the runner.
# If that fails, keep the running executor and exit; the restart policy brings the runner back to try
# again; a probe that hangs past RUNNER_PROBE_TIMEOUT_SECONDS counts as failed. A first install (no executor
# running) has nothing to break and skips the probe.
running_executor=$(docker compose ps --quiet executor 2>/dev/null || true)
if [ -n "$running_executor" ]; then
    # A backup or restore runs as storage_runner.py inside the executor; recreating the container kills it and
    # the renter's backup fails as "stopped reporting progress" (26 backups at executor-v1.167, 2 Nov 21Z).
    case "$(docker top "$running_executor" -o args 2>/dev/null || true)" in
        *storage_runner.py*)
            echo "Update held, running executor kept: a backup or restore is running in it. Retrying in ${RUNNER_STORAGE_RETRY_SECONDS:-120}s."
            sleep "${RUNNER_STORAGE_RETRY_SECONDS:-120}"
            exit 1 ;;
    esac
    # `timeout` stops only the docker client: a hung probe container outlives it, and every retry would add one.
    leftovers=$(docker ps --all --quiet --filter label=io.lium.executor-runner.probe=nvidia-runtime 2>/dev/null || true)
    if [ -n "$leftovers" ] && ! docker rm --force $leftovers >/dev/null 2>&1; then
        echo "Update held, running executor kept: a GPU probe container from an earlier try cannot be removed ($(echo $leftovers)); reboot the host once no renter is on it. Retrying in ${RUNNER_PROBE_RETRY_SECONDS:-300}s."
        sleep "${RUNNER_PROBE_RETRY_SECONDS:-300}"
        exit 1
    fi
    probe_image=$(docker inspect --format '{{.Image}}' "$running_executor" 2>/dev/null || true)
    if [ -n "$probe_image" ] && ! probe_error=$(timeout "${RUNNER_PROBE_TIMEOUT_SECONDS:-120}" docker run --rm --gpus all --label io.lium.executor-runner.probe=nvidia-runtime --entrypoint true "$probe_image" 2>&1); then
        case "$probe_error" in
            *"driver/library version mismatch"*)
                cause="the host's NVIDIA driver and its libraries do not match (driver updated without a reboot)"
                remedy="reboot the host or reload the NVIDIA driver once no renter is on it" ;;
            *)
                cause="the NVIDIA runtime cannot start a GPU container"
                remedy="fix the NVIDIA container runtime on the host (nvidia-smi, nvidia-container-toolkit)" ;;
        esac
        echo "Update held, running executor kept: $cause; $remedy. Retrying in ${RUNNER_PROBE_RETRY_SECONDS:-300}s. Probe error: $(echo "$probe_error" | tail -n 1)"
        sleep "${RUNNER_PROBE_RETRY_SECONDS:-300}"
        exit 1
    fi
fi

docker compose up --pull always --detach --wait --force-recreate

# The executor is pulled by digest, so its old images are not dangling and `image prune` keeps them
# (2-5 GB per release, every past release too). Remove every executor image but the one now running;
# Docker refuses to remove an image a container still uses.
current_image=$(docker inspect --format '{{.Image}}' "$(docker compose ps --quiet executor 2>/dev/null)" 2>/dev/null || true)
# The validator's probes and Jupyter copies run the :latest tag. Tag the running image first, so removing the
# previous image (which drops its :latest) never leaves them pulling from the registry; no tag, no cleanup.
if [ -n "$current_image" ] && docker tag "$current_image" daturaai/compute-subnet-executor:latest; then
    for image in $(docker image ls --quiet --no-trunc daturaai/compute-subnet-executor 2>/dev/null || true); do
        if [ "$image" != "$current_image" ] && docker image rm "$image" >/dev/null 2>&1; then
            echo "Removed old executor image $image"
        fi
    done
fi

# Clean docker images
docker image prune -f

while true
do
    docker compose logs -f
    echo 'All containers died'
    sleep 10
done
