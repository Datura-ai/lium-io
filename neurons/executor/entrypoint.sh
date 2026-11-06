#!/bin/sh
set -eu

# The runner keeps the stack's updater current, so a change of updater reaches every host without a manual
# step. A stack started from an older compose file still runs `nickfedor/watchtower` or `containrrr/watchtower`,
# which pull by tag (DAH-3419); the runner is the one container every updater recreates. Only an updater
# container of the runner's own compose project is replaced, never created: a stack without one (the dstack
# TEE) is left alone. If the new updater cannot start, the old one is put back.
UPDATER_IMAGE="daturaai/lium-watchtower:1.2.0"
ensure_updater() {
    [ -n "${HOSTNAME:-}" ] || return 0
    project=$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$HOSTNAME" 2>/dev/null) || return 0
    [ -n "$project" ] || return 0
    updater=$(docker ps --all --quiet --filter "label=com.docker.compose.project=$project" --filter label=com.docker.compose.service=watchtower 2>/dev/null) || return 0
    case "$updater" in "" | *[[:space:]]*) return 0 ;; esac
    [ "$(docker inspect --format '{{.Config.Image}}' "$updater" 2>/dev/null)" != "$UPDATER_IMAGE" ] || return 0
    name=$(docker inspect --format '{{.Name}}' "$updater" 2>/dev/null) || return 0
    name=${name#/}
    if ! docker pull --quiet "$UPDATER_IMAGE" >/dev/null 2>&1; then
        echo "Updater kept: $UPDATER_IMAGE cannot be pulled; retrying at the next runner start."
        return 0
    fi
    docker rename "$updater" "$name-replaced" >/dev/null 2>&1 || return 0
    docker stop "$updater" >/dev/null 2>&1 || true
    if docker run --detach --name "$name" --restart unless-stopped -v /var/run/docker.sock:/var/run/docker.sock \
        --label "com.docker.compose.project=$project" --label com.docker.compose.service=watchtower \
        "$UPDATER_IMAGE" >/dev/null 2>&1; then
        docker rm --force "$updater" >/dev/null 2>&1 || true
        echo "Updater replaced: $name now runs $UPDATER_IMAGE"
    else
        docker rename "$updater" "$name" >/dev/null 2>&1 || true
        docker start "$updater" >/dev/null 2>&1 || true
        echo "Updater kept: $UPDATER_IMAGE does not start; the old updater runs again."
    fi
}
ensure_updater

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
