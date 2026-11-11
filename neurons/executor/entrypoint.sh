#!/bin/sh
set -eu

# The runner keeps the stack's updater current, so a change of updater reaches every host without a manual
# step. A stack started from an older compose file still runs `nickfedor/watchtower` or `containrrr/watchtower`,
# which pull by tag (DAH-3419); the runner is the one container every updater recreates. Only an updater
# container of the runner's own compose project is replaced, never created: a stack without one (the dstack
# TEE) is left alone. If the new updater cannot start, the old one is put back.
UPDATER_IMAGE="daturaai/lium-watchtower:1.2.2"
ensure_updater() {
    # The runner's own container, by the ID in its mount paths (/var/lib/docker/containers/<id>/hostname), not
    # $HOSTNAME (an updater that recreates the runner copies the old hostname) and not a label lookup (that can
    # find another stack's runner on the same host).
    own=$(sed -n 's#.*/containers/\([0-9a-f]\{64\}\)/.*#\1#p' "${RUNNER_MOUNTINFO:-/proc/self/mountinfo}" 2>/dev/null | head -n 1) || return 0
    [ -n "$own" ] || return 0
    project=$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$own" 2>/dev/null) || return 0
    [ -n "$project" ] || return 0
    found=$(docker ps --all --filter "label=com.docker.compose.project=$project" --filter label=com.docker.compose.service=watchtower --format '{{.ID}} {{.Names}}' 2>/dev/null) || return 0
    # shellcheck disable=SC2086
    set -- $found
    # A swap cut short (daemon or host restart) leaves the old updater stopped as <name>-replaced, alone or
    # next to a new one that never started. Finish it: keep the new updater if it runs, else put the old one back.
    if [ $# -eq 2 ] && [ "${2%-replaced}" != "$2" ]; then
        docker rename "$1" "${2%-replaced}" >/dev/null 2>&1 || return 0
        docker start "$1" >/dev/null 2>&1 || true
        echo "Updater swap finished after an interruption: ${2%-replaced} runs again."
        set -- "$1" "${2%-replaced}"
    elif [ $# -eq 4 ]; then
        case "$2:$4" in
            # only the pair a swap leaves: <name>-replaced and <name>; any other container is not ours to touch
            *-replaced:*) [ "$4" = "${2%-replaced}" ] || return 0; old=$1 old_name=$2 new=$3 ;;
            *:*-replaced) [ "$2" = "${4%-replaced}" ] || return 0; old=$3 old_name=$4 new=$1 ;;
            *) return 0 ;;
        esac
        docker start "$new" >/dev/null 2>&1 || true
        if [ "$(docker inspect --format '{{.State.Running}}' "$new" 2>/dev/null)" = true ]; then
            docker rm --force "$old" >/dev/null 2>&1 || true
            set -- "$new" "${old_name%-replaced}"
        else
            docker rm --force "$new" >/dev/null 2>&1 || return 0
            docker rename "$old" "${old_name%-replaced}" >/dev/null 2>&1 || return 0
            docker start "$old" >/dev/null 2>&1 || true
            set -- "$old" "${old_name%-replaced}"
        fi
        echo "Updater swap finished after an interruption: ${old_name%-replaced} runs again."
    fi
    [ $# -eq 2 ] || return 0
    updater=$1
    current=$(docker inspect --format '{{.Config.Image}}' "$updater" 2>/dev/null) || return 0
    # A dev stack runs the -staging build of the same updater (docker-compose.dev.yml). The dev runner is
    # built with UPDATER_STAGING=1 (docker_runner_build.sh, TAG=dev), whatever updater its stack runs now.
    case "${UPDATER_STAGING:-}:$current" in
        1:* | *-staging) want="$UPDATER_IMAGE-staging" ;;
        *) want="$UPDATER_IMAGE" ;;
    esac
    [ "$current" != "$want" ] || return 0
    name=$(docker inspect --format '{{.Name}}' "$updater" 2>/dev/null) || return 0
    name=${name#/}
    # The compose labels go to the new container, so `docker compose` still finds it as the service's container.
    labels=$(docker inspect --format '{{range $k, $v := .Config.Labels}}{{$k}}={{$v}}{{"\n"}}{{end}}' "$updater" 2>/dev/null) || return 0
    set --
    while IFS= read -r label; do
        case "$label" in com.docker.compose.*) set -- "$@" --label "$label" ;; esac
    done <<LABELS
$labels
LABELS
    if ! docker pull --quiet "$want" >/dev/null 2>&1; then
        echo "Updater kept: $want cannot be pulled; retrying at the next runner start."
        return 0
    fi
    # An image built for another architecture (an amd64-only updater on an arm64 host) would never start.
    if [ "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$want" 2>/dev/null)" != \
        "$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null)" ]; then
        echo "Updater kept: $want is not built for this host's platform; retrying at the next runner start."
        return 0
    fi
    docker rename "$updater" "$name-replaced" >/dev/null 2>&1 || return 0
    docker stop "$updater" >/dev/null 2>&1 || true
    if docker run --detach --name "$name" --restart unless-stopped -v /var/run/docker.sock:/var/run/docker.sock \
        "$@" "$want" >/dev/null 2>&1; then
        docker rm --force "$updater" >/dev/null 2>&1 || true
        echo "Updater replaced: $name now runs $want"
    else
        # a container that was created but did not start holds the name; remove it first
        docker rm --force "$name" >/dev/null 2>&1 || true
        docker rename "$updater" "$name" >/dev/null 2>&1 || true
        docker start "$updater" >/dev/null 2>&1 || true
        echo "Updater kept: $want does not start; the old updater runs again."
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
