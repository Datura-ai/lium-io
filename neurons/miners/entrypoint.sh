#!/bin/sh
set -eu

# The runner keeps the stack's updater current, so a change of updater reaches every host without a manual
# step. A stack started from an older compose file still runs `containrrr/watchtower`, which stops on Docker
# Engine 29 and later; the runner is the one container every updater recreates. Only an updater container of
# the runner's own compose project is replaced, never created: a stack without one (a source install) is left
# alone. If the new updater cannot start, the old one is put back.
# Pinned by digest: the updater holds the host's Docker socket, so a re-pushed tag must not reach it.
UPDATER_IMAGE="nickfedor/watchtower:1.24.0@sha256:e96976bcbeaf0a9adeddc9016a539c27a2245ef6bb7d3ca97ed106783350d387"
ensure_updater() {
    # The runner is found by its compose service label, not $HOSTNAME: an updater that recreates the runner
    # copies the old container's hostname, which then names no container.
    project=$(docker ps --filter label=com.docker.compose.service=miner-runner --format '{{.Label "com.docker.compose.project"}}' 2>/dev/null) || return 0
    case "$project" in "" | *[[:space:]]*) return 0 ;; esac
    updater=$(docker ps --all --quiet --filter "label=com.docker.compose.project=$project" --filter label=com.docker.compose.service=watchtower 2>/dev/null) || return 0
    case "$updater" in "" | *[[:space:]]*) return 0 ;; esac
    current=$(docker inspect --format '{{.Config.Image}}' "$updater" 2>/dev/null) || return 0
    want="$UPDATER_IMAGE"
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
    docker rename "$updater" "$name-replaced" >/dev/null 2>&1 || return 0
    docker stop "$updater" >/dev/null 2>&1 || true
    if docker run --detach --name "$name" --restart unless-stopped -v /var/run/docker.sock:/var/run/docker.sock \
        "$@" "$want" --interval 60 --cleanup --label-enable >/dev/null 2>&1; then
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

docker compose up --pull always --detach --wait --force-recreate

# Clean docker images
docker image prune -f

while true
do
    docker compose logs -f
    echo 'All containers died'
    sleep 10
done
