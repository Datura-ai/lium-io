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
            *-replaced:*) old=$1 old_name=$2 new=$3 ;;
            *:*-replaced) old=$3 old_name=$4 new=$1 ;;
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
