#!/bin/sh
set -eu

# The runner keeps the stack's updater current, so a change of updater reaches every host without a manual
# step. A stack started from an older compose file still runs `containrrr/watchtower`, which stops on Docker
# Engine 29 and later; the runner is the one container every updater recreates. Only an updater container of
# the runner's own compose project is replaced, never created: a stack without one (a source install) is left
# alone. If the new updater cannot start, the old one is put back.
UPDATER_IMAGE="nickfedor/watchtower"
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
        "$UPDATER_IMAGE" --interval 60 --cleanup --label-enable >/dev/null 2>&1; then
        docker rm --force "$updater" >/dev/null 2>&1 || true
        echo "Updater replaced: $name now runs $UPDATER_IMAGE"
    else
        docker rename "$updater" "$name" >/dev/null 2>&1 || true
        docker start "$updater" >/dev/null 2>&1 || true
        echo "Updater kept: $UPDATER_IMAGE does not start; the old updater runs again."
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
