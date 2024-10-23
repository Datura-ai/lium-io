#!/bin/sh
set -eu

docker compose up --pull always --detach --wait --force-recreate

# Clean docker images
docker image prune -f

while true
do
    docker compose logs -f
    echo 'All containers died'
    sleep 10
done
