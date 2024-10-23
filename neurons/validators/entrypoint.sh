#!/bin/sh
set -eu

docker compose up --pull always --detach --wait --force-recreate

# Clean docker images
docker image prune -f

# Remove all Docker images with a name but no tag
# docker images --filter "dangling=false" --format "{{.Repository}}:{{.Tag}} {{.ID}}" | grep ":<none>" | awk '{print $2}' | xargs -r docker rmi

while true
do
    docker compose logs -f
    echo 'All containers died'
    sleep 10
done
