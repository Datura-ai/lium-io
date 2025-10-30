#!/bin/sh
set -eux -o pipefail

# ensure docker group matches host socket
SOCK=/var/run/docker.sock
if [ -S "$SOCK" ]; then
  SGID=$(stat -c %g "$SOCK")
  GNAME=$(getent group "$SGID" | cut -d: -f1 || true)
  if [ -z "$GNAME" ]; then
    groupadd -o -g "$SGID" dockersock
    GNAME=dockersock
  fi
  usermod -aG "$GNAME" liumuser || true
fi

# start ssh service
ssh-keygen -A
service ssh start

# db migrate
alembic upgrade head

# run fastapi app
python src/executor.py
