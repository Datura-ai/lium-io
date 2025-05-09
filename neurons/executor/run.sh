#!/bin/sh
set -eux -o pipefail

# start ssh service
ssh-keygen -A
service ssh start

# run fastapi app
python src/executor.py