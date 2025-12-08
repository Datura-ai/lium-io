#!/bin/sh

# db migrate (skip in central mode)
# Convert CENTRAL_MODE to lowercase for case-insensitive comparison
CENTRAL_MODE_LOWER=$(echo "$CENTRAL_MODE" | tr '[:upper:]' '[:lower:]')
if [ "$CENTRAL_MODE_LOWER" != "true" ]; then
  pdm run alembic upgrade head
  # migrate validator hotkey
  pdm run src/cli.py migrate-validator-hotkey
fi

# run fastapi app
pdm run src/miner.py