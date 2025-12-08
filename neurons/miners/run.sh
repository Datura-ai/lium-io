#!/bin/sh

# db migrate (skip in central mode)
if [ "$CENTRAL_MODE" != "true" ]; then
  pdm run alembic upgrade head
  # migrate validator hotkey
  pdm run src/cli.py migrate-validator-hotkey
fi

# run fastapi app
pdm run src/miner.py