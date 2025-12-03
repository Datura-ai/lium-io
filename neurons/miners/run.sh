#!/bin/sh

# db migrate
pdm run alembic upgrade head

# migrate validator hotkey
pdm run src/cli.py migrate-validator-hotkey

# run fastapi app
pdm run src/miner.py