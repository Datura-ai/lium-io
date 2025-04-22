#!/bin/sh

# db migrate
pdm run alembic upgrade head

# run fastapi app
pdm run src/miner.py