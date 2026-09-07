#!/usr/bin/env bash
# docker compose for this stack with stack.env applied (compose only auto-loads a file named .env, which the repo
# ignores). Same flags the Makefile uses: ./compose.sh ps | logs miner | --profile tools run --rm tester pytest …
cd "$(dirname "$0")" && set -a && . ./stack.env && set +a
exec docker compose --env-file stack.env -f docker-compose.e2e.yml ${E2E_GPU:+-f docker-compose.gpu.yml} "$@"
