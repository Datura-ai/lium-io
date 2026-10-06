#!/usr/bin/env bash
# docker compose for this stack with stack.env applied (compose only auto-loads a file named .env, which the repo
# ignores). Same flags the Makefile uses: ./compose.sh ps | logs miner | --profile tools run --rm tester pytest …
# E2E_GPU=1 in the caller's environment adds the GPU override and points E2E_EXECUTOR_IP at the host-network
# executor on the compose gateway, as the Makefile's GPU branch does (a shell variable outranks --env-file).
cd "$(dirname "$0")" || exit 1
[ -n "${E2E_GPU:-}" ] && export E2E_EXECUTOR_IP=172.30.0.1
exec docker compose --env-file stack.env -f docker-compose.e2e.yml ${E2E_GPU:+-f docker-compose.gpu.yml} "$@"
