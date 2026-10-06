#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Phala Network <dstack@phala.network>
#
# SPDX-License-Identifier: Apache-2.0

# Starts the SGX services through the CVM upgrade guard. The guard starts the
# exact pinned key-provider image and builds only on a host with no CVM disk.
# `run.sh upgrade` rebuilds the key provider; it is refused while any CVM disk
# exists on the host (see ../cvm_upgrade_guard.sh).

set -e

GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cvm_upgrade_guard.sh"

case "${1:-start}" in
start)
    echo "Starting all SGX services through the CVM upgrade guard..."
    "$GUARD" start
    ;;
upgrade)
    echo "Upgrading the key provider through the CVM upgrade guard..."
    shift
    "$GUARD" upgrade "$@"
    [ "${1:-}" = "--dry-run" ] && exit 0
    ;;
*)
    echo "Usage: $0 [start|upgrade [--dry-run]]" >&2
    exit 1
    ;;
esac

echo "=========================="
echo "Services started!"
echo "=========================="
echo "Key provider endpoint: https://localhost:3443"
echo "  - Using shared socket with AESM service"
echo "  - Socket location: /var/run/aesmd/aesm.socket"
echo 
echo "Check logs with:"
echo "  docker compose logs -f aesmd"
echo "  docker compose logs -f gramine-sealing-key-provider"
echo "=========================="
