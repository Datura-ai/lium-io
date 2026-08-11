#!/usr/bin/env bash
# DAH-2620: raise the cluster overlay, point NCCL at it, then hand off to the workload.
#
# The validator injects LIUM_WIREGUARD_CONF_B64 (this node's wg-quick config, base64) into every pod
# of a group rental. A single-node rental has no such variable, so this whole block is skipped and
# the pod behaves exactly like an ordinary one.
set -euo pipefail

raise_cluster_overlay() {
    local conf_b64="${LIUM_WIREGUARD_CONF_B64:-}"
    if [[ -z "$conf_b64" ]]; then
        echo "lium-cluster: no cluster config injected, running as a standalone node" >&2
        return 0
    fi

    mkdir -p /etc/wireguard
    # Create the file restricted FIRST, then write into it: the config holds this node's private
    # key. Done with `install` rather than `umask` on purpose — a umask set here would survive into
    # `exec "$@"` below and silently make every file the customer's job writes mode 600.
    install -m 600 /dev/null /etc/wireguard/wg0.conf
    echo "$conf_b64" | base64 -d > /etc/wireguard/wg0.conf

    # wg-quick needs NET_ADMIN, which a group-rental pod is given; if it is missing we surface it
    # rather than letting NCCL silently fall back to the unreachable bridge address later.
    if ! wg-quick up wg0; then
        echo "lium-cluster: failed to bring up wg0 — the node cannot join the cluster" >&2
        exit 1
    fi

    # The one contract with the workload: the overlay is always called wg0. NCCL and gloo do not
    # pick a second interface on their own, so we name it for them here and the renter never has to.
    export NCCL_SOCKET_IFNAME=wg0
    export GLOO_SOCKET_IFNAME=wg0
    # Bootstrap rides one flow otherwise; these fan it across the wire (measured 7.3x on our fabric).
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=8

    echo "lium-cluster: wg0 up at $(wg show wg0 2>/dev/null | awk '/interface/{print}')" >&2
}

configure_nested_docker() {
    # The renter's own image runs under the pod's inner daemon, and RDMA there needs devices, the
    # IPC_LOCK capability and an unlimited memlock. Docker can default the ulimit and nothing else,
    # so the rest arrives through a default runtime that edits the OCI spec (lium-rdma-runc).
    mkdir -p /etc/docker
    if [[ -e /etc/docker/daemon.json ]]; then
        echo "lium-cluster: /etc/docker/daemon.json already exists, leaving it alone" >&2
        return 0
    fi
    cat > /etc/docker/daemon.json <<'JSON'
{
  "runtimes": {"lium-rdma": {"path": "/usr/local/bin/lium-rdma-runc"}},
  "default-runtime": "lium-rdma",
  "default-ulimits": {"memlock": {"Name": "memlock", "Hard": -1, "Soft": -1}}
}
JSON
}

raise_cluster_overlay
configure_nested_docker

# Hand off to the base image's own entrypoint, which starts the inner Docker daemon and the rest of
# the pod's services. Replacing it outright is what left a cluster pod without dockerd.
exec /pytorch-entrypoint.sh "$@"
