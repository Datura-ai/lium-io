"""DAH-2620: give a node of a multi-node group rental what it needs to join the cluster overlay.

The InfiniBand data path already works from a plain container (DAH-2571), but NCCL never reaches it.
NCCL's bootstrap is a TCP ring built independently of torchrun's rendezvous: every rank binds a
listen socket to the address of the interface `NCCL_SOCKET_IFNAME` selects, and those addresses are
allgathered so each rank can dial its ring neighbour. On the default docker bridge every container
believes it is 172.17.0.2, so a rank dials itself and the IB transport never starts. Publishing host
ports does not help — NCCL advertises the container's own address, and peers dial it directly.

The answer is one routable address per rank, carried by a private WireGuard overlay. The backend
mints the whole mesh at group-rental time, because every node's peer list has to agree and only the
backend knows the group; the validator's job is just to hand one node the config it was given and
open the port WireGuard needs.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

# WireGuard listens here inside every pod. UDP is not a choice: WireGuard has no TCP mode.
#
# DAH-2842: this is the container side only. The host side is the port the backend allocated for this
# node from the executor's verified ports, because a provider forwards only those, and several
# executors can sit behind one public address with a range each. Publishing this port on the host as
# well, as the fleet did before, gave every such node an endpoint nothing forwards.
WIREGUARD_LISTEN_PORT = 51820


@dataclass(frozen=True, slots=True)
class ClusterPodNetworking:
    """What a cluster node needs on top of an ordinary pod."""

    environment: dict[str, str]
    # The host port that carries this node's overlay traffic to WIREGUARD_LISTEN_PORT in the pod.
    overlay_host_port: int


def cluster_pod_networking(
    wireguard_conf: str,
    ssh_private_key: str,
    ssh_authorized_key: str,
    overlay_udp_port: int,
) -> ClusterPodNetworking:
    """The config in env for the template's entrypoint, and the host port to publish for it.

    Base64 because the config and the private key are multi-line and travel as environment
    variables. DAH-2664: the SSH pair is the group's shared login, so `mpirun` and pdsh can start
    ranks on the peers; an older backend sends neither and the entrypoint then installs nothing.

    DAH-2842: `overlay_udp_port` is the host port this node's peers dial, allocated by the backend
    from the executor's own verified ports. An older backend sends nothing and the payload default
    keeps the fleet's previous behaviour.
    """
    environment: dict[str, str] = {
        "LIUM_WIREGUARD_CONF_B64": base64.b64encode(wireguard_conf.encode()).decode()
    }
    if ssh_private_key and ssh_authorized_key:
        environment["LIUM_CLUSTER_SSH_KEY_B64"] = base64.b64encode(ssh_private_key.encode()).decode()
        environment["LIUM_CLUSTER_SSH_PUBKEY"] = ssh_authorized_key
    return ClusterPodNetworking(
        environment=environment,
        overlay_host_port=overlay_udp_port,
    )
