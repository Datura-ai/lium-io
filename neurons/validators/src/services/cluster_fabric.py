"""DAH-2620: build the WireGuard overlay that lets a rented multi-node group act as one cluster.

The InfiniBand data path already works from a plain container (DAH-2571), but NCCL never reaches it:
its bootstrap advertises the container's private bridge address (172.17.0.2), the peer dials
something unreachable, and the IB transport never starts. Every node in a group therefore needs one
routable address the others can reach — for the low-bandwidth socket bootstrap only; the tensors
still travel over IB verbs.

A WireGuard overlay gives exactly that and nothing more: an address on a private `wg0` mesh that
touches neither the executor host's network stack nor the miner's own addressing. The validator owns
the whole group, so it mints every keypair and writes each node's peer list here; the pod's template
only brings `wg0` up from what it was handed and points NCCL at it. The customer configures nothing.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

# The overlay subnet handed to a cluster. Private, fixed, and never the miner's — the mesh is ours,
# so nothing on the host or the fabric can collide with it. /24 caps a cluster at 254 nodes, far
# above anything rentable today.
CLUSTER_OVERLAY_CIDR = "10.42.0.0/24"

# The UDP port WireGuard listens on inside every pod. Published 1:1 to the host (port mappings are
# 1:1 across the fleet), so a peer reaches it at the host's public address on this same port.
WIREGUARD_LISTEN_PORT = 51820


@dataclass(frozen=True, slots=True)
class ClusterNode:
    """One machine in a rented group, as the validator already knows it."""

    node_index: int
    public_endpoint_host: str


@dataclass(frozen=True, slots=True)
class WireguardPeer:
    """Another node this pod must be able to reach on the overlay."""

    public_key: str
    overlay_ip: str
    endpoint: str


@dataclass(frozen=True, slots=True)
class WireguardNodeConfig:
    """Everything one pod needs to raise `wg0` and join the mesh — injected into it at start."""

    node_index: int
    overlay_ip: str
    private_key: str
    listen_port: int
    peers: tuple[WireguardPeer, ...]

    def to_wg_quick_conf(self) -> str:
        """Render the `wg-quick` config file the pod's entrypoint writes to /etc/wireguard/wg0.conf."""
        lines = [
            "[Interface]",
            f"Address = {self.overlay_ip}/24",
            f"ListenPort = {self.listen_port}",
            f"PrivateKey = {self.private_key}",
        ]
        for peer in self.peers:
            lines += [
                "",
                "[Peer]",
                f"PublicKey = {peer.public_key}",
                f"AllowedIPs = {peer.overlay_ip}/32",
                f"Endpoint = {peer.endpoint}",
                "PersistentKeepalive = 25",
            ]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class _NodeKeypair:
    private_key: str
    public_key: str


def _generate_keypair() -> _NodeKeypair:
    private = X25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return _NodeKeypair(
        private_key=base64.b64encode(private_bytes).decode(),
        public_key=base64.b64encode(public_bytes).decode(),
    )


def _overlay_ip_for(node_index: int) -> str:
    # .1 is the network's first host; node 0 -> 10.42.0.1, node 1 -> 10.42.0.2, and so on.
    network_prefix = CLUSTER_OVERLAY_CIDR.rsplit(".", 1)[0]
    return f"{network_prefix}.{node_index + 1}"


def build_cluster_wireguard_configs(
    nodes: tuple[ClusterNode, ...],
) -> tuple[WireguardNodeConfig, ...]:
    """One full-mesh WireGuard config per node: every node peers with every other node.

    A training cluster is all-to-all — allreduce touches every rank — so there is no hub to route
    through; each node lists all the others as peers. Keys are minted here and never leave the
    validator except as the per-node config injected into that node's own pod.
    """
    if len(nodes) < 2:
        raise ValueError(f"a cluster needs at least two nodes, got {len(nodes)}")
    if len({node.node_index for node in nodes}) != len(nodes):
        raise ValueError("cluster node indexes must be unique")

    keypairs = {node.node_index: _generate_keypair() for node in nodes}
    nodes_by_index = {node.node_index: node for node in nodes}

    configs: list[WireguardNodeConfig] = []
    for node in nodes:
        peers = tuple(
            WireguardPeer(
                public_key=keypairs[peer.node_index].public_key,
                overlay_ip=_overlay_ip_for(peer.node_index),
                endpoint=f"{nodes_by_index[peer.node_index].public_endpoint_host}:{WIREGUARD_LISTEN_PORT}",
            )
            for peer in nodes
            if peer.node_index != node.node_index
        )
        configs.append(
            WireguardNodeConfig(
                node_index=node.node_index,
                overlay_ip=_overlay_ip_for(node.node_index),
                private_key=keypairs[node.node_index].private_key,
                listen_port=WIREGUARD_LISTEN_PORT,
                peers=peers,
            )
        )
    return tuple(configs)


def cluster_env_and_ports(
    wireguard_conf: str,
) -> tuple[dict[str, str], "tuple[int, ...]"]:
    """What a cluster pod needs on top of an ordinary one: the config in env, and the UDP port open.

    Returns the extra environment (the base64 config the template's entrypoint reads) and the UDP
    ports to publish (just the WireGuard port). Kept tiny and pure so `docker_service` stays a
    table of contents.
    """
    env = {"LIUM_WIREGUARD_CONF_B64": base64.b64encode(wireguard_conf.encode()).decode()}
    return env, (WIREGUARD_LISTEN_PORT,)
