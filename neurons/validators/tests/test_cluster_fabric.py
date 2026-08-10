"""DAH-2620: the validator mints a full-mesh WireGuard config per node in a rented cluster."""
from __future__ import annotations

import base64

import pytest

from services.cluster_fabric import (
    WIREGUARD_LISTEN_PORT,
    ClusterNode,
    build_cluster_wireguard_configs,
)


def _two_node_group() -> tuple[ClusterNode, ...]:
    return (
        ClusterNode(node_index=0, public_endpoint_host="69.63.236.160"),
        ClusterNode(node_index=1, public_endpoint_host="69.63.236.161"),
    )


def test_every_node_peers_with_every_other_node() -> None:
    configs = build_cluster_wireguard_configs(
        (
            ClusterNode(0, "69.63.236.160"),
            ClusterNode(1, "69.63.236.161"),
            ClusterNode(2, "69.63.236.166"),
        )
    )

    assert len(configs) == 3
    for config in configs:
        # all-to-all: a rank talks to every other rank, so no node routes through a hub.
        assert len(config.peers) == 2
        assert config.node_index not in {p.overlay_ip for p in config.peers}


def test_overlay_addresses_are_distinct_and_sequential() -> None:
    configs = build_cluster_wireguard_configs(_two_node_group())

    assert configs[0].overlay_ip == "10.42.0.1"
    assert configs[1].overlay_ip == "10.42.0.2"


def test_a_peer_endpoint_is_the_hosts_public_address_on_the_wireguard_port() -> None:
    configs = build_cluster_wireguard_configs(_two_node_group())

    peer_of_node0 = configs[0].peers[0]
    assert peer_of_node0.endpoint == f"69.63.236.161:{WIREGUARD_LISTEN_PORT}"
    assert peer_of_node0.overlay_ip == "10.42.0.2"


def test_each_node_advertises_the_public_key_matching_the_private_key_it_holds() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    configs = build_cluster_wireguard_configs(_two_node_group())

    # The public key node 1 lists for node 0 must derive from the private key node 0 was handed.
    node0_private = X25519PrivateKey.from_private_bytes(base64.b64decode(configs[0].private_key))
    node0_public = base64.b64encode(
        node0_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    ).decode()
    peer_view_of_node0 = next(p for p in configs[1].peers if p.overlay_ip == "10.42.0.1")
    assert peer_view_of_node0.public_key == node0_public


def test_private_keys_are_unique_per_node() -> None:
    configs = build_cluster_wireguard_configs(_two_node_group())

    assert configs[0].private_key != configs[1].private_key


def test_the_rendered_config_is_a_valid_wg_quick_file() -> None:
    config = build_cluster_wireguard_configs(_two_node_group())[0]

    text = config.to_wg_quick_conf()

    assert "[Interface]" in text
    assert "Address = 10.42.0.1/24" in text
    assert f"ListenPort = {WIREGUARD_LISTEN_PORT}" in text
    assert text.count("[Peer]") == 1
    assert "Endpoint = 69.63.236.161:51820" in text
    assert "AllowedIPs = 10.42.0.2/32" in text
    # keepalive so a NAT pinhole stays open even when the bootstrap traffic is bursty.
    assert "PersistentKeepalive = 25" in text


def test_a_single_node_is_not_a_cluster() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        build_cluster_wireguard_configs((ClusterNode(0, "69.63.236.160"),))


def test_duplicate_node_indexes_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_cluster_wireguard_configs(
            (ClusterNode(0, "69.63.236.160"), ClusterNode(0, "69.63.236.161"))
        )


# --- validator-side injection: an ordinary pod never sees any of this ---


def test_cluster_pod_gets_the_config_in_env_and_the_udp_port() -> None:
    from services.cluster_fabric import cluster_env_and_ports

    conf = build_cluster_wireguard_configs(_two_node_group())[0].to_wg_quick_conf()
    env, udp_ports = cluster_env_and_ports(conf)

    import base64

    assert base64.b64decode(env["LIUM_WIREGUARD_CONF_B64"]).decode() == conf
    assert udp_ports == (WIREGUARD_LISTEN_PORT,)
