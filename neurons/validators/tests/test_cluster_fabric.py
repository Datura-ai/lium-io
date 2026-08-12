"""DAH-2620: a cluster node is handed its overlay config and the port WireGuard needs."""
from __future__ import annotations

import base64
from unittest.mock import Mock

import pytest

from payload_models.payloads import ClusterMembership, ContainerCreateRequest, CustomOptions
from services.cluster_fabric import WIREGUARD_LISTEN_PORT, cluster_pod_networking
from services.docker_service import DockerService
from services.rental_docker_sdk import GpuDockerConfig

_RENDERED_CONF = """[Interface]
Address = 10.42.0.1/24
ListenPort = 51820
PrivateKey = cHJpdmF0ZS1rZXktZm9yLXRlc3Rpbmc=

[Peer]
PublicKey = cHVibGljLWtleS1mb3ItdGVzdGluZw==
AllowedIPs = 10.42.0.2/32
Endpoint = 69.63.236.161:51820
PersistentKeepalive = 25
"""

_SSH_PRIVATE_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nZmFrZS1rZXk=\n-----END OPENSSH PRIVATE KEY-----\n"
_SSH_AUTHORIZED_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIfake lium-cluster"


def test_the_config_reaches_the_pod_intact_through_the_environment() -> None:
    networking = cluster_pod_networking(_RENDERED_CONF)

    # Base64 because the config is multi-line and travels as an env var; the entrypoint decodes it.
    decoded = base64.b64decode(networking.environment["LIUM_WIREGUARD_CONF_B64"]).decode()
    assert decoded == _RENDERED_CONF


def test_the_wireguard_port_is_published_over_udp() -> None:
    networking = cluster_pod_networking(_RENDERED_CONF)

    # WireGuard has no TCP mode, and the fleet publishes only TCP by default — without this the
    # handshake never happens and the overlay never forms.
    assert networking.published_udp_ports == (WIREGUARD_LISTEN_PORT,)


# --- the run spec: an ordinary rental must not notice any of this ---


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _payload(cluster_membership: ClusterMembership | None) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        cluster_membership=cluster_membership,
    )


def _run_spec(docker_service: DockerService, payload: ContainerCreateRequest):
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_test",
        custom_options=CustomOptions(),
        port_maps=[(2000, 2000, 3000)],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )


def test_an_ordinary_rental_gets_no_overlay_config_and_no_udp_port(docker_service) -> None:
    run_spec = _run_spec(docker_service, _payload(cluster_membership=None))

    assert "LIUM_WIREGUARD_CONF_B64" not in run_spec.environment
    assert [port.protocol for port in run_spec.ports] == ["tcp"]


def test_a_cluster_node_gets_its_config_and_the_udp_port_alongside_the_usual_ones(
    docker_service,
) -> None:
    payload = _payload(ClusterMembership(node_index=0, wireguard_conf=_RENDERED_CONF))

    run_spec = _run_spec(docker_service, payload)

    decoded = base64.b64decode(run_spec.environment["LIUM_WIREGUARD_CONF_B64"]).decode()
    assert decoded == _RENDERED_CONF
    udp_ports = [port for port in run_spec.ports if port.protocol == "udp"]
    assert [(port.container_port, port.host_port) for port in udp_ports] == [
        (WIREGUARD_LISTEN_PORT, WIREGUARD_LISTEN_PORT)
    ]
    # The rental's own published ports are still there — the overlay is added, not substituted.
    assert any(port.protocol == "tcp" and port.container_port == 2000 for port in run_spec.ports)

# --- DAH-2664: the group's shared SSH login ---


def test_the_shared_ssh_login_reaches_the_pod_intact() -> None:
    networking = cluster_pod_networking(_RENDERED_CONF, _SSH_PRIVATE_KEY, _SSH_AUTHORIZED_KEY)

    # The private key is multi-line, so it travels base64'd like the config; the public line is not.
    decoded = base64.b64decode(networking.environment["LIUM_CLUSTER_SSH_KEY_B64"]).decode()
    assert decoded == _SSH_PRIVATE_KEY
    assert networking.environment["LIUM_CLUSTER_SSH_PUBKEY"] == _SSH_AUTHORIZED_KEY


def test_a_backend_that_sends_no_ssh_login_injects_nothing(docker_service) -> None:
    # An older backend has no DAH-2664 fields, and the pod must still come up with its overlay.
    payload = _payload(ClusterMembership(node_index=0, wireguard_conf=_RENDERED_CONF))

    run_spec = _run_spec(docker_service, payload)

    assert "LIUM_WIREGUARD_CONF_B64" in run_spec.environment
    assert "LIUM_CLUSTER_SSH_KEY_B64" not in run_spec.environment
    assert "LIUM_CLUSTER_SSH_PUBKEY" not in run_spec.environment


def test_a_cluster_node_carries_the_shared_ssh_login_into_the_container(docker_service) -> None:
    payload = _payload(
        ClusterMembership(
            node_index=0,
            wireguard_conf=_RENDERED_CONF,
            ssh_private_key=_SSH_PRIVATE_KEY,
            ssh_authorized_key=_SSH_AUTHORIZED_KEY,
        )
    )

    run_spec = _run_spec(docker_service, payload)

    decoded = base64.b64decode(run_spec.environment["LIUM_CLUSTER_SSH_KEY_B64"]).decode()
    assert decoded == _SSH_PRIVATE_KEY
    assert run_spec.environment["LIUM_CLUSTER_SSH_PUBKEY"] == _SSH_AUTHORIZED_KEY
