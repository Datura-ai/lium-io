"""DAH-3199: every rental container runs on the ICC-off bridge `lium-rentals`, not on docker0.

The daemon's default bridge lets every container on it reach every other one, and a pod holds
NET_ADMIN, so two rentals on a split host could otherwise open connections to each other's
unpublished ports. The validator builds the run spec, so the property is pinned here:
`_build_rental_container_run_spec` names the network for an ordinary rental, a sysbox rental and a
cluster node alike; the CVM quote broker, which talks over unix sockets only, stays where it was.
"""

from unittest.mock import Mock

import pytest
from payload_models.payloads import ClusterMembership, ContainerCreateRequest, CustomOptions
from services.cluster_fabric import WIREGUARD_LISTEN_PORT
from services.cvm_quote_broker import quote_broker_run_spec
from services.docker_service import DockerService
from services.rental_docker_observability import rental_run_spec_log_fields
from services.rental_docker_sdk import RENTAL_NETWORK_NAME, GpuDockerConfig


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _payload(*, is_sysbox: bool = False, cluster_membership: ClusterMembership | None = None) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        is_sysbox=is_sysbox,
        cluster_membership=cluster_membership,
    )


def _run_spec(docker_service: DockerService, payload: ContainerCreateRequest):
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_test",
        custom_options=CustomOptions(),
        # (docker_port, internal_port, external_port) as the backend sends them; the host publishes internal_port
        port_maps=[(22, 30022, 40022)],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )


@pytest.mark.parametrize("is_sysbox", [False, True], ids=["runc", "sysbox"])
def test_a_rental_joins_the_isolated_network(docker_service, is_sysbox) -> None:
    run_spec = _run_spec(docker_service, _payload(is_sysbox=is_sysbox))

    assert run_spec.network == RENTAL_NETWORK_NAME
    # what the pod needs from docker0 it keeps on the new bridge: its published ports and NET_ADMIN
    assert [(port.container_port, port.host_port) for port in run_spec.ports] == [(22, 30022)]
    assert "NET_ADMIN" in run_spec.cap_add


def test_a_cluster_node_joins_the_isolated_network_and_keeps_its_wireguard_port(docker_service) -> None:
    membership = ClusterMembership(node_index=0, wireguard_conf="[Interface]\nPrivateKey = x\n")

    run_spec = _run_spec(docker_service, _payload(cluster_membership=membership))

    # the overlay peers reach this node through the host-published UDP port, not through docker0
    assert run_spec.network == RENTAL_NETWORK_NAME
    assert any(port.protocol == "udp" and port.host_port == WIREGUARD_LISTEN_PORT for port in run_spec.ports)


def test_the_network_is_in_the_run_log_fields(docker_service) -> None:
    fields = rental_run_spec_log_fields(_run_spec(docker_service, _payload()))

    assert fields["network"] == RENTAL_NETWORK_NAME


def test_the_quote_broker_stays_on_the_default_bridge() -> None:
    # it serves the pod over a unix socket bind mount and never opens a TCP port to another container
    assert quote_broker_run_spec().network is None
