"""The dstack guest-agent socket reaches a customer rental on a CVM node — and nothing else.

A TEE workload's attestation tooling dials ``/var/run/dstack.sock`` (the dstack SDK default) from
inside the pod to take its own TDX quote. On a dstack CVM guest the executor container already has
that socket bind-mounted from the guest; a customer rental on the same guest gets the identical
``source == target`` bind. Bare-metal nodes have no such socket (dockerd would create an empty
directory at the source path), and fillers have no attestation use for it — neither gets the mount.
"""

from unittest.mock import Mock

import pytest
from payload_models.payloads import ContainerCreateRequest, CustomOptions, WorkloadKind

from services.docker_service import (
    DSTACK_GUEST_SOCKET_PATH,
    DockerService,
    _build_dstack_socket_mounts,
)
from services.rental_docker_sdk import GpuDockerConfig, _binds


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _make_payload(
    workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL,
) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        workload_kind=workload_kind,
        active_container_names=[],
    )


def _build_run_spec(
    docker_service: DockerService, payload: ContainerCreateRequest, *, in_cvm: bool
):
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_x",
        custom_options=CustomOptions(),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
        in_cvm=in_cvm,
    )


def test_socket_path_is_the_dstack_sdk_default():
    # The whole point is that unmodified renter tooling finds the socket where the dstack SDK looks.
    assert DSTACK_GUEST_SOCKET_PATH == "/var/run/dstack.sock"


def test_customer_rental_in_cvm_gets_the_socket_bind():
    mounts = _build_dstack_socket_mounts(_make_payload(), in_cvm=True)

    assert [(m.source, m.target, m.read_only) for m in mounts] == [
        (DSTACK_GUEST_SOCKET_PATH, DSTACK_GUEST_SOCKET_PATH, False)
    ]


def test_customer_rental_outside_cvm_gets_nothing():
    # No dstack guest, no socket: a bind of a missing source makes dockerd create an empty directory.
    assert _build_dstack_socket_mounts(_make_payload(), in_cvm=False) == []


def test_filler_in_cvm_gets_nothing():
    assert _build_dstack_socket_mounts(_make_payload(WorkloadKind.FILLER), in_cvm=True) == []


def test_run_spec_appends_socket_after_the_rental_volume(docker_service):
    run_spec = _build_run_spec(docker_service, _make_payload(), in_cvm=True)

    assert [(m.source, m.target) for m in run_spec.volumes] == [
        ("volume_pod", "/root"),
        (DSTACK_GUEST_SOCKET_PATH, DSTACK_GUEST_SOCKET_PATH),
    ]
    # What actually reaches dockerd — a plain rw host bind, exactly the compose recipe
    # `- /var/run/dstack.sock:/var/run/dstack.sock` dstack apps use.
    assert _binds(run_spec.volumes)[-1] == "/var/run/dstack.sock:/var/run/dstack.sock:rw"


def test_run_spec_default_is_no_socket(docker_service):
    # in_cvm defaults to False so every other caller of the builder is unchanged.
    run_spec = docker_service._build_rental_container_run_spec(
        payload=_make_payload(),
        container_name="pod_x",
        custom_options=CustomOptions(),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )

    assert [(m.source, m.target) for m in run_spec.volumes] == [("volume_pod", "/root")]


@pytest.mark.parametrize(
    ("workload_kind", "in_cvm"),
    [
        (WorkloadKind.CUSTOMER_RENTAL, False),
        (WorkloadKind.FILLER, True),
        (WorkloadKind.FILLER, False),
    ],
)
def test_run_spec_omits_socket_for_every_other_combination(docker_service, workload_kind, in_cvm):
    run_spec = _build_run_spec(docker_service, _make_payload(workload_kind), in_cvm=in_cvm)

    assert DSTACK_GUEST_SOCKET_PATH not in {m.target for m in run_spec.volumes}
