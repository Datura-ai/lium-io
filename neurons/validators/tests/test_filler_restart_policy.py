"""DAH-3475: a filler whose image ends its own run gets `restart: on-failure`; everything else keeps
`unless-stopped`.

Under `restart: unless-stopped` dockerd restarts an image's "nothing to serve" exit at once with
fresh counters, so the validator never sees a stopped container and the backend's missing-container
close never fires. The policy is gated on the job spec's `self_ending` flag, not on the FILLER
class: FILLER also covers provider-owned jobs and ENGY, whose images have no exit contract, and no
backend sends the flag until its image and its cap backoff exist. A customer rental never gets it
(DAH-2306 reboot recovery relies on `unless-stopped`).
"""

from unittest.mock import Mock

import pytest
from payload_models.payloads import ContainerCreateRequest, CustomOptions, WorkloadKind
from services.docker_service import (
    DEFAULT_RESTART_POLICY,
    SELF_ENDING_FILLER_RESTART_POLICY,
    DockerService,
    _restart_policy_for,
)
from services.rental_docker_sdk import GpuDockerConfig


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _payload(**fields) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        **fields,
    )


def _run_spec(docker_service: DockerService, payload: ContainerCreateRequest):
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod",
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


def test_self_ending_filler_run_spec_restarts_on_failure_without_a_cap(docker_service):
    # No retry cap: PEARL's PID 1 re-exits its child's code, so 143 can come from inside the
    # container; a capped policy would leave it `exited` with 143, which rental_verification reads
    # as a host stop and withholds incentive for.
    run_spec = _run_spec(docker_service, _payload(workload_kind=WorkloadKind.FILLER, self_ending=True))

    assert run_spec.restart_policy == SELF_ENDING_FILLER_RESTART_POLICY == "on-failure"


def test_filler_without_the_capability_keeps_unless_stopped(docker_service):
    # The FILLER class alone is not enough: provider-owned jobs and ENGY are FILLERs too and their
    # images never exit 0 on purpose, so `on-failure` would leave nothing for them to gain and a
    # SIGTERM-exit-0 image down after a host reboot.
    run_spec = _run_spec(docker_service, _payload(workload_kind=WorkloadKind.FILLER))

    assert run_spec.restart_policy == DEFAULT_RESTART_POLICY == "unless-stopped"


def test_customer_rental_never_gets_the_self_ending_policy(docker_service):
    run_spec = _run_spec(
        docker_service, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, self_ending=True)
    )

    assert run_spec.restart_policy == "unless-stopped"


def test_a_backend_that_sends_no_workload_kind_and_no_self_ending_changes_nothing():
    # An old backend sends no workload_kind and no self_ending: the defaults are CUSTOMER_RENTAL and
    # False, so nothing it launches becomes a self-ending filler.
    payload = _payload()

    assert payload.workload_kind is WorkloadKind.CUSTOMER_RENTAL
    assert payload.self_ending is False
    assert _restart_policy_for(payload) == "unless-stopped"


def test_self_ending_survives_deserialization_of_the_backend_request():
    # Declared on the model, or pydantic drops the unknown key and the flag can never arrive.
    payload = ContainerCreateRequest.model_validate(
        {
            "message_type": "ContainerCreateRequest",
            "miner_hotkey": "hk",
            "executor_id": "ex",
            "pod_id": "pod",
            "docker_image": "img:tag",
            "gpu_uuids": ["g0"],
            "workload_kind": "FILLER",
            "self_ending": True,
        }
    )

    assert payload.self_ending is True
    assert _restart_policy_for(payload) == "on-failure"
