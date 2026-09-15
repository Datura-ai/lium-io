"""DAH-3475: a filler ends its own run; a customer pod keeps coming back.

The Dolphin image exits 0 once every worker has failed its spawn cap (computenet-docker-images#71).
Under `restart: unless-stopped` dockerd restarted that container at once with fresh counters, so the
validator never saw a stopped container and the backend's missing-container close never fired.
A filler now gets `on-failure:5`; a customer rental keeps `unless-stopped` (DAH-2306 reboot recovery
relies on it).
"""

from unittest.mock import Mock

import pytest
from payload_models.payloads import ContainerCreateRequest, CustomOptions, WorkloadKind
from services.docker_service import (
    CUSTOMER_RENTAL_RESTART_POLICY,
    FILLER_RESTART_POLICY,
    DockerService,
    _restart_policy_for,
)
from services.rental_docker_sdk import GpuDockerConfig, _restart_policy


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _run_spec(docker_service: DockerService, workload_kind: WorkloadKind):
    payload = ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        workload_kind=workload_kind,
    )
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


def test_filler_run_spec_restarts_on_failure_only(docker_service):
    run_spec = _run_spec(docker_service, WorkloadKind.FILLER)

    assert run_spec.restart_policy == FILLER_RESTART_POLICY == "on-failure:5"


def test_customer_rental_run_spec_keeps_unless_stopped(docker_service):
    run_spec = _run_spec(docker_service, WorkloadKind.CUSTOMER_RENTAL)

    assert run_spec.restart_policy == CUSTOMER_RENTAL_RESTART_POLICY == "unless-stopped"


def test_default_workload_kind_is_a_customer_rental():
    # A backend that does not send workload_kind is renting for a customer: the field defaults to
    # CUSTOMER_RENTAL, so an old backend can never turn a pod into a self-ending filler.
    payload = ContainerCreateRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod", docker_image="img:tag", gpu_uuids=["g0"]
    )

    assert _restart_policy_for(payload.workload_kind) == "unless-stopped"


def test_sdk_maps_retry_cap_to_docker_host_config():
    assert _restart_policy("on-failure:5") == {"Name": "on-failure", "MaximumRetryCount": 5}


def test_sdk_keeps_plain_policy_names_unchanged():
    assert _restart_policy("unless-stopped") == {"Name": "unless-stopped"}
    assert _restart_policy(None) is None


def test_sdk_passes_a_zero_cap_through_as_docker_no_limit():
    # Docker reads MaximumRetryCount 0 as "no limit"; the mapping must not turn it into a rejection
    # or a missing key, or a caller asking for unlimited retries would get a policy error.
    assert _restart_policy("on-failure:0") == {"Name": "on-failure", "MaximumRetryCount": 0}


@pytest.mark.parametrize("policy", ["unless-stopped:3", "always:1"])
def test_sdk_refuses_a_retry_cap_on_a_policy_that_has_none(policy: str):
    # Same rule as the docker CLI: "maximum retry count cannot be used with restart policy".
    with pytest.raises(ValueError, match="only valid with on-failure"):
        _restart_policy(policy)


@pytest.mark.parametrize("policy", ["on-failure:", "on-failure:five", "on-failure:-1"])
def test_sdk_refuses_a_retry_cap_that_is_not_a_count(policy: str):
    with pytest.raises(ValueError, match="non-negative integer"):
        _restart_policy(policy)
