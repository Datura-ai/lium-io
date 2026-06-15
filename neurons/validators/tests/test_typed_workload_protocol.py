import json
from uuid import uuid4

from payload_models.payloads import (
    BaseServerRequest,
    BaseValidatorResponse,
    ContainerCreateRequest,
    ContainerCreated,
    ContainerDeleteRequest,
    WorkloadKind,
)
from services.docker_service import DockerService
from services.miner_service import _bypasses_renting_in_progress


def _base_create_request(**overrides):
    values = {
        "miner_hotkey": "miner",
        "executor_id": str(uuid4()),
        "pod_id": str(uuid4()),
        "docker_image": "daturaai/pytorch:test",
        "user_public_keys": ["ssh-ed25519 test-key"],
        "gpu_uuids": ["GPU-test"],
    }
    values.update(overrides)
    return ContainerCreateRequest(**values)


def test_missing_workload_kind_defaults_to_customer_rental_for_create_request():
    request = _base_create_request()
    payload = json.loads(request.model_dump_json())
    payload.pop("workload_kind")

    parsed = BaseServerRequest.parse(json.dumps(payload))

    assert isinstance(parsed, ContainerCreateRequest)
    assert parsed.workload_kind == WorkloadKind.CUSTOMER_RENTAL


def test_filler_create_request_round_trips_workload_kind_and_pod_id():
    filler_run_id = str(uuid4())
    request = _base_create_request(
        pod_id=filler_run_id,
        workload_kind=WorkloadKind.FILLER,
    )

    parsed = BaseServerRequest.parse(request.model_dump_json())

    assert isinstance(parsed, ContainerCreateRequest)
    assert parsed.workload_kind == WorkloadKind.FILLER
    assert parsed.pod_id == filler_run_id


def test_create_request_defaults_local_volume_enabled():
    request = _base_create_request()

    assert request.local_volume_enabled is True


def test_container_created_response_echoes_filler_workload_kind():
    filler_run_id = str(uuid4())
    response = ContainerCreated(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=filler_run_id,
        workload_kind=WorkloadKind.FILLER,
        container_name=f"filler_{filler_run_id}",
        volume_name=f"volume_{filler_run_id}",
        port_maps=[],
    )

    parsed = BaseValidatorResponse.parse(response.model_dump_json())

    assert isinstance(parsed, ContainerCreated)
    assert parsed.workload_kind == WorkloadKind.FILLER
    assert parsed.pod_id == filler_run_id


def test_validator_runtime_name_is_derived_from_workload_kind():
    pod_id = str(uuid4())
    filler_run_id = str(uuid4())

    customer_request = _base_create_request(pod_id=pod_id)
    filler_request = _base_create_request(
        pod_id=filler_run_id,
        workload_kind=WorkloadKind.FILLER,
    )

    assert DockerService.get_container_name(customer_request) == f"pod_{pod_id}"
    assert DockerService.get_container_name(filler_request) == f"filler_{filler_run_id}"


def test_filler_delete_bypasses_pending_rent_guard():
    filler_run_id = str(uuid4())
    request = ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=filler_run_id,
        workload_kind=WorkloadKind.FILLER,
        container_name=f"filler_{filler_run_id}",
    )

    assert _bypasses_renting_in_progress(request) is True


def test_customer_delete_does_not_bypass_pending_rent_guard():
    pod_id = str(uuid4())
    request = ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=pod_id,
        container_name=f"pod_{pod_id}",
    )

    assert _bypasses_renting_in_progress(request) is False
