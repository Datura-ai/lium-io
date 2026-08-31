"""DAH-2475: failure texts have two audiences, carried in two protocol fields.

`msg` is the renter-safe HEADLINE — the backend surfaces it in customer-facing events, so it must
never carry the executor's host details. `detail` is the full structured text (headline + the extra
dict with the actual exception, executor IP/SSH, failure step) — ops-only: the backend stores it in
filler_run.failure_reason and logs. Packing the full text into msg is what leaked miner host details
into renter dashboards.
"""

from unittest.mock import MagicMock, Mock

from core.utils import _m
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    FailedContainerErrorCodes,
)
from services.miner_service import MinerService


def _miner_service() -> MinerService:
    return MinerService(
        ssh_service=Mock(),
        task_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        backend_client=MagicMock(),
        file_encrypt_service=MagicMock(),
    )


def _create_payload() -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id="00000000-0000-0000-0000-000000000000",
        pod_id="pod-1",
        user_public_keys=[],
        docker_image="alpine:3.20",
        docker_image_tag="3.20",
        gpu_uuids=[],
    )


def test_the_headline_goes_to_msg_and_the_full_text_to_detail() -> None:
    log_text = _m(
        "Resulted in an exception",
        extra={"executor_ip_address": "149.36.1.151", "executor_ssh_port": 2200, "error": "boom"},
    )

    failed = _miner_service()._handle_container_error(
        _create_payload(), log_text, FailedContainerErrorCodes.UnknownError
    )

    assert failed.msg == "Resulted in an exception"
    assert "149.36.1.151" not in failed.msg, "host details reached the renter-facing headline"
    assert "149.36.1.151" in failed.detail
    assert "boom" in failed.detail


def test_a_plain_string_failure_has_no_detail() -> None:
    delete_payload = ContainerDeleteRequest(
        miner_hotkey="miner",
        executor_id="00000000-0000-0000-0000-000000000000",
        pod_id="pod-1",
        container_name="pod_1",
        volume_name="volume_1",
    )

    failed = _miner_service()._handle_container_error(
        delete_payload, "Failed to submit SSH key", FailedContainerErrorCodes.UnknownError
    )

    assert failed.msg == "Failed to submit SSH key"
    assert failed.detail is None
