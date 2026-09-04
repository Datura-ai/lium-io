"""DAH-2748: an availability error means we could not reach something, and hides the node."""

from unittest.mock import MagicMock

import pytest
from services.task.availability import (
    AVAILABILITY_CATEGORY,
    AvailabilityErrorCode,
    Reached,
    Reacher,
    build_availability_event,
    build_ssh_unreachable_event,
    first_availability_error_code,
)
from services.task.models import build_msg


def test_ssh_unreachable_is_an_availability_error() -> None:
    # Arrange / Act
    event = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="Not allowed at this time"
    )

    # Assert
    assert event.category == AVAILABILITY_CATEGORY
    assert event.is_availability_error is True
    assert event.reason_code == AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE
    assert event.what_we_saw["reacher"] == "validator"
    assert event.what_we_saw["reached"] == "executor_ssh"
    assert event.what_we_saw["ssh_port"] == 2200
    assert event.remediation
    assert first_availability_error_code([event]) == "EXECUTOR_SSH_UNREACHABLE"


def test_an_ordinary_check_failure_is_not_an_availability_error() -> None:
    # Arrange: a normal verdict about the machine itself.
    event = build_msg(
        event="GPU count does not match",
        reason="GPU_COUNT_MISMATCH",
        severity="error",
        impact="Score is zero for this cycle.",
    )

    # Act / Assert
    assert event.is_availability_error is False
    assert first_availability_error_code([event]) is None


def test_no_events_mean_no_availability_error() -> None:
    assert first_availability_error_code(None) is None
    assert first_availability_error_code([]) is None


def test_an_availability_error_anywhere_in_the_cycle_counts() -> None:
    # Arrange: the reachability check fails early and an ordinary verdict follows it.
    unreachable = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="refused"
    )
    later_verdict = build_msg(
        event="GPU count does not match",
        reason="GPU_COUNT_MISMATCH",
        severity="error",
        impact="Score is zero for this cycle.",
    )

    # Act / Assert
    assert first_availability_error_code([unreachable, later_verdict]) == "EXECUTOR_SSH_UNREACHABLE"


def test_the_class_names_who_could_not_reach_what() -> None:
    # Arrange: a container that cannot reach an image registry. A new check adds its members
    # to the two enums and the code; nothing downstream changes.
    event = build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reacher=Reacher.CONTAINER,
        reached=Reached.EXECUTOR_SSH,
        event="Container cannot reach the image registry",
        impact="The node is hidden from the market.",
        remediation="Check outbound network on the node.",
        what={"registry": "docker.io"},
    )

    # Act / Assert
    assert event.is_availability_error is True
    assert event.what_we_saw["reacher"] == "container"
    assert event.what_we_saw["reached"] == "executor_ssh"
    assert event.what_we_saw["registry"] == "docker.io"


@pytest.mark.asyncio
async def test_a_cycle_that_cannot_open_ssh_scores_zero_and_reports_the_code() -> None:
    """The except branch in TaskService.create_task must mark the cycle unreachable."""
    import asyncssh
    from datura.requests.miner_requests import ExecutorSSHInfo
    from payload_models.payloads import MinerJobRequestPayload
    from services.task.service import TaskService, _is_ssh_transport_failure

    assert _is_ssh_transport_failure(asyncssh.Error(code=1, reason="refused")) is True
    assert _is_ssh_transport_failure(ValueError("a failed check")) is False

    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(side_effect=asyncssh.Error(code=1, reason="refused"))

    executor = ExecutorSSHInfo(
        uuid="node-9",
        address="10.0.0.5",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1",
        miner_hotkey="5Miner",
        miner_coldkey="5Cold",
        miner_address="10.0.0.5",
        miner_port=8080,
        executors=[],
    )

    result = await service.create_task(
        miner_info=miner,
        executor_info=executor,
        keypair=MagicMock(ss58_address="5Val"),
        private_key="key",
        public_key="pub",
        encrypted_files=MagicMock(),
        rented_data=MagicMock(),
        default_docker_image_digests={},
    )

    assert result.score == 0
    assert result.availability_error_code == "EXECUTOR_SSH_UNREACHABLE"
    assert "EXECUTOR_SSH_UNREACHABLE" in result.log_text
