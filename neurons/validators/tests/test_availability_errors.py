"""DAH-2748: an availability error means we could not reach something, and hides the node."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from services.attestation_service import HostPolicyResult
from services.task import service as task_service_module
from services.task.service import TaskService, _is_ssh_transport_failure
from services.task.availability import (
    AVAILABILITY_CATEGORY,
    AvailabilityErrorCode,
    ReachSource,
    ReachTarget,
    build_availability_event,
    build_ssh_unreachable_event,
    availability_error_codes,
)
from services.task.models import JobResult, build_msg


def test_ssh_unreachable_is_an_availability_error() -> None:
    # Arrange / Act
    event = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="Not allowed at this time"
    )

    # Assert
    assert event.category == AVAILABILITY_CATEGORY
    assert event.is_availability_error is True
    assert event.reason_code == AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE
    assert event.what_we_saw["reach_source"] == "validator"
    assert event.what_we_saw["reach_target"] == "executor_ssh"
    assert event.what_we_saw["ssh_port"] == 2200
    assert event.remediation
    assert availability_error_codes([event]) == ["EXECUTOR_SSH_UNREACHABLE"]


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
    assert availability_error_codes([event]) == []


def test_no_events_mean_no_availability_error() -> None:
    assert availability_error_codes(None) == []
    assert availability_error_codes([]) == []


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
    assert availability_error_codes([unreachable, later_verdict]) == ["EXECUTOR_SSH_UNREACHABLE"]


def test_the_class_names_who_could_not_reach_what() -> None:
    # Arrange: a container that cannot reach an image registry. A new check adds its members
    # to the two enums and the code; nothing downstream changes.
    event = build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.CONTAINER,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="Container cannot reach the image registry",
        impact="The node is hidden from the market.",
        remediation="Check outbound network on the node.",
        what_we_saw={"registry": "docker.io"},
    )

    # Act / Assert
    assert event.is_availability_error is True
    assert event.what_we_saw["reach_source"] == "container"
    assert event.what_we_saw["reach_target"] == "executor_ssh"
    assert event.what_we_saw["registry"] == "docker.io"


def _task_service_that_reaches_the_ssh_connect() -> TaskService:
    """A TaskService whose only live steps are key decryption and the attestation policy."""
    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="decrypted-key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(return_value=HostPolicyResult())
    return service


def _a_job_for(executor_uuid: str) -> tuple[MinerJobRequestPayload, ExecutorSSHInfo]:
    executor = ExecutorSSHInfo(
        uuid=executor_uuid,
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
    return miner, executor


async def _run_cycle(
    service: TaskService, miner: MinerJobRequestPayload, executor: ExecutorSSHInfo
) -> JobResult:
    return await service.create_task(
        miner_info=miner,
        executor_info=executor,
        keypair=MagicMock(ss58_address="5Val"),
        private_key="key",
        public_key="pub",
        encrypted_files=MagicMock(),
        rented_data=MagicMock(),
        default_docker_image_digests={},
    )


class _ShellThatRefusesToConnect:
    """Stands in for InteractiveShellService: the handshake fails, the body never runs."""

    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> "_ShellThatRefusesToConnect":
        raise asyncssh.Error(code=1, reason="Connection refused")

    async def __aexit__(self, *_: object) -> bool:
        return False


@pytest.mark.asyncio
async def test_a_cycle_that_cannot_open_ssh_scores_zero_and_reports_the_code(monkeypatch) -> None:
    """A refused SSH handshake must score zero and name the node unreachable."""
    assert _is_ssh_transport_failure(asyncssh.Error(code=1, reason="refused")) is True
    assert _is_ssh_transport_failure(ValueError("a failed check")) is False

    monkeypatch.setattr(
        task_service_module, "InteractiveShellService", _ShellThatRefusesToConnect
    )
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(_task_service_that_reaches_the_ssh_connect(), miner, executor)

    assert result.score == 0
    assert result.availability_error_codes == ["EXECUTOR_SSH_UNREACHABLE"]
    assert "EXECUTOR_SSH_UNREACHABLE" in result.log_text


@pytest.mark.asyncio
async def test_a_network_error_before_the_ssh_connect_is_not_blamed_on_the_node() -> None:
    """The attestation verifier's own HTTP call raises OSError types too (aiohttp, timeouts).

    Nothing about the node's sshd was measured there, so the node must not be hidden.
    """
    service = _task_service_that_reaches_the_ssh_connect()
    service.attestation_service.prepare_host_policy = AsyncMock(
        side_effect=ConnectionRefusedError("TDX verifier is down")
    )
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(service, miner, executor)

    assert result.score == 0
    # None, not []: this cycle never reached the connect, so it must not clear what the
    # cycle before it found.
    assert result.availability_error_codes is None
    assert "EXECUTOR_SSH_UNREACHABLE" not in result.log_text


def test_every_failed_reachability_check_is_reported() -> None:
    """Two network checks failing in one cycle must both reach the provider."""
    # Arrange
    ssh = build_ssh_unreachable_event(
        executor_uuid="node-1", host="1.2.3.4", port=2200, error="refused"
    )
    registry = build_availability_event(
        code=AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE,
        reach_source=ReachSource.CONTAINER,
        reach_target=ReachTarget.EXECUTOR_SSH,
        event_text="Container cannot reach the image registry",
        impact="The node is hidden from the market.",
        remediation="Check outbound network on the node.",
        what_we_saw={"registry": "docker.io"},
    )
    registry.reason_code = "CONTAINER_CANNOT_REACH_DOCKER_HUB"

    # Act
    codes = availability_error_codes([ssh, registry, ssh])

    # Assert - both, in the order the checks ran, and no repeat
    assert codes == ["EXECUTOR_SSH_UNREACHABLE", "CONTAINER_CANNOT_REACH_DOCKER_HUB"]


@pytest.mark.asyncio
async def test_a_failure_after_the_shell_opened_reports_the_node_as_reachable() -> None:
    """DAH-2748: the connect proved the node is there, so a later crash must not keep it hidden."""
    import asyncssh

    from datura.requests.miner_requests import ExecutorSSHInfo
    from payload_models.payloads import MinerJobRequestPayload
    from services.task.service import TaskService

    class _Shell:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(
        return_value=MagicMock(
            known_hosts=None, attestation_digest=None, tee_type=None,
            gpu_attestation_passed=None, attestation_passed=False,
        )
    )
    service.pipeline_factory = MagicMock()
    # The shell opens, then the pipeline blows up with a network error of its own.
    service.pipeline_factory.build_context = AsyncMock(side_effect=asyncssh.Error(code=1, reason="later"))

    executor = ExecutorSSHInfo(
        uuid="node-9", address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
        python_path="/usr/bin/python3", root_dir="/root/app",
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1", miner_hotkey="5Miner", miner_coldkey="5Cold",
        miner_address="10.0.0.5", miner_port=8080, executors=[],
    )

    with patch("services.task.service.InteractiveShellService", return_value=_Shell()):
        result = await service.create_task(
            miner_info=miner, executor_info=executor, keypair=MagicMock(ss58_address="5Val"),
            private_key="key", public_key="pub", encrypted_files=MagicMock(),
            rented_data=MagicMock(), default_docker_image_digests={},
        )

    assert result.availability_error_codes == []
