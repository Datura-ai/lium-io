"""DAH-2395: capture container death diagnostics before cleanup removes the evidence.

When create_container fails after the container was created (e.g. a Dolphin filler
whose entrypoint dies with exit 137), cleanup_failed_container_creation runs
`docker rm -fv` and destroys the only artifacts explaining the death. These tests
pin the new behavior: `docker inspect .State` (OOMKilled/ExitCode) and a bounded
`docker logs` tail are captured and logged BEFORE the container is removed, and a
capture failure never blocks the cleanup itself.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest
from core.docker_utils import (
    _CONTAINER_DEATH_LOG_MAX_CHARS,
    ContainerDeathDiagnostics,
)
from services.docker_service import DockerService

CONTAINER_NAME = "filler_63409a62-e024-414d-99aa-ba6f30be1bfd"
DEFAULT_EXTRA: dict = {"miner_hotkey": "test_miner", "executor_uuid": "test_executor"}

INSPECT_STATE_JSON = (
    '{"Status": "exited", "OOMKilled": true, "ExitCode": 137,'
    ' "Error": "", "StartedAt": "2026-07-13T06:06:42Z", "FinishedAt": "2026-07-13T06:06:57Z"}'
)


class _SSHRunResult:
    def __init__(self, exit_status: int = 0, stdout: str = "", stderr: str = ""):
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr


class _FakeSSHClient:
    """Records every command; per-substring overrides drive each command's result."""

    def __init__(self):
        self.commands: list[str] = []
        self.results_by_substring: dict[str, _SSHRunResult] = {}
        self.errors_by_substring: dict[str, Exception] = {}

    async def run(self, command: str) -> _SSHRunResult:
        self.commands.append(command)
        for substring, error in self.errors_by_substring.items():
            if substring in command:
                raise error
        for substring, result in self.results_by_substring.items():
            if substring in command:
                return result
        return _SSHRunResult()

    def command_index(self, substring: str) -> int:
        for index, command in enumerate(self.commands):
            if substring in command:
                return index
        raise AssertionError(f"no command containing {substring!r} was run: {self.commands}")


@pytest.fixture
def docker_service() -> DockerService:
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _diagnostics_log_extras(mock_logger) -> list[dict]:
    extras: list[dict] = []
    for call in mock_logger.warning.call_args_list:
        structured_message = call.args[0]
        if structured_message.message == "Failed container diagnostics before cleanup":
            extras.append(structured_message.extra)
    return extras


@pytest.mark.asyncio
async def test_cleanup_captures_death_diagnostics_before_removal(docker_service):
    # Arrange
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(stdout=INSPECT_STATE_JSON)
    ssh_client.results_by_substring["docker logs"] = _SSHRunResult(
        stdout="RuntimeError: CUDA out of memory"
    )

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        await docker_service.cleanup_failed_container_creation(
            ssh_client=ssh_client,
            default_extra=DEFAULT_EXTRA,
            container_name=CONTAINER_NAME,
        )

    # Assert: evidence is read strictly before `docker rm -fv` destroys it
    removal_index = ssh_client.command_index("docker rm -fv")
    assert ssh_client.command_index("docker inspect") < removal_index
    assert ssh_client.command_index("docker logs") < removal_index

    extras = _diagnostics_log_extras(mock_logger)
    assert len(extras) == 1
    assert extras[0]["container_name"] == CONTAINER_NAME
    assert extras[0]["container_oom_killed"] is True
    assert extras[0]["container_exit_code"] == 137
    assert "CUDA out of memory" in extras[0]["container_logs_tail"]
    assert extras[0]["diagnostics_capture_error"] is None


@pytest.mark.asyncio
async def test_cleanup_still_removes_container_when_diagnostics_capture_fails(docker_service):
    # Arrange: both diagnostic commands blow up, removal itself works
    ssh_client = _FakeSSHClient()
    ssh_client.errors_by_substring["docker inspect"] = ConnectionError("ssh channel died")
    ssh_client.errors_by_substring["docker logs"] = ConnectionError("ssh channel died")

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        await docker_service.cleanup_failed_container_creation(
            ssh_client=ssh_client,
            default_extra=DEFAULT_EXTRA,
            container_name=CONTAINER_NAME,
        )

    # Assert: cleanup is never blocked by diagnostics
    assert ssh_client.command_index("docker rm -fv") >= 0
    extras = _diagnostics_log_extras(mock_logger)
    assert len(extras) == 1
    assert "ssh channel died" in extras[0]["diagnostics_capture_error"]


@pytest.mark.asyncio
async def test_diagnostics_reports_missing_container_without_failing(docker_service):
    # Arrange: container already gone — inspect exits non-zero
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(
        exit_status=1, stderr="Error: No such container"
    )
    ssh_client.results_by_substring["docker logs"] = _SSHRunResult(
        stdout="Error response from daemon: No such container"
    )

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        diagnostics = await docker_service.capture_failed_container_diagnostics(
            ssh_client=ssh_client,
            default_extra=DEFAULT_EXTRA,
            container_name=CONTAINER_NAME,
        )

    # Assert
    assert isinstance(diagnostics, ContainerDeathDiagnostics)
    assert diagnostics.exit_code is None
    assert diagnostics.oom_killed is None
    assert "No such container" in diagnostics.capture_error
    assert len(_diagnostics_log_extras(mock_logger)) == 1


@pytest.mark.asyncio
async def test_diagnostics_treats_non_object_inspect_json_as_capture_error(docker_service):
    # Arrange: valid JSON that is not an object must not crash the capture
    # (a crash here would propagate into cleanup and skip `docker rm -fv`)
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(stdout='"exited"')
    ssh_client.results_by_substring["docker logs"] = _SSHRunResult(stdout="some logs")

    # Act
    with patch("services.docker_service.logger") as mock_logger:
        diagnostics = await docker_service.capture_failed_container_diagnostics(
            ssh_client=ssh_client,
            default_extra=DEFAULT_EXTRA,
            container_name=CONTAINER_NAME,
        )

    # Assert
    assert diagnostics.status is None
    assert "inspect" in diagnostics.capture_error
    assert len(_diagnostics_log_extras(mock_logger)) == 1


@pytest.mark.asyncio
async def test_diagnostics_logs_tail_is_size_bounded(docker_service):
    # Arrange: an oversized log where the fatal line is at the very end
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(stdout=INSPECT_STATE_JSON)
    oversized_logs = ("x" * (_CONTAINER_DEATH_LOG_MAX_CHARS * 3)) + "FATAL: worker crashed"
    ssh_client.results_by_substring["docker logs"] = _SSHRunResult(stdout=oversized_logs)

    # Act
    with patch("services.docker_service.logger"):
        diagnostics = await docker_service.capture_failed_container_diagnostics(
            ssh_client=ssh_client,
            default_extra=DEFAULT_EXTRA,
            container_name=CONTAINER_NAME,
        )

    # Assert: bounded, and the tail (where the death reason lives) is preserved
    assert len(diagnostics.logs_tail) == _CONTAINER_DEATH_LOG_MAX_CHARS
    assert diagnostics.logs_tail.endswith("FATAL: worker crashed")


@pytest.mark.asyncio
async def test_diagnostics_propagates_cancellation(docker_service):
    # Arrange
    ssh_client = _FakeSSHClient()
    ssh_client.errors_by_substring["docker inspect"] = asyncio.CancelledError()

    # Act / Assert: cancellation must never be swallowed as a capture error
    with pytest.raises(asyncio.CancelledError):
        await docker_service.capture_failed_container_diagnostics(
            ssh_client=ssh_client,
            default_extra=DEFAULT_EXTRA,
            container_name=CONTAINER_NAME,
        )


# DAH-2703: a create failure whose container is already gone from the host is a different
# offence from a container that failed on its own — the backend counts only the former.


@pytest.mark.asyncio
async def test_diagnostics_flag_a_container_that_vanished_from_the_host(docker_service):
    # Arrange: inspect says the object does not exist any more
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(
        exit_status=1, stderr=f"Error: No such object: {CONTAINER_NAME}"
    )

    # Act
    diagnostics = await docker_service.capture_failed_container_diagnostics(
        ssh_client=ssh_client,
        default_extra=DEFAULT_EXTRA,
        container_name=CONTAINER_NAME,
    )

    # Assert
    assert diagnostics.container_missing is True


@pytest.mark.asyncio
async def test_diagnostics_do_not_flag_a_container_that_died_on_its_own(docker_service):
    # Arrange: the container is still there, it just exited
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(stdout=INSPECT_STATE_JSON)

    # Act
    diagnostics = await docker_service.capture_failed_container_diagnostics(
        ssh_client=ssh_client,
        default_extra=DEFAULT_EXTRA,
        container_name=CONTAINER_NAME,
    )

    # Assert
    assert diagnostics.container_missing is False


@pytest.mark.asyncio
async def test_cleanup_reports_whether_the_container_was_already_gone(docker_service):
    # Arrange
    ssh_client = _FakeSSHClient()
    ssh_client.results_by_substring["docker inspect"] = _SSHRunResult(
        exit_status=1, stderr=f"Error: No such object: {CONTAINER_NAME}"
    )

    # Act
    container_missing = await docker_service.cleanup_failed_container_creation(
        ssh_client=ssh_client,
        default_extra=DEFAULT_EXTRA,
        container_name=CONTAINER_NAME,
    )

    # Assert: the cleanup still runs, and it reports what it found
    assert container_missing is True
    assert ssh_client.command_index("docker rm -fv") >= 0


@pytest.mark.asyncio
async def test_cleanup_reports_no_kill_when_diagnostics_cannot_be_captured(docker_service):
    # Arrange: SSH is broken, so "gone" is unproven — fail open, never accuse the host
    ssh_client = _FakeSSHClient()
    ssh_client.errors_by_substring["docker inspect"] = ConnectionError("ssh channel died")

    # Act
    container_missing = await docker_service.cleanup_failed_container_creation(
        ssh_client=ssh_client,
        default_extra=DEFAULT_EXTRA,
        container_name=CONTAINER_NAME,
    )

    # Assert
    assert container_missing is False
