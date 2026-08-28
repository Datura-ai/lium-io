from datetime import UTC, datetime

import pytest

from neurons.validators.src.services.task.checks.upload_files import UploadFilesCheck
from neurons.validators.src.services.task.messages import UploadFilesMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state


class DummySFTPClient:
    """Mock SFTP client that simulates file upload."""

    def __init__(self, *, should_raise: bool = False, error_message: str = ""):
        self.should_raise = should_raise
        self.error_message = error_message
        self.put_called_with: dict | None = None

    async def put(self, local_path: str, remote_path: str, recurse: bool = False):
        """Mock method that simulates SFTP put operation."""
        self.put_called_with = {
            "local_path": local_path,
            "remote_path": remote_path,
            "recurse": recurse,
        }

        if self.should_raise:
            raise RuntimeError(self.error_message)


class DummyProbeRunner:
    """Answers the interpreter probe UploadFilesCheck runs before it skips the upload."""

    def __init__(self, *, success: bool):
        self.success = success
        self.commands: list[str] = []

    async def run(self, command: str, timeout: int = 300, retryable: bool = False, stdin_text=None):
        self.commands.append(command)
        return SSHCommandResult(
            command=command,
            command_id="probe",
            exit_code=0 if self.success else 1,
            stdout="",
            stderr="",
            duration_ms=1,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            success=self.success,
        )


class DummySSHClient:
    """Mock SSH client that provides SFTP access."""

    def __init__(self, *, sftp_should_raise: bool = False, sftp_error: str = ""):
        self.sftp_client = DummySFTPClient(
            should_raise=sftp_should_raise,
            error_message=sftp_error,
        )

    def start_sftp_client(self):
        """Return an async context manager for SFTP."""
        return self

    async def __aenter__(self):
        """Enter the async context - return the SFTP client."""
        return self.sftp_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context."""
        pass


@pytest.mark.parametrize(
    "has_local_dir,has_executor_root,upload_success,error_msg,expected_pass,expected_reason",
    [
        # Missing local_dir
        (False, True, True, "", False, Msg.CONFIG_MISSING.reason),
        # Missing executor_root
        (True, False, True, "", False, Msg.CONFIG_MISSING.reason),
        # Upload succeeds
        (True, True, True, "", True, Msg.UPLOAD_OK.reason),
        # Upload fails with error
        (True, True, False, "Permission denied", False, Msg.UPLOAD_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_upload_files_check(
    has_local_dir,
    has_executor_root,
    upload_success,
    error_msg,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Create mock SSH client with SFTP
    ssh_client = DummySSHClient(
        sftp_should_raise=not upload_success,
        sftp_error=error_msg,
    )

    # Setup services
    services = build_services()

    # Setup config
    config = build_context_config(
        executor_root="/root/app" if has_executor_root else None,
    )

    # Setup state
    state = build_state(
        upload_local_dir="/local/validator/files" if has_local_dir else None,
    )

    # Create context
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        ssh=ssh_client,
    )

    # Run the check
    result = await UploadFilesCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Verify SFTP interactions
    if has_local_dir and has_executor_root:
        # SFTP should have been called
        assert ssh_client.sftp_client.put_called_with is not None
        assert ssh_client.sftp_client.put_called_with["local_path"] == "/local/validator/files"
        # Remote path should be executor_root + random hex (32 chars)
        remote_path = ssh_client.sftp_client.put_called_with["remote_path"]
        assert remote_path.startswith("/root/app/")
        assert len(remote_path) == len("/root/app/") + 32  # UUID hex is 32 chars
        assert ssh_client.sftp_client.put_called_with["recurse"] is True

        # Verify state update on success
        if expected_pass:
            assert "state" in result.updates
            updated_state = result.updates["state"]
            assert updated_state.upload_remote_dir == remote_path
            assert updated_state.remote_dir == remote_path
    else:
        # SFTP should not have been called if config is missing
        assert ssh_client.sftp_client.put_called_with is None


@pytest.mark.parametrize("probe_succeeds", [True, False])
@pytest.mark.asyncio
async def test_upload_files_check_uploads_only_when_the_executor_cannot_run_the_source(
    probe_succeeds,
    context_factory,
):
    # Arrange
    ssh_client = DummySSHClient()
    runner = DummyProbeRunner(success=probe_succeeds)
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
    )

    # Act
    result = await UploadFilesCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert runner.commands == ["/usr/bin/python -I -c 'import psutil, cryptography.fernet'"]
    if probe_succeeds:
        assert result.event.reason_code == Msg.UPLOAD_SKIPPED.reason
        assert ssh_client.sftp_client.put_called_with is None
        assert result.updates["state"].scrape_over_stdin is True
        assert result.updates["state"].remote_dir is None
    else:
        assert result.event.reason_code == Msg.UPLOAD_OK.reason
        assert ssh_client.sftp_client.put_called_with is not None
        assert result.updates["state"].scrape_over_stdin is False
        assert result.updates["state"].remote_dir is not None
