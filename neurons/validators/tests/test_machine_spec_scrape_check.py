import json
from datetime import datetime, UTC
import pytest

from neurons.validators.src.services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state


# Mock SSH command result matching the real SSHCommandResult
def make_command_result(
    success: bool,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    command: str = "test command",
) -> SSHCommandResult:
    """Helper to create mock SSH command results."""
    return SSHCommandResult(
        command=command,
        command_id="cmd-123",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=100,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        success=success,
        error_type=None if success else "execution_failed",
    )


# Mock SSHCommandRunner
class DummySSHCommandRunner:
    def __init__(self, *, result: SSHCommandResult):
        """
        Args:
            result: The SSHCommandResult to return when run() is called
        """
        self.result = result
        self.called_with: dict | None = None

    async def run(self, command: str, timeout: int = 300, retryable: bool = False) -> SSHCommandResult:
        """Mock method that mimics the real SSH command runner."""
        # Track what parameters we were called with
        self.called_with = {
            "command": command,
            "timeout": timeout,
            "retryable": retryable,
        }
        return self.result


# Mock SSHService for decryption
class DummySSHService:
    def __init__(self, *, decrypted_data: dict):
        """
        Args:
            decrypted_data: The decrypted machine specs to return
        """
        self.decrypted_data = decrypted_data
        self.decrypt_called_with: dict | None = None

    def decrypt_payload(self, encrypt_key: str, payload: str) -> str:
        """Mock decrypt method - just returns JSON of our mock data."""
        self.decrypt_called_with = {
            "encrypt_key": encrypt_key,
            "payload": payload,
        }
        return json.dumps(self.decrypted_data)


@pytest.mark.parametrize(
    "has_remote_dir,has_script_filename,scrape_success,stdout,has_encrypt_key,expected_pass,expected_reason",
    [
        # No remote_dir - should fail
        (False, True, True, "", True, False, Msg.REMOTE_DIR_MISSING.reason),
        # No script filename - should fail
        (True, False, True, "", True, False, Msg.CONFIG_MISSING.reason),
        # Scrape command fails - should fail
        (True, True, False, "", True, False, Msg.SCRAPE_FAILED.reason),
        # Scrape succeeds but empty stdout - should fail
        (True, True, True, "", True, False, Msg.SCRAPE_FAILED.reason),
        # Scrape succeeds with valid output - should pass
        (True, True, True, "encrypted_payload_here", True, True, Msg.SCRAPE_OK.reason),
        # Scrape succeeds but no encrypt_key - should fail (parse error)
        (True, True, True, "encrypted_payload_here", False, False, Msg.SCRAPE_PARSE_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_machine_spec_scrape_check(
    has_remote_dir,
    has_script_filename,
    scrape_success,
    stdout,
    has_encrypt_key,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Setup mock machine specs that will be "decrypted"
    mock_specs = {
        "gpu": {
            "count": 2,
            "details": [
                {"name": "NVIDIA RTX 3090", "uuid": "GPU-abc123"},
                {"name": "NVIDIA RTX 3090", "uuid": "GPU-def456"},
            ],
        },
        "cpu": {"cores": 8},
        "gpu_processes": [{"pid": 1234, "name": "test"}],
        "sysbox_runtime": True,
    }

    # Create mock SSH command result
    command_result = make_command_result(
        success=scrape_success,
        stdout=stdout,
        stderr="some error" if not scrape_success else "",
        exit_code=0 if scrape_success else 1,
    )

    # Create mock runner
    runner = DummySSHCommandRunner(result=command_result)

    # Create mock SSH service for decryption
    ssh_service = DummySSHService(decrypted_data=mock_specs)

    # Setup services
    services = build_services(ssh=ssh_service)

    # Setup config
    config = build_context_config(
        machine_scrape_filename="scrape.sh" if has_script_filename else None,
        machine_scrape_timeout=300,
        obfuscation_keys={},
    )

    # Setup state
    state = build_state(
        remote_dir="/remote/path" if has_remote_dir else None,
    )

    # Create context
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        runner=runner,
        encrypt_key="test-encrypt-key" if has_encrypt_key else None,
    )

    # Run the check
    result = await MachineSpecScrapeCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Verify runner was called correctly (if we got that far)
    if has_remote_dir and has_script_filename:
        assert runner.called_with is not None
        assert "chmod +x /remote/path/scrape.sh && /remote/path/scrape.sh" in runner.called_with["command"]
        assert runner.called_with["timeout"] == 300
        assert runner.called_with["retryable"] is False

    # Verify state update on success
    if expected_pass:
        assert "state" in result.updates
        updated_state = result.updates["state"]
        # Check that specs were parsed and stored correctly
        assert updated_state.specs == mock_specs
        assert updated_state.gpu_count == 2
        assert updated_state.gpu_model == "NVIDIA RTX 3090"
        assert updated_state.gpu_model_count == "NVIDIA RTX 3090:2"
        assert updated_state.gpu_uuids == "GPU-abc123,GPU-def456"
        assert updated_state.sysbox_runtime is True
        assert len(updated_state.gpu_details) == 2
        assert len(updated_state.gpu_processes) == 1

        # Verify decryption was called
        assert ssh_service.decrypt_called_with is not None
        assert ssh_service.decrypt_called_with["encrypt_key"] == "test-encrypt-key"
        assert ssh_service.decrypt_called_with["payload"] == "encrypted_payload_here"
