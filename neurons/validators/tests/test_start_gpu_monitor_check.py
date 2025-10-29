from datetime import datetime, UTC
from unittest.mock import Mock
import pytest

from neurons.validators.src.services.task.checks.start_gpu_monitor import StartGPUMonitorCheck
from neurons.validators.src.services.task.messages import StartGpuMonitorMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state


def make_command_result(success: bool, stdout: str = "", stderr: str = "") -> SSHCommandResult:
    """Helper to create mock SSH command results."""
    return SSHCommandResult(
        command="test command",
        command_id="cmd-123",
        exit_code=0 if success else 1,
        stdout=stdout,
        stderr=stderr,
        duration_ms=100,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        success=success,
    )


class DummySSHCommandRunner:
    def __init__(self, *, check_result: SSHCommandResult, start_result: SSHCommandResult):
        """
        Args:
            check_result: Result for the 'ps aux | grep' check command
            start_result: Result for the 'nohup python ...' start command
        """
        self.check_result = check_result
        self.start_result = start_result
        self.commands_called: list[dict] = []

    async def run(self, command: str, timeout: int = 300, retryable: bool = False) -> SSHCommandResult:
        """Mock method that returns different results based on the command."""
        self.commands_called.append({
            "command": command,
            "timeout": timeout,
            "retryable": retryable,
        })

        # Return check_result for ps aux grep command
        if "ps aux | grep" in command:
            return self.check_result
        # Return start_result for nohup command
        elif "nohup" in command:
            return self.start_result
        # Return success for pip install
        else:
            return make_command_result(success=True)


@pytest.mark.parametrize(
    "has_url,already_running,start_success,expected_pass,expected_reason",
    [
        # Missing config (no URL)
        (False, False, True, False, Msg.CONFIG_MISSING.reason),
        # Already running - should pass without starting
        (True, True, True, True, Msg.ALREADY_RUNNING.reason),
        # Not running, start succeeds
        (True, False, True, True, Msg.STARTED.reason),
        # Not running, start fails
        (True, False, False, False, Msg.START_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_start_gpu_monitor_check(
    has_url,
    already_running,
    start_success,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Create mock keypair (always present in production)
    mock_keypair = Mock()
    mock_keypair.sign.return_value = b"\x00" * 64  # Mock signature
    mock_keypair.ss58_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

    # Create command results
    check_result = make_command_result(
        success=True,
        stdout="python /root/app/src/gpus_utility.py" if already_running else "",
    )
    start_result = make_command_result(
        success=start_success,
        stderr="" if start_success else "Failed to start",
    )

    # Create mock runner
    runner = DummySSHCommandRunner(
        check_result=check_result,
        start_result=start_result,
    )

    # Setup services
    services = build_services()

    # Setup config
    config = build_context_config(
        validator_keypair=mock_keypair,
        compute_rest_app_url="http://validator:8000" if has_url else None,
        gpu_monitor_script_relative="src/gpus_utility.py",
    )

    state = build_state()

    # Create context
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        runner=runner,
    )

    # Run the check
    result = await StartGPUMonitorCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Verify runner interactions based on scenario
    if not has_url:
        # Should not call runner if config is missing
        assert len(runner.commands_called) == 0
    elif already_running:
        # Should only check if running, not start
        assert len(runner.commands_called) == 1
        assert "ps aux | grep" in runner.commands_called[0]["command"]
        assert runner.commands_called[0]["timeout"] == 10
        assert runner.commands_called[0]["retryable"] is False
    else:
        # Should check, install deps, then start
        assert len(runner.commands_called) == 3

        # First: check if running
        assert "ps aux | grep" in runner.commands_called[0]["command"]

        # Second: install dependencies
        assert "pip install" in runner.commands_called[1]["command"]
        assert "aiohttp" in runner.commands_called[1]["command"]
        assert runner.commands_called[1]["retryable"] is True

        # Third: start the monitor with nohup
        assert "nohup" in runner.commands_called[2]["command"]
        assert "/src/gpus_utility.py" in runner.commands_called[2]["command"]
        assert "--program_id" in runner.commands_called[2]["command"]
        assert "--signature 0x" in runner.commands_called[2]["command"]
        assert f"--executor_id {ctx.executor.uuid}" in runner.commands_called[2]["command"]
        assert "--validator_hotkey 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY" in runner.commands_called[2]["command"]
        assert "--compute_rest_app_url http://validator:8000" in runner.commands_called[2]["command"]
        assert runner.commands_called[2]["timeout"] == 50
        assert runner.commands_called[2]["retryable"] is False
