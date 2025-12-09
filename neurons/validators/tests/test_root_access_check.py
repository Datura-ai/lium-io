import tempfile
from datetime import datetime, UTC
from unittest.mock import Mock
import pytest

from neurons.validators.src.services.task.checks.root_access import RootAccessCheck
from neurons.validators.src.services.task.messages import RootAccessMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state


def make_command_result(success: bool, stdout: str = "", stderr: str = "", exit_code: int = 0) -> SSHCommandResult:
    """Helper to create mock SSH command results."""
    return SSHCommandResult(
        command="test command",
        command_id="cmd-123",
        exit_code=exit_code if not success else 0,
        stdout=stdout,
        stderr=stderr,
        duration_ms=100,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        success=success,
    )


class DummySSHCommandRunner:
    def __init__(self, *, sshd_running: bool = True, inject_key_success: bool = True):
        """
        Args:
            sshd_running: Whether sshd is running on host
            inject_key_success: Whether SSH key injection succeeds
        """
        self.sshd_running = sshd_running
        self.inject_key_success = inject_key_success
        self.commands_called: list[dict] = []

    async def run(self, command: str, timeout: int = 300, retryable: bool = False) -> SSHCommandResult:
        """Mock method that returns different results based on the command."""
        self.commands_called.append({
            "command": command,
            "timeout": timeout,
            "retryable": retryable,
        })

        # SSH key injection command
        if "docker run" in command and "authorized_keys" in command and "echo" in command:
            return make_command_result(success=self.inject_key_success)

        # SSHD check command
        elif "grep -l sshd /host_proc/*/comm" in command:
            stdout = "/host_proc/1234/comm" if self.sshd_running else ""
            return make_command_result(success=True, stdout=stdout)

        else:
            return make_command_result(success=True)


class DummyShellService:
    def __init__(self, *, connection_success: bool = True, uid: str = "0"):
        """
        Args:
            connection_success: Whether SSH connection test succeeds
            uid: The user ID to return from test_ssh_connection
        """
        self.connection_success = connection_success
        self.uid = uid
        self.test_connection_called = False
        self.test_connection_params = None

    async def test_ssh_connection(self, host: str, port: int, user: str, private_key_path: str) -> tuple[bool, str]:
        """Mock test_ssh_connection method."""
        self.test_connection_called = True
        self.test_connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "private_key_path": private_key_path,
        }

        if self.connection_success:
            return True, self.uid
        else:
            return False, "Connection failed: Authentication failed"


@pytest.mark.parametrize(
    "uid,sshd_running,connection_success,has_keys,expected_pass,expected_reason",
    [
        # Root access confirmed
        ("0", True, True, True, True, Msg.ROOT_OK.reason),
        # Root access confirmed - sshd not running (doesn't affect outcome)
        ("0", False, True, True, True, Msg.ROOT_OK.reason),
        # Root access failed - not root user
        ("1000", True, True, True, False, Msg.ROOT_FAILED.reason),
        # Root access failed - connection failed
        ("0", True, False, True, False, Msg.ROOT_FAILED.reason),
        # Skip check - no keys configured
        ("0", True, True, False, True, Msg.ROOT_OK.reason),
    ],
)
@pytest.mark.asyncio
async def test_root_access_check(
    uid,
    sshd_running,
    connection_success,
    has_keys,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Create temporary key files if needed
    private_key_path = None
    public_key_path = None

    if has_keys:
        # Create temporary key files
        private_key_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_id_rsa')
        private_key_file.write("-----BEGIN RSA PRIVATE KEY-----\nfake_private_key\n-----END RSA PRIVATE KEY-----\n")
        private_key_file.close()
        private_key_path = private_key_file.name

        public_key_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_id_rsa.pub')
        public_key_file.write("ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDfake_public_key test@example.com\n")
        public_key_file.close()
        public_key_path = public_key_file.name

    try:
        # Create mock runner
        runner = DummySSHCommandRunner(
            sshd_running=sshd_running,
            inject_key_success=True,
        )

        # Create mock shell service
        shell_service = DummyShellService(
            connection_success=connection_success,
            uid=uid,
        )

        # Setup services
        services = build_services(shell=shell_service)

        # Setup config
        config = build_context_config(
            root_ssh_private_key_path=private_key_path,
            root_ssh_public_key_path=public_key_path,
        )

        state = build_state()

        # Create mock executor
        mock_executor = Mock()
        mock_executor.address = "127.0.0.1"
        mock_executor.ssh_port = 22

        # Create context
        ctx = context_factory(
            executor=mock_executor,
            services=services,
            config=config,
            state=state,
            runner=runner,
        )

        # Run the check
        result = await RootAccessCheck().run(ctx)

        # Verify result
        assert result.passed is expected_pass
        assert result.event.reason_code == expected_reason

        # Verify behavior based on configuration
        if not has_keys:
            # Should skip and return early
            assert result.event.what_we_saw.get("skipped") is True
            assert not shell_service.test_connection_called
        else:
            # Verify SSH key injection was attempted
            inject_commands = [cmd for cmd in runner.commands_called if "authorized_keys" in cmd["command"] and "docker run" in cmd["command"]]
            assert len(inject_commands) == 1
            assert "echo" in inject_commands[0]["command"]
            assert inject_commands[0]["timeout"] == 15

            # Verify sshd check was performed
            sshd_commands = [cmd for cmd in runner.commands_called if "grep -l sshd" in cmd["command"]]
            assert len(sshd_commands) == 1
            assert sshd_commands[0]["timeout"] == 10

            # Verify test_ssh_connection was called
            assert shell_service.test_connection_called
            assert shell_service.test_connection_params["host"] == "127.0.0.1"
            assert shell_service.test_connection_params["port"] == 22
            assert shell_service.test_connection_params["user"] == "root"
            assert shell_service.test_connection_params["private_key_path"] == private_key_path

            # Verify event details
            if expected_pass and not result.event.what_we_saw.get("skipped"):
                assert result.event.what_we_saw["uid"] == uid
                assert result.event.what_we_saw["sshd_running"] == sshd_running
                assert result.event.what_we_saw["ssh_keys_injected"] == 1

                # Verify specs were updated with root_access
                assert result.updates is not None
                assert "state" in result.updates
                assert result.updates["state"].specs["root_access"] is True

    finally:
        # Cleanup temporary files
        import os
        if private_key_path and os.path.exists(private_key_path):
            os.unlink(private_key_path)
        if public_key_path and os.path.exists(public_key_path):
            os.unlink(public_key_path)
