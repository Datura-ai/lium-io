import json
from datetime import datetime, UTC
import pytest

from neurons.validators.src.services.task.checks.machine_spec_scrape import (
    MachineSpecScrapeCheck,
    _normalize_gpu_details,
)
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import DummySSHClient, build_context_config, build_services, build_state


# Mock SSH command result matching the real SSHCommandResult
def make_command_result(
    success: bool,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    command: str = "test command",
    duration_ms: int = 100,
) -> SSHCommandResult:
    """Helper to create mock SSH command results."""
    return SSHCommandResult(
        command=command,
        command_id="cmd-123",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        success=success,
        error_type=None if success else "execution_failed",
    )


# Mock SSHCommandRunner
class DummySSHCommandRunner:
    def __init__(self, *, result: SSHCommandResult | None = None, results: list | None = None):
        """
        Args:
            result: The SSHCommandResult to return on every run() call
            results: One result per run() call, in order — for the stdin-then-binary fallback
        """
        self.results = results if results is not None else [result]
        self.calls: list[dict] = []
        self.called_with: dict | None = None

    async def run(
        self,
        command: str,
        timeout: int = 300,
        retryable: bool = False,
        stdin_text: str | None = None,
    ) -> SSHCommandResult:
        """Mock method that mimics the real SSH command runner."""
        # Track what parameters we were called with
        self.called_with = {
            "command": command,
            "timeout": timeout,
            "retryable": retryable,
            "stdin_text": stdin_text,
        }
        self.calls.append(self.called_with)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


# Mock SSHService for decryption
class DummySSHService:
    def __init__(self, *, decrypted_data: dict, valid_payload: str | None = None):
        """
        Args:
            decrypted_data: The decrypted machine specs to return
            valid_payload: The only line that authenticates, as Fernet would — anything else raises
        """
        self.decrypted_data = decrypted_data
        self.valid_payload = valid_payload
        self.decrypt_called_with: dict | None = None

    def decrypt_payload(self, encrypt_key: str, payload: str) -> str:
        """Mock decrypt method - just returns JSON of our mock data."""
        self.decrypt_called_with = {
            "encrypt_key": encrypt_key,
            "payload": payload,
        }
        if self.valid_payload is not None and payload != self.valid_payload:
            raise ValueError("Invalid token")
        return json.dumps(self.decrypted_data)


def test_normalize_gpu_details_canonicalizes_a10_alias():
    assert _normalize_gpu_details(
        [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]
    ) == [
        {"name": "NVIDIA A10 Tensor Core GPU", "uuid": "GPU-abc123"}
    ]


@pytest.mark.asyncio
async def test_machine_spec_scrape_preserves_raw_a10_name_for_native_challenge(
    context_factory,
):
    raw_specs = {
        "gpu": {
            "count": 1,
            "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}],
        },
    }
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout="encrypted_payload_here")
    )
    services = build_services(ssh=DummySSHService(decrypted_data=raw_specs))
    config = build_context_config(
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
    )
    ctx = context_factory(
        services=services,
        config=config,
        state=build_state(remote_dir="/remote/path"),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    result = await MachineSpecScrapeCheck().run(ctx)

    assert result.passed is True
    updated_state = result.updates["state"]
    assert updated_state.specs["gpu"]["details"][0]["name"] == "NVIDIA A10"
    assert updated_state.gpu_details[0]["name"] == "NVIDIA A10 Tensor Core GPU"
    assert updated_state.gpu_model == "NVIDIA A10 Tensor Core GPU"
    assert updated_state.gpu_model_count == "NVIDIA A10 Tensor Core GPU:1"


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
        assert updated_state.specs.get("gpu") == mock_specs["gpu"]
        assert updated_state.specs.get("cpu") == mock_specs["cpu"]
        assert updated_state.specs.get("gpu_processes") == mock_specs["gpu_processes"]
        assert updated_state.specs.get("sysbox_runtime") == mock_specs["sysbox_runtime"]
        # EMA network fields are always added; both None since mock_specs has no network data
        assert updated_state.specs.get("network", {}).get("ema_download_speed") is None
        assert updated_state.specs.get("network", {}).get("ema_upload_speed") is None
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


@pytest.mark.asyncio
async def test_machine_spec_scrape_pipes_source_to_the_executor_interpreter(context_factory):
    # Arrange
    source = "print('obfuscated scrape')"
    raw_specs = {"gpu": {"count": 1, "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]}}
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout="encrypted_payload_here")
    )
    ctx = context_factory(
        services=build_services(ssh=DummySSHService(decrypted_data=raw_specs)),
        config=build_context_config(machine_scrape_source=source),
        # No remote_dir: source delivery uploads nothing, so there is no directory to name.
        state=build_state(scrape_over_stdin=True),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert runner.called_with["command"] == "/usr/bin/python -I -"
    assert runner.called_with["stdin_text"] == source


@pytest.mark.asyncio
async def test_machine_spec_scrape_runs_the_uploaded_binary_when_stdin_was_not_chosen(
    context_factory,
):
    # The source is built every cycle, but only a state that says so puts this executor on the
    # stdin path — with the flag off, UploadFilesCheck uploaded the binary and it is what runs.
    # Arrange
    raw_specs = {"gpu": {"count": 1, "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]}}
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout="encrypted_payload_here")
    )
    ctx = context_factory(
        services=build_services(ssh=DummySSHService(decrypted_data=raw_specs)),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(remote_dir="/remote/path", scrape_over_stdin=False),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert runner.called_with["command"] == "chmod +x /remote/path/scrape.sh && /remote/path/scrape.sh"
    assert runner.called_with["stdin_text"] is None


@pytest.mark.asyncio
async def test_machine_spec_scrape_uploads_the_binary_when_the_source_will_not_run(
    context_factory,
):
    # DAH-2794: nothing was uploaded, and this executor's interpreter rejects the source —
    # missing psutil, wrong Python, a cryptography too old. The binary carries all of it.
    # Arrange
    raw_specs = {"gpu": {"count": 1, "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]}}
    runner = DummySSHCommandRunner(
        results=[
            make_command_result(
                success=False,
                exit_code=1,
                stderr="ModuleNotFoundError: No module named 'psutil'",
                duration_ms=1600,
            ),
            make_command_result(success=True, stdout="encrypted_payload_here"),
        ]
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(ssh=DummySSHService(decrypted_data=raw_specs)),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(scrape_over_stdin=True, upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.event.reason_code == Msg.SCRAPE_OK.reason
    assert [call["command"] for call in runner.calls] == [
        "/usr/bin/python -I -",
        f"chmod +x {ssh_client.sftp_client.put_called_with['remote_path']}/scrape.sh"
        f" && {ssh_client.sftp_client.put_called_with['remote_path']}/scrape.sh",
    ]
    assert result.event.what_we_saw["delivery"] == "upload"
    # The stdin failure is only visible here — its own event was discarded with the retry.
    assert result.event.what_we_saw["fallback_from"]["reason"] == Msg.SCRAPE_FAILED.reason
    assert "psutil" in result.event.what_we_saw["fallback_from"]["stderr_tail"]
    assert result.updates["state"].remote_dir == ssh_client.sftp_client.put_called_with["remote_path"]


@pytest.mark.asyncio
async def test_machine_spec_scrape_keeps_the_stdin_verdict_when_the_failure_was_slow(
    context_factory,
):
    # A scrape that dies deep in the run — past the network benchmark — is not an incompatible
    # interpreter, and an upload plus a second full scrape would blow the per-executor budget.
    # Arrange
    runner = DummySSHCommandRunner(
        result=make_command_result(success=False, exit_code=1, stderr="boom", duration_ms=240_000)
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(scrape_over_stdin=True, upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is False
    assert result.event.reason_code == Msg.SCRAPE_FAILED.reason
    assert result.event.what_we_saw["delivery"] == "stdin"
    assert len(runner.calls) == 1
    assert ssh_client.sftp_client.put_called_with is None


@pytest.mark.parametrize("over_stdin", [True, False])
@pytest.mark.asyncio
async def test_machine_spec_scrape_finds_the_payload_among_other_stdout_lines(
    over_stdin, context_factory
):
    # The image decides what else lands on stdout — a .pth prints before the scrape, an atexit
    # handler after it. Neither the first nor the last line is a safe bet on either path.
    # Arrange
    raw_specs = {"gpu": {"count": 1, "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]}}
    ssh_service = DummySSHService(
        decrypted_data=raw_specs, valid_payload="encrypted_payload_here"
    )
    runner = DummySSHCommandRunner(
        result=make_command_result(
            success=True,
            stdout="sitecustomize: loaded\nencrypted_payload_here\nExiting worker thread",
        )
    )
    ctx = context_factory(
        services=build_services(ssh=ssh_service),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(scrape_over_stdin=over_stdin, remote_dir="/remote/path"),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.updates["state"].gpu_count == 1


@pytest.mark.asyncio
async def test_machine_spec_scrape_falls_back_when_the_stdin_payload_will_not_decrypt(
    context_factory,
):
    # The failure mode a probe cannot see: the modules import, the scrape runs, and the token it
    # produces is not one this validator can read.
    # Arrange
    raw_specs = {"gpu": {"count": 1, "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]}}
    ssh_service = DummySSHService(
        decrypted_data=raw_specs, valid_payload="encrypted_payload_here"
    )
    runner = DummySSHCommandRunner(
        results=[
            make_command_result(success=True, stdout="unreadable_token", duration_ms=2000),
            make_command_result(success=True, stdout="encrypted_payload_here"),
        ]
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(ssh=ssh_service),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(scrape_over_stdin=True, upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert len(runner.calls) == 2
    assert ssh_client.sftp_client.put_called_with is not None
    fallback_from = result.event.what_we_saw["fallback_from"]
    # exit 0 and an empty stderr say nothing here — the exception and the output are the evidence.
    assert fallback_from["reason"] == Msg.SCRAPE_PARSE_FAILED.reason
    assert fallback_from["stdout_head"] == "unreadable_token"


@pytest.mark.asyncio
async def test_machine_spec_scrape_reports_the_stdin_failure_when_the_fallback_upload_fails(
    context_factory,
):
    # Arrange
    runner = DummySSHCommandRunner(
        result=make_command_result(success=False, exit_code=1, stderr="boom", duration_ms=1600)
    )
    ssh_client = DummySSHClient(sftp_should_raise=True, sftp_error="Permission denied")
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(scrape_over_stdin=True, upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is False
    assert result.event.reason_code == Msg.SCRAPE_FAILED.reason
    assert "Permission denied" in result.event.what_we_saw["fallback_upload_error"]
    assert result.event.what_we_saw["fallback_from"]["stderr_tail"] == "boom"
