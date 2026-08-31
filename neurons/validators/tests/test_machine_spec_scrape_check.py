import json
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any

import pytest

from neurons.validators.src.services.task.checks.machine_spec_scrape import (
    MachineSpecScrapeCheck,
    _normalize_gpu_details,
)
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import (
    FERNET_TOKEN,
    DummySSHClient,
    build_context_config,
    build_services,
    build_state,
)

RAW_SPECS = {"gpu": {"count": 1, "details": [{"name": "NVIDIA A10", "uuid": "GPU-abc123"}]}}
UNREADABLE_TOKEN = "gAAAAABnot-ours"


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


@dataclass(frozen=True)
class RunCall:
    command: str
    timeout: int
    retryable: bool
    stdin_text: str | None


class DummySSHCommandRunner:
    def __init__(
        self,
        *,
        result: SSHCommandResult | None = None,
        results: list[SSHCommandResult] | None = None,
    ):
        """
        Args:
            result: The SSHCommandResult to return on every run() call
            results: One result per run() call, in order — for the stdin-then-binary fallback
        """
        self.results = results or [result]
        self.calls: list[RunCall] = []

    @property
    def called_with(self) -> RunCall | None:
        return self.calls[-1] if self.calls else None

    async def run(
        self,
        command: str,
        timeout: int = 300,
        retryable: bool = False,
        stdin_text: str | None = None,
    ) -> SSHCommandResult:
        self.calls.append(RunCall(command, timeout, retryable, stdin_text))
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


# Mock SSHService for decryption
class DummySSHService:
    def __init__(self, *, decrypted_data: dict[str, Any], valid_payload: str | None = None):
        """
        Args:
            decrypted_data: The decrypted machine specs to return
            valid_payload: The only line that authenticates, as Fernet would — anything else raises
        """
        self.decrypted_data = decrypted_data
        self.valid_payload = valid_payload
        self.decrypt_called_with: dict | None = None
        self.decrypt_call_count = 0

    def decrypt_payload(self, encrypt_key: str, payload: str) -> str:
        """Mock decrypt method - just returns JSON of our mock data."""
        self.decrypt_call_count += 1
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
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout=FERNET_TOKEN)
    )
    services = build_services(ssh=DummySSHService(decrypted_data=RAW_SPECS))
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
        (True, True, True, FERNET_TOKEN, True, True, Msg.SCRAPE_OK.reason),
        # Scrape succeeds but no encrypt_key - should fail (parse error)
        (True, True, True, FERNET_TOKEN, False, False, Msg.SCRAPE_PARSE_FAILED.reason),
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
        assert "chmod +x /remote/path/scrape.sh && /remote/path/scrape.sh" in runner.called_with.command
        assert runner.called_with.timeout == 300
        assert runner.called_with.retryable is False

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
        assert ssh_service.decrypt_called_with["payload"] == FERNET_TOKEN


@pytest.mark.asyncio
async def test_machine_spec_scrape_pipes_source_to_the_executor_interpreter(context_factory):
    # Arrange
    source = "print('obfuscated scrape')"
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout=FERNET_TOKEN)
    )
    ctx = context_factory(
        services=build_services(ssh=DummySSHService(decrypted_data=RAW_SPECS)),
        config=build_context_config(machine_scrape_source=source),
        # No remote_dir: source delivery uploads nothing, so there is no directory to name.
        state=build_state(),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert runner.called_with.command == "/usr/bin/python -I -"
    assert runner.called_with.stdin_text == source


@pytest.mark.asyncio
async def test_machine_spec_scrape_runs_the_uploaded_binary_when_no_source_was_delivered(
    context_factory,
):
    # With ENABLE_SCRAPE_SOURCE_DELIVERY off the config carries no source, so UploadFilesCheck
    # uploaded the binary and it is what runs.
    # Arrange
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout=FERNET_TOKEN)
    )
    ctx = context_factory(
        services=build_services(ssh=DummySSHService(decrypted_data=RAW_SPECS)),
        config=build_context_config(machine_scrape_source=None),
        state=build_state(remote_dir="/remote/path"),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert runner.called_with.command == "chmod +x /remote/path/scrape.sh && /remote/path/scrape.sh"
    assert runner.called_with.stdin_text is None


@pytest.mark.asyncio
async def test_machine_spec_scrape_uploads_the_binary_when_the_source_will_not_run(
    context_factory,
):
    # DAH-2794: nothing was uploaded, and this executor's interpreter rejects the source —
    # missing psutil, wrong Python, a cryptography too old. The binary carries all of it.
    # Arrange
    runner = DummySSHCommandRunner(
        results=[
            make_command_result(
                success=False,
                exit_code=1,
                stderr="ModuleNotFoundError: No module named 'psutil'",
                duration_ms=1600,
            ),
            make_command_result(success=True, stdout=FERNET_TOKEN),
        ]
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(ssh=DummySSHService(decrypted_data=RAW_SPECS)),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.event.reason_code == Msg.SCRAPE_OK.reason
    assert [call.command for call in runner.calls] == [
        "/usr/bin/python -I -",
        f"chmod +x {ssh_client.sftp_client.put_called_with.remote_path}/scrape.sh"
        f" && {ssh_client.sftp_client.put_called_with.remote_path}/scrape.sh",
    ]
    assert result.event.what_we_saw["delivery"] == "upload"
    # The stdin failure is only visible here — its own event was discarded with the retry.
    assert result.event.what_we_saw["fallback_from"]["reason"] == Msg.SCRAPE_FAILED.reason
    assert "psutil" in result.event.what_we_saw["fallback_from"]["stderr_tail"]


@pytest.mark.parametrize("duration_ms", [60_001, 240_000])
@pytest.mark.asyncio
async def test_machine_spec_scrape_keeps_the_stdin_verdict_when_the_failure_was_slow(
    duration_ms, context_factory
):
    # A scrape that dies deep in the run — past the network benchmark — is not an incompatible
    # interpreter, and an upload plus a second full scrape would blow the per-executor budget.
    # Arrange
    runner = DummySSHCommandRunner(
        result=make_command_result(
            success=False, exit_code=1, stderr="boom", duration_ms=duration_ms
        )
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(upload_local_dir="/local/validator/files"),
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


@pytest.mark.parametrize("machine_scrape_source", ["print('scrape')", None])
@pytest.mark.asyncio
async def test_machine_spec_scrape_finds_the_payload_among_other_stdout_lines(
    machine_scrape_source, context_factory
):
    # The image decides what else lands on stdout — a .pth prints before the scrape, an atexit
    # handler after it. Neither the first nor the last line is a safe bet on either path.
    # Arrange
    ssh_service = DummySSHService(decrypted_data=RAW_SPECS, valid_payload=FERNET_TOKEN)
    runner = DummySSHCommandRunner(
        result=make_command_result(
            success=True,
            stdout=f"sitecustomize: loaded\n{FERNET_TOKEN}\nExiting worker thread",
        )
    )
    ctx = context_factory(
        services=build_services(ssh=ssh_service),
        config=build_context_config(machine_scrape_source=machine_scrape_source),
        state=build_state(remote_dir="/remote/path"),
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
    ssh_service = DummySSHService(
        decrypted_data=RAW_SPECS, valid_payload=FERNET_TOKEN
    )
    runner = DummySSHCommandRunner(
        results=[
            make_command_result(success=True, stdout=UNREADABLE_TOKEN, duration_ms=15_000),
            make_command_result(success=True, stdout=FERNET_TOKEN),
        ]
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(ssh=ssh_service),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(upload_local_dir="/local/validator/files"),
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
    assert fallback_from["stdout_head"] == UNREADABLE_TOKEN


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
        state=build_state(upload_local_dir="/local/validator/files"),
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
    # One attempt, not two: the retry the legacy path spends is what keeps 60 s + 300 s + 300 s
    # inside the per-executor timeout.
    assert ssh_client.sftp_client.put_call_count == 1


@pytest.mark.asyncio
async def test_machine_spec_scrape_decrypts_nothing_but_the_token_on_a_spammed_stdout(
    context_factory,
):
    # The miner decides how many lines land in front of the payload, and every decrypt runs on the
    # event loop shared with the rest of the cycle — so only token-shaped lines are tried.
    # Arrange
    ssh_service = DummySSHService(decrypted_data=RAW_SPECS, valid_payload=FERNET_TOKEN)
    noise = "\n".join(f"chatter {i}" for i in range(10_000))
    runner = DummySSHCommandRunner(
        result=make_command_result(success=True, stdout=f"{noise}\n{FERNET_TOKEN}")
    )
    ctx = context_factory(
        services=build_services(ssh=ssh_service),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert ssh_service.decrypt_call_count == 1


@pytest.mark.asyncio
async def test_machine_spec_scrape_reads_a_token_longer_than_the_search_window(
    context_factory,
):
    # A byte-wise cut of the tail lands inside a big payload and takes the `gAAAAA` prefix with
    # it, after which no reader recognises the line. Whole lines are searched instead.
    # Arrange
    long_token = FERNET_TOKEN + "x" * (400 * 1024)
    ssh_service = DummySSHService(decrypted_data=RAW_SPECS, valid_payload=long_token)
    runner = DummySSHCommandRunner(result=make_command_result(success=True, stdout=f"chatter\n{long_token}"))
    ctx = context_factory(
        services=build_services(ssh=ssh_service),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert ssh_service.decrypt_call_count == 1


@pytest.mark.asyncio
async def test_machine_spec_scrape_keeps_the_stdin_verdict_when_the_scrape_reported_its_own_error(
    context_factory,
):
    # The scrape printed its own failure and exited 1: the interpreter ran it, so uploading the
    # binary would buy 13 MB and a second full run to be told the same thing.
    # Arrange
    runner = DummySSHCommandRunner(
        result=make_command_result(
            success=False,
            exit_code=1,
            stdout='{"error": "no_gpu_details"}',
            duration_ms=40_000,
        )
    )
    ssh_client = DummySSHClient()
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(machine_scrape_source="print('scrape')"),
        state=build_state(upload_local_dir="/local/validator/files"),
        runner=runner,
        ssh=ssh_client,
        encrypt_key="test-encrypt-key",
    )

    # Act
    result = await MachineSpecScrapeCheck().run(ctx)

    # Assert
    assert result.passed is False
    assert result.event.what_we_saw["delivery"] == "stdin"
    assert len(runner.calls) == 1
    assert ssh_client.sftp_client.put_called_with is None
