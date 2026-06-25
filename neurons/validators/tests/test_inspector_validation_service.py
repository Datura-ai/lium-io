from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationService,
)
from neurons.validators.src.services.task.messages import InspectorMessages as Msg


@pytest.fixture(autouse=True)
def enable_collector_ensure(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.settings",
        SimpleNamespace(INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK=True),
    )


@pytest.fixture(autouse=True)
def matching_lib_checksums(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.sha256_from_path",
        lambda _path: "abc123",
    )


class FakeShell:
    def __init__(self, *, checksums: str = "md5:abc123") -> None:
        self.checksums = checksums

    async def get_checksums_over_scp(self, _path: str) -> str:
        return self.checksums


class FakeValidator:
    def __init__(self, *_args, **_kwargs) -> None:
        self.session_closed = False
        created_validators.append(self)

    def start_session(self) -> None:
        pass

    def close_session(self) -> None:
        self.session_closed = True

    def handshake_start(self) -> str:
        return '{"hello": "validator"}'

    def handshake_finish(self, reply_json: str) -> None:
        assert reply_json == '{"hello": "executor"}'

    def generate(self) -> str:
        return "request-cipher"

    def verify(self, response_cipher: str) -> dict:
        assert response_cipher == "response-cipher"
        return {
            "canary_ok": True,
            "health": {
                "ok": True,
                "collector_started_unix": 1,
                "events_dropped": 0,
                "bytes_buffered": 0,
            },
            "findings": [],
            "summary": {"malicious": 0},
        }


class FakeStdin:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def write(self, data: str) -> None:
        self.messages.append(json.loads(data))


class FakeStdout:
    def __init__(self, lines: list[str], *, delay_first: float = 0) -> None:
        self.lines = lines
        self.delay_first = delay_first
        self._delayed = False

    async def readline(self) -> str:
        if self.delay_first and not self._delayed:
            self._delayed = True
            await asyncio.sleep(self.delay_first)
        return self.lines.pop(0)


class FakeStderr:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    async def readline(self) -> str:
        if not self.lines:
            return ""
        return self.lines.pop(0)


def _interactive_stdout(*, ensure_ok: bool = True) -> list[str]:
    lines = []
    if ensure_ok:
        lines.append(json.dumps({"ok": True, "result": ""}) + "\n")
    else:
        lines.append(json.dumps({"ok": False, "error": "collector_start failed"}) + "\n")
    lines.extend([
        json.dumps({"ok": True, "result": '{"hello": "executor"}'}) + "\n",
        json.dumps({"ok": True, "result": "response-cipher"}) + "\n",
        json.dumps({"ok": True, "result": ""}) + "\n",
    ])
    return lines


class FakeProcess:
    def __init__(self, *, ensure_ok: bool = True) -> None:
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(_interactive_stdout(ensure_ok=ensure_ok))
        self.stderr = FakeStderr([])
        self.waited = False

    async def wait(self) -> None:
        self.waited = True


class FakeSSH:
    def __init__(self, *, ensure_ok: bool = True) -> None:
        self.process = FakeProcess(ensure_ok=ensure_ok)
        self.command = ""

    async def create_process(self, command: str):
        self.command = command
        return self.process


created_validators: list[FakeValidator] = []


@pytest.fixture(autouse=True)
def fake_validator(monkeypatch):
    created_validators.clear()
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.InspectorValidator",
        FakeValidator,
    )


@pytest.mark.asyncio
async def test_validate_rented_executor_uses_interactive_json_protocol():
    service = InspectorValidationService()
    ssh = FakeSSH()
    shell = FakeShell()
    executor = SimpleNamespace(
        uuid="exec-1",
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )

    result = await service.validate_rented_executor(
        shell,
        ssh,
        executor,
        {"executor_uuid": "exec-1"},
    )

    assert result.error is None
    assert result.report is not None
    assert result.report["canary_ok"] is True
    assert result.report["findings"] == []
    assert "python3 /root/app/src/inspector_executor.py --interactive" in ssh.command
    assert result.diagnostics is not None
    assert "collector_ensure_error" not in result.diagnostics
    assert ssh.process.stdin.messages == [
        {"cmd": "start-collector"},
        {"cmd": "handshake-reply", "open_json": '{"hello": "validator"}'},
        {"cmd": "execute", "request_cipher": "request-cipher"},
        {"cmd": "quit"},
    ]
    assert ssh.process.waited is True
    assert len(created_validators) == 1
    assert created_validators[0].session_closed is True


@pytest.mark.asyncio
async def test_validate_rented_executor_skips_collector_ensure_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.settings",
        SimpleNamespace(INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK=False),
    )
    service = InspectorValidationService()
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout([
        json.dumps({"ok": True, "result": '{"hello": "executor"}'}) + "\n",
        json.dumps({"ok": True, "result": "response-cipher"}) + "\n",
        json.dumps({"ok": True, "result": ""}) + "\n",
    ])
    shell = FakeShell()
    executor = SimpleNamespace(
        uuid="exec-1",
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )

    result = await service.validate_rented_executor(
        shell,
        ssh,
        executor,
        {"executor_uuid": "exec-1"},
    )

    assert result.error is None
    assert ssh.process.stdin.messages[0]["cmd"] == "handshake-reply"
    assert "collector_ensure_error" not in (result.diagnostics or {})


@pytest.mark.asyncio
async def test_validate_rented_executor_continues_after_collector_ensure_failure():
    service = InspectorValidationService()
    ssh = FakeSSH(ensure_ok=False)
    shell = FakeShell()
    executor = SimpleNamespace(
        uuid="exec-1",
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )

    result = await service.validate_rented_executor(
        shell,
        ssh,
        executor,
        {"executor_uuid": "exec-1"},
    )

    assert result.error is None
    assert result.diagnostics is not None
    assert result.diagnostics["collector_ensure_error"] == "collector_start failed"
    assert result.report is not None


@pytest.mark.asyncio
async def test_validate_rented_executor_returns_interactive_error():
    service = InspectorValidationService()
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout([
        json.dumps({"ok": True, "result": ""}) + "\n",
        json.dumps({"ok": False, "error": "bad handshake"}) + "\n",
        json.dumps({"ok": True, "result": ""}) + "\n",
    ])
    ssh.process.stderr = FakeStderr([
        "return_string: decrypt failed\n",
    ])
    shell = FakeShell()
    executor = SimpleNamespace(
        uuid="exec-1",
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )

    result = await service.validate_rented_executor(
        shell,
        ssh,
        executor,
        {"executor_uuid": "exec-1"},
    )

    assert result.report is None
    assert result.error == "bad handshake"
    assert result.diagnostics is not None
    assert result.message is not None
    assert result.message.reason == Msg.FAILED_INTERACTIVE.reason
    assert result.diagnostics["error_type"] == "InspectorInteractiveError"
    assert result.diagnostics["executor_stderr"] == "return_string: decrypt failed"


@pytest.mark.asyncio
async def test_validate_rented_executor_returns_lib_mismatch_without_ssh_process():
    service = InspectorValidationService()
    ssh = FakeSSH()
    shell = FakeShell(checksums="md5:different")

    result = await service.validate_rented_executor(
        shell,
        ssh,
        SimpleNamespace(
            uuid="exec-1",
            python_path="/usr/bin/python3",
            root_dir="/root/app",
        ),
        {"executor_uuid": "exec-1"},
    )

    assert result.report is None
    assert "outdated libinspector" in (result.error or "")
    assert result.diagnostics is not None
    assert result.message is not None
    assert result.message.reason == Msg.FAILED_LIB_MISMATCH.reason
    assert result.diagnostics["local_sha256"] == "abc123"
    assert result.diagnostics["executor_sha256"] == "different"
    assert ssh.command == ""


@pytest.mark.asyncio
async def test_validate_rented_executor_classifies_ssh_transport_error():
    service = InspectorValidationService()
    shell = FakeShell()

    class BrokenSSH:
        async def create_process(self, _command: str):
            raise OSError("connection reset")

    result = await service.validate_rented_executor(
        shell,
        BrokenSSH(),
        SimpleNamespace(
            uuid="exec-1",
            python_path="/usr/bin/python3",
            root_dir="/root/app",
        ),
        {"executor_uuid": "exec-1"},
    )

    assert result.diagnostics is not None
    assert result.message is not None
    assert result.message.reason == Msg.FAILED_SSH_TRANSPORT.reason


@pytest.mark.asyncio
async def test_validate_rented_executor_concurrent_uses_separate_validators():
    service = InspectorValidationService()
    shell = FakeShell()
    executor_a = SimpleNamespace(
        uuid="exec-a",
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    executor_b = SimpleNamespace(
        uuid="exec-b",
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    ssh_a = FakeSSH()
    ssh_b = FakeSSH()
    ssh_a.process.stdout = FakeStdout(_interactive_stdout(), delay_first=0.05)
    ssh_b.process.stdout = FakeStdout(_interactive_stdout(), delay_first=0.05)

    result_a, result_b = await asyncio.gather(
        service.validate_rented_executor(shell, ssh_a, executor_a, {}),
        service.validate_rented_executor(shell, ssh_b, executor_b, {}),
    )

    assert result_a.error is None
    assert result_b.error is None
    assert result_a.report is not None
    assert result_b.report is not None
    assert len(created_validators) == 2
    assert created_validators[0] is not created_validators[1]
    assert all(v.session_closed for v in created_validators)


@pytest.mark.asyncio
async def test_capture_stderr_is_bounded():
    service = InspectorValidationService(
        stderr_capture_timeout=0.01,
        stderr_capture_max_bytes=12,
    )
    process = SimpleNamespace(stderr=FakeStderr(["rust side failure\n"]))

    captured = await service._capture_stderr(process)

    assert captured == "rust side fa"


@pytest.mark.asyncio
async def test_capture_stderr_does_not_wait_forever():
    class BlockingStderr:
        async def readline(self) -> str:
            await asyncio.sleep(10)
            return "too late"

    service = InspectorValidationService(stderr_capture_timeout=0.01)
    process = SimpleNamespace(stderr=BlockingStderr())

    captured = await service._capture_stderr(process)

    assert captured is None
