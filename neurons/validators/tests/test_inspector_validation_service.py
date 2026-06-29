from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidator,
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
    def __init__(self, *, sha256: str = "abc123") -> None:
        self.sha256 = sha256
        self.scp_checksum_calls = 0
        self.remote_checksum_calls = 0

    async def get_checksums_over_scp(self, _path: str) -> str:
        self.scp_checksum_calls += 1
        raise AssertionError("inspector must not download libinspector.so for checksums")

    async def get_sha256_checksum_by_path(self, _path: str) -> str:
        self.remote_checksum_calls += 1
        return self.sha256


class FakeValidator:
    @classmethod
    def load_library(cls, _path: str):
        return "fake-lib"

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
    assert shell.remote_checksum_calls == 1
    assert shell.scp_checksum_calls == 0


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
    shell = FakeShell(sha256="different")

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
    assert shell.remote_checksum_calls == 1
    assert shell.scp_checksum_calls == 0


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
    assert shell.remote_checksum_calls == 2
    assert shell.scp_checksum_calls == 0


@pytest.mark.asyncio
async def test_validate_rented_executor_reuses_loaded_library(monkeypatch):
    loaded_lib = object()
    created: list[SharedLibValidator] = []

    class SharedLibValidator(FakeValidator):
        load_calls = 0

        @classmethod
        def load_library(cls, _path: str):
            cls.load_calls += 1
            return loaded_lib

        def __init__(self, lib) -> None:
            assert lib is loaded_lib
            super().__init__(lib)
            created.append(self)

    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.InspectorValidator",
        SharedLibValidator,
    )
    service = InspectorValidationService()
    shell = FakeShell()

    for executor_id in ("exec-a", "exec-b"):
        ssh = FakeSSH()
        result = await service.validate_rented_executor(
            shell,
            ssh,
            SimpleNamespace(
                uuid=executor_id,
                python_path="/usr/bin/python3",
                root_dir="/root/app",
            ),
            {},
        )
        assert result.error is None

    assert SharedLibValidator.load_calls == 1
    assert len(created) == 2
    assert created[0] is not created[1]


def test_inspector_validator_keeps_interleaved_sessions_separate():
    class FakeNativeLib:
        def __init__(self) -> None:
            self.next_session = 100
            self.calls: list[tuple[str, object]] = []
            self.deleted_strings = 0

        def session_new(self):
            self.next_session += 1
            return self.next_session

        def session_del(self, session) -> None:
            self.calls.append(("session_del", session))

        def session_handshake_start(self, session):
            self.calls.append(("handshake_start", session))
            return b'{"open": true}'

        def session_handshake_finish(self, session, reply: bytes) -> int:
            self.calls.append(("handshake_finish", session, reply))
            return 0

        def inspector_generate(self, session):
            self.calls.append(("generate", session))
            return b"request-cipher"

        def inspector_verify(self, session, response: bytes):
            self.calls.append(("verify", session, response))
            return b'{"ok": true}'

        def str_del(self, _ptr) -> None:
            self.deleted_strings += 1

    lib = FakeNativeLib()
    validator_a = InspectorValidator(lib)
    validator_b = InspectorValidator(lib)

    validator_a.start_session()
    validator_b.start_session()

    session_a = validator_a.session
    session_b = validator_b.session
    assert session_a != session_b

    assert validator_a.handshake_start() == '{"open": true}'
    assert validator_b.handshake_start() == '{"open": true}'

    validator_b.handshake_finish('{"reply": "b"}')
    validator_a.handshake_finish('{"reply": "a"}')

    assert validator_a.generate() == "request-cipher"
    assert validator_b.generate() == "request-cipher"

    assert validator_b.verify("response-b") == {"ok": True}
    assert validator_a.verify("response-a") == {"ok": True}

    validator_b.close_session()
    validator_a.close_session()

    assert lib.calls == [
        ("handshake_start", session_a),
        ("handshake_start", session_b),
        ("handshake_finish", session_b, b'{"reply": "b"}'),
        ("handshake_finish", session_a, b'{"reply": "a"}'),
        ("generate", session_a),
        ("generate", session_b),
        ("verify", session_b, b"response-b"),
        ("verify", session_a, b"response-a"),
        ("session_del", session_b),
        ("session_del", session_a),
    ]
    assert lib.deleted_strings == 6
    assert validator_a.session is None
    assert validator_b.session is None


@pytest.mark.asyncio
async def test_validate_rented_executor_reuses_local_checksum(monkeypatch):
    checksum_calls = 0

    def fake_sha256_from_path(_path: str) -> str:
        nonlocal checksum_calls
        checksum_calls += 1
        return "abc123"

    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.sha256_from_path",
        fake_sha256_from_path,
    )
    service = InspectorValidationService()
    shell = FakeShell()

    for executor_id in ("exec-a", "exec-b"):
        ssh = FakeSSH()
        result = await service.validate_rented_executor(
            shell,
            ssh,
            SimpleNamespace(
                uuid=executor_id,
                python_path="/usr/bin/python3",
                root_dir="/root/app",
            ),
            {},
        )
        assert result.error is None

    assert checksum_calls == 1


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
