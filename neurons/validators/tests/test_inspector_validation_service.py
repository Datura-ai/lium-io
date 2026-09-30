from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import shutil
import signal
from types import SimpleNamespace

import pytest
from neurons.validators.src.core.config import Settings, VerifyXSettings
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidator,
    InspectorValidationService,
)
from neurons.validators.src.services.task.messages import InspectorMessages as Msg


FETCH_URL = "https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/libinspector.so"
VALIDATOR_SHA256 = "abc123"


@pytest.fixture(autouse=True)
def enable_collector_ensure(monkeypatch):
    fake_settings = SimpleNamespace(
        INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK=True,
        INSPECTOR_LIBRARY_FETCH_URL=FETCH_URL,
        verifyx=SimpleNamespace(LIBRARY_REFRESH_ENABLED=False),
    )
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.settings",
        fake_settings,
    )
    return fake_settings


@pytest.fixture(autouse=True)
def matching_lib_checksums(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.sha256_from_path",
        lambda _path: VALIDATOR_SHA256,
    )


class FakeSSHClient:
    """`shell.ssh_client`: records every command; `respond(command)` answers it."""

    def __init__(self, respond=None) -> None:
        self.commands: list[str] = []
        self.respond = respond or (lambda _command: SimpleNamespace(stdout="", stderr="", exit_status=0))

    async def run(self, command: str, timeout: float | None = None):
        self.commands.append(command)
        return self.respond(command)


class FakeShell:
    def __init__(self, *, sha256: str = VALIDATOR_SHA256, respond=None) -> None:
        self.sha256 = sha256
        self.scp_checksum_calls = 0
        self.remote_checksum_calls = 0
        self.ssh_client = FakeSSHClient(respond)

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
    assert result.diagnostics["library_refresh"] == "INSPECTOR_LIBRARY_MISMATCH_NO_REFRESH"
    assert ssh.command == ""
    assert shell.remote_checksum_calls == 1
    assert shell.scp_checksum_calls == 0
    # the switch is off: nothing is run on the executor, so /usr/lib is never written
    assert shell.ssh_client.commands == []


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


@pytest.mark.asyncio
async def test_validate_rented_executor_on_an_attested_host_trusts_the_measured_image_not_the_shell():
    # DAH-3275: on a dstack CVM the executor image is measured by the TDX quote; a sha256sum
    # answered by the provider's shell adds nothing, so it is not asked for — and the
    # diagnostics say which of the two vouched for the sensor.
    service = InspectorValidationService()
    ssh = FakeSSH()
    shell = FakeShell(sha256="different")

    result = await service.validate_rented_executor(
        shell,
        ssh,
        SimpleNamespace(uuid="exec-1", python_path="/usr/bin/python3", root_dir="/root/app"),
        {"executor_uuid": "exec-1"},
        sensor_attested=True,
    )

    assert result.error is None
    assert shell.remote_checksum_calls == 0
    assert result.diagnostics["sensor_integrity"] == "tdx_measured_image"


@pytest.mark.asyncio
async def test_validate_rented_executor_marks_the_shell_checksum_unattested():
    service = InspectorValidationService()
    result = await service.validate_rented_executor(
        FakeShell(),
        FakeSSH(),
        SimpleNamespace(uuid="exec-1", python_path="/usr/bin/python3", root_dir="/root/app"),
        {"executor_uuid": "exec-1"},
    )

    assert result.error is None
    assert result.diagnostics["sensor_integrity"] == "shell_sha256_unattested"


# Library refresh for libinspector.so: the libverifyx.so mechanism, under the same
# VERIFYX_LIBRARY_REFRESH_ENABLED switch, fetching INSPECTOR_LIBRARY_FETCH_URL.

REPO = pathlib.Path(__file__).resolve().parents[3]
STALE_SHA256 = "different"
STALE_BYTES = b"stale libinspector build"
EXECUTOR = SimpleNamespace(uuid="exec-1", python_path="/usr/bin/python3", root_dir="/root/app")
needs_shell_tools = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("curl", "mktemp", "sha256sum")),
    reason="needs curl, mktemp and sha256sum",
)


def _ok(stdout: str = "", stderr: str = "", exit_status: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_status=exit_status)


def _install_stdout(sha: str, *, curl_rc: int = 0, mv_rc: int | None = 0) -> str:
    out = f"TMP:/usr/lib/.libinspector.so.AbC123\nCURL_RC:{curl_rc}\nSHA256:{sha}\n"
    return out + (f"MV_RC:{mv_rc}\n" if mv_rc is not None else "")


def refreshing_executor(
    *,
    writable: bool = True,
    install_stdout: str | None = None,
    install_stderr: str = "",
    installs_as: str = VALIDATOR_SHA256,
    install_raises: Exception | None = None,
) -> FakeShell:
    """An executor on a stale libinspector.so that answers the refresh commands with canned output."""

    def respond(command: str):
        if command.startswith("if [ -w "):
            return _ok(f"WRITE_OK:{int(writable)}\n")
        if "mktemp" in command:
            if install_raises:
                raise install_raises
            stdout = install_stdout if install_stdout is not None else _install_stdout(f"{VALIDATOR_SHA256:0>64}")
            if "MV_RC:0" in stdout:
                shell.sha256 = installs_as
            return _ok(stdout, install_stderr)
        return _ok()

    shell = FakeShell(sha256=STALE_SHA256, respond=respond)
    return shell


class LocalExecutor:
    """The executor side for real: the validator's commands run in /bin/sh against a temp
    directory standing in for /usr/lib. `limits` is prepended to every command (e.g. a
    `ulimit -f` that plays a full disk)."""

    def __init__(self, *, limits: str = "") -> None:
        self.limits = limits
        self.ssh_client = self
        self.commands: list[str] = []
        self.outputs: list[str] = []
        self.remote_checksum_calls = 0

    async def run(self, command: str, timeout: float | None = None):
        self.commands.append(command)
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", self.limits + command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        self.outputs.append(out.decode())
        return _ok(out.decode(), err.decode(), proc.returncode)

    async def get_sha256_checksum_by_path(self, path: str) -> str:
        self.remote_checksum_calls += 1
        file = pathlib.Path(path)
        return hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else ""

    def temp_used(self) -> pathlib.Path:
        return pathlib.Path(next(line[4:] for out in self.outputs for line in out.splitlines() if line.startswith("TMP:")))


@pytest.fixture
def refresh_on(enable_collector_ensure):
    enable_collector_ensure.verifyx.LIBRARY_REFRESH_ENABLED = True
    return enable_collector_ensure


@pytest.fixture
def full_sha_validator(monkeypatch):
    """The validator's libinspector.so digest as a real 64-hex sha256 (sha256sum's output)."""
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.sha256_from_path",
        lambda _path: f"{VALIDATOR_SHA256:0>64}",
    )
    return f"{VALIDATOR_SHA256:0>64}"


@pytest.fixture
def local_library(tmp_path, monkeypatch, refresh_on):
    """A stale /usr/lib/libinspector.so in tmp_path, the validator's file served over file://,
    and a service whose executor-side path is the stale one."""
    lib_dir = tmp_path / "usr_lib"
    lib_dir.mkdir()
    lib = lib_dir / "libinspector.so"
    lib.write_bytes(STALE_BYTES)
    source = tmp_path / "raw" / "libinspector.so"
    source.parent.mkdir()
    source.write_bytes(os.urandom(256 * 1024))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    refresh_on.INSPECTOR_LIBRARY_FETCH_URL = source.as_uri()
    monkeypatch.setattr(
        "neurons.validators.src.services.inspector_validation_service.sha256_from_path",
        lambda _path: sha,
    )
    return SimpleNamespace(
        lib=lib, source=source, sha=sha, settings=refresh_on,
        service=InspectorValidationService(lib_path=str(lib)),
        leftovers=lambda: sorted(p.name for p in lib_dir.iterdir() if p.name != "libinspector.so"),
    )


def _kinds(shell) -> list[str]:
    return [
        "write-check" if command.startswith("if ") else "install" if "mktemp" in command else command.split()[0]
        for command in shell.ssh_client.commands
    ]


async def _validate(shell, ssh: FakeSSH | None = None, service: InspectorValidationService | None = None):
    return await (service or InspectorValidationService()).validate_rented_executor(
        shell, ssh or FakeSSH(), EXECUTOR, {"executor_uuid": "exec-1"}
    )


def test_library_refresh_is_off_by_default_and_fetches_the_executors_libinspector_so():
    assert VerifyXSettings.model_fields["LIBRARY_REFRESH_ENABLED"].default is False
    default_url = Settings.model_fields["INSPECTOR_LIBRARY_FETCH_URL"].default
    assert default_url == FETCH_URL
    # the default URL serves the executor's copy, which is the validator's file byte for byte
    shipped = REPO / "neurons/executor/libinspector.so"
    assert shipped.read_bytes() == (REPO / "neurons/validators/libinspector.so").read_bytes()


@pytest.mark.asyncio
async def test_refresh_on_and_hash_match_runs_nothing_on_the_executor(refresh_on):
    shell = refreshing_executor()
    shell.sha256 = VALIDATOR_SHA256
    result = await _validate(shell)
    assert result.error is None
    assert shell.ssh_client.commands == []
    assert "library_refresh" not in result.diagnostics
    assert shell.remote_checksum_calls == 1


@needs_shell_tools
@pytest.mark.asyncio
async def test_mismatch_fetches_installs_by_rename_and_the_check_passes(local_library):
    shell = LocalExecutor()
    ssh = FakeSSH()
    old_inode = local_library.lib.stat().st_ino
    result = await _validate(shell, ssh, local_library.service)
    assert result.error is None
    assert result.report is not None
    assert result.diagnostics["library_refresh"] == "INSPECTOR_LIBRARY_REPLACED"
    assert _kinds(shell) == ["write-check", "install"]
    assert local_library.lib.read_bytes() == local_library.source.read_bytes()
    # a rename puts a new inode at the path; a copy would have rewritten the old one in place
    assert local_library.lib.stat().st_ino != old_inode
    assert oct(local_library.lib.stat().st_mode & 0o777) == "0o644"
    # downloaded beside the library (same filesystem, so mv is a rename), and nothing left over
    assert shell.temp_used().parent == local_library.lib.parent
    assert shell.temp_used().name.startswith(".libinspector.so.")
    assert local_library.leftovers() == []
    assert shell.remote_checksum_calls == 2
    assert "--interactive" in ssh.command


@needs_shell_tools
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "limits", "fetch_error"),
    [
        pytest.param(lambda lib: setattr(lib.service, "local_checksum", "f" * 64), "",
                     "fetched sha256 {sha} != validator " + "f" * 64, id="fetched-hash-differs"),
        pytest.param(lambda lib: setattr(lib.settings, "INSPECTOR_LIBRARY_FETCH_URL", lib.source.with_name("missing.so").as_uri()),
                     "", "curl exit 37: ", id="fetch-fails"),
        # a full disk: `ulimit -f 32` stops writes far below the 256 KiB download
        pytest.param(None, "ulimit -f 32; ", "curl exit ", id="full-disk"),
        # curl wrote every byte (the hash matches) but reported an error, e.g. --max-time at the end
        pytest.param(None, 'curl() { command curl "$@"; return 28; }; ', "curl exit 28: ", id="curl-error-at-end"),
    ],
)
async def test_a_failed_fetch_installs_nothing_and_is_logged(local_library, caplog, setup, limits, fetch_error):
    if setup:
        setup(local_library)
    shell, ssh = LocalExecutor(limits=limits), FakeSSH()
    with caplog.at_level("WARNING"):
        result = await _validate(shell, ssh, local_library.service)
    assert result.message.reason == Msg.FAILED_LIB_MISMATCH.reason
    assert result.diagnostics["library_refresh"] == "INSPECTOR_LIBRARY_FETCH_FAILED"
    assert result.diagnostics["fetch_error"].startswith(fetch_error.format(sha=local_library.sha))
    assert "INSPECTOR_LIBRARY_FETCH_FAILED" in caplog.text
    assert shell.temp_used().parent == local_library.lib.parent
    assert local_library.lib.read_bytes() == STALE_BYTES
    assert local_library.leftovers() == []
    assert shell.remote_checksum_calls == 1
    assert ssh.command == ""


@needs_shell_tools
@pytest.mark.asyncio
async def test_a_dropped_ssh_session_removes_the_download(local_library):
    stalled = local_library.source.with_name("stalled.so")
    os.mkfifo(stalled)  # curl blocks opening it: a download in flight
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh", "-c", local_library.service._install_command(stalled.as_uri()),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    for _ in range(250):
        if local_library.leftovers():
            break
        await asyncio.sleep(0.02)
    assert local_library.leftovers(), "the temp file exists while the download runs"
    os.killpg(proc.pid, signal.SIGHUP)  # what the session's processes get when SSH drops
    await asyncio.wait_for(proc.communicate(), 10)
    assert local_library.leftovers() == []
    assert local_library.lib.read_bytes() == STALE_BYTES


@needs_shell_tools
@pytest.mark.asyncio
async def test_a_url_with_a_quote_and_command_substitution_is_one_word(local_library):
    source = local_library.source.with_name("lib'$(id).so")
    local_library.source.rename(source)
    url = f"file://{source}"
    local_library.settings.INSPECTOR_LIBRARY_FETCH_URL = url
    shell = LocalExecutor()
    result = await _validate(shell, service=local_library.service)
    assert result.diagnostics["library_refresh"] == "INSPECTOR_LIBRARY_REPLACED"
    assert local_library.lib.read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_a_stray_stdout_line_is_not_read_as_the_fetched_hash(refresh_on, full_sha_validator):
    shell = refreshing_executor(
        install_stdout=_install_stdout(full_sha_validator) + f"{'e' * 64}  motd\n",
        installs_as=full_sha_validator,
    )
    result = await _validate(shell)
    assert result.error is None
    assert _kinds(shell) == ["write-check", "install"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("executor", "outcome", "field", "expected", "checksum_calls"),
    [
        (lambda: refreshing_executor(install_stdout="TMP:/usr/lib/.x\nCURL_RC:0\nSHA256:sha256sum: not found\n"),
         "FETCH_FAILED", "fetch_error", "fetched sha256 None != validator {sha}", 1),
        (lambda: refreshing_executor(install_raises=OSError("connection reset")),
         "FETCH_FAILED", "fetch_error", "OSError: connection reset", 1),
        (lambda: refreshing_executor(writable=False), "WRITE_DENIED", "error",
         "Executor libinspector.so hash mismatch and /usr/lib is not writable", 1),
        (lambda: refreshing_executor(install_stdout="MKTEMP_FAILED\n", install_stderr="mktemp: No space left on device"),
         "FETCH_FAILED", "fetch_error", "mktemp next to /usr/lib/libinspector.so failed", 1),
        (lambda: refreshing_executor(install_stdout=_install_stdout(f"{VALIDATOR_SHA256:0>64}", mv_rc=1)),
         "FETCH_FAILED", "fetch_error", "mv exit 1", 1),
        (lambda: refreshing_executor(installs_as="e" * 64), "STILL_MISMATCHED", "executor_sha256", "e" * 64, 2),
    ],
    ids=["hash-marker-not-a-sha256", "transport-error", "read-only-usr-lib", "mktemp-fails", "rename-fails",
         "still-mismatched-after-install"],
)
async def test_a_refresh_that_does_not_end_on_the_validators_hash_fails_the_check(
    refresh_on, full_sha_validator, executor, outcome, field, expected, checksum_calls
):
    shell, ssh = executor(), FakeSSH()
    result = await _validate(shell, ssh)
    assert result.message.reason == Msg.FAILED_LIB_MISMATCH.reason
    assert result.diagnostics["library_refresh"] == f"INSPECTOR_LIBRARY_{outcome}"
    detail = result.error if field == "error" else result.diagnostics[field]
    assert detail.startswith(expected.format(sha=full_sha_validator))
    assert _kinds(shell) == (["write-check"] if outcome == "WRITE_DENIED" else ["write-check", "install"])
    assert shell.remote_checksum_calls == checksum_calls
    assert ssh.command == ""


@pytest.mark.asyncio
async def test_attested_host_never_refreshes(refresh_on):
    shell = refreshing_executor()
    result = await InspectorValidationService().validate_rented_executor(
        shell, FakeSSH(), EXECUTOR, {"executor_uuid": "exec-1"}, sensor_attested=True
    )
    assert result.error is None
    assert shell.ssh_client.commands == []
