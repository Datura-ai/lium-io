"""The validator runs only its own scripts on an executor, with the interpreter it names.

`root_dir` and `python_path` come from the executor's answer. Here they point at a directory holding scripts and an
interpreter wrapper that write marker files; the validator's own payloads write a marker of their own. The commands
run in a local shell, with the service directory and the trusted interpreter moved into the test's temp dir.
"""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services import executor_payload
from services.inspector_validation_service import InspectorValidationService
from services.task.checks.node_account import NodeAccountCheck
from services.task.checks.start_gpu_monitor import StartGPUMonitorCheck
from services.task.checks.upload_files import UploadFilesCheck
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult
from services.verifyx_validation_service import VerifyXValidationService
from tests.helpers import build_context_config, build_state, make_context

pytestmark = pytest.mark.asyncio

EXECUTOR_SOURCES = Path(__file__).resolve().parents[2] / "executor" / "src"
EMPTY_ACCOUNT_FILES = "hotkey=\nclaim=\n"


def _marker_script(marker: Path, *, sleep: bool = False) -> str:
    # records its arguments, then (a monitor) stays up like the real one
    tail = "import time; time.sleep(60)\n" if sleep else ""
    return f"import sys; open({str(marker)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n{tail}"


@pytest.fixture
def node(tmp_path, monkeypatch):
    """An executor whose reported root_dir and python_path lead to marker-writing code, and a validator whose payloads
    write a marker of their own."""
    attacker = tmp_path / "volume"
    for name in ("gpus_utility.py", "inspector_executor.py", "verifyx_executor.py"):
        (attacker / "src").mkdir(parents=True, exist_ok=True)
        (attacker / "src" / name).write_text(_marker_script(tmp_path / "ATTACKER_EXECUTED"))
    wrapper = attacker / "pywrap"
    wrapper.write_text(f"#!/bin/sh\ntouch {tmp_path / 'INTERPRETER_EXECUTED'}\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)

    payloads = tmp_path / "validator-payloads"
    (payloads / "services").mkdir(parents=True)
    (payloads / "gpus_utility.py").write_text(_marker_script(tmp_path / "VALIDATOR_MONITOR", sleep=True))
    (payloads / "services" / "pull_lock.py").write_text("")
    (payloads / "inspector_executor.py").write_text(_marker_script(tmp_path / "VALIDATOR_INSPECTOR"))
    (payloads / "verifyx_executor.py").write_text(_marker_script(tmp_path / "VALIDATOR_VERIFYX"))
    (tmp_path / "service").mkdir()
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path / "service"))
    monkeypatch.setattr(executor_payload, "EXECUTOR_PYTHON", sys.executable)

    executor = ExecutorSSHInfo(
        uuid="node-1", address="203.0.113.7", port=8001, ssh_username="root", ssh_port=22,
        python_path=str(wrapper), root_dir=str(attacker), expected_node_hotkeys=["5Account"], expected_account_claim="c",
    )
    yield SimpleNamespace(tmp=tmp_path, executor=executor)
    # the monitor payload stays up like the real one: stop what the tests started
    out = subprocess.run(["pgrep", "-f", str(tmp_path / "service")], capture_output=True, text=True).stdout
    for pid in out.split():
        os.kill(int(pid), signal.SIGKILL)


class LocalRunner:
    """The validator's SSH runner, on a local shell; the /etc/liumd read answers with no identity files."""

    def __init__(self, cwd: Path):
        self.cwd, self.commands = cwd, []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):  # noqa: ASYNC109
        from datetime import UTC, datetime

        self.commands.append(cmd)
        if "/etc/liumd/" in cmd:
            code, out, err = 0, EMPTY_ACCOUNT_FILES, ""
        else:
            done = subprocess.run(["sh", "-c", cmd], cwd=self.cwd, capture_output=True, text=True, timeout=20)
            code, out, err = done.returncode, done.stdout, done.stderr
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd, command_id="c", exit_code=code, stdout=out, stderr=err,
            duration_ms=1, started_at=now, finished_at=now, success=code == 0,
        )


def _ctx(node, runner, **config):
    keypair = SimpleNamespace(sign=lambda data: b"\x01", ss58_address="5Validator")
    return make_context(
        executor=node.executor, runner=runner, miner_hotkey="5Account",
        config=build_context_config(validator_keypair=keypair, executor_root=node.executor.root_dir, **config),
        state=build_state(),
    )


async def _until(path: Path, seconds: float = 10.0) -> bool:
    for _ in range(int(seconds * 20)):
        if path.exists() and path.read_text():
            return True
        await asyncio.sleep(0.05)
    return False


def _attacker_ran(node) -> bool:
    return (node.tmp / "ATTACKER_EXECUTED").exists() or (node.tmp / "INTERPRETER_EXECUTED").exists()


async def test_gpu_monitor_cannot_execute_script_from_executor_root_dir(node):
    runner = LocalRunner(node.tmp)
    result = await StartGPUMonitorCheck().run(_ctx(node, runner))
    assert result.passed
    assert await _until(node.tmp / "VALIDATOR_MONITOR")  # only the validator's payload ran
    assert not (node.tmp / "ATTACKER_EXECUTED").exists()
    assert not any(node.executor.root_dir in cmd for cmd in runner.commands)


async def test_gpu_monitor_cannot_execute_executor_supplied_interpreter(node):
    runner = LocalRunner(node.tmp)
    await StartGPUMonitorCheck().run(_ctx(node, runner))
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    assert not (node.tmp / "INTERPRETER_EXECUTED").exists()
    assert not any(node.executor.python_path in cmd or node.executor.root_dir in cmd for cmd in runner.commands)


async def test_missing_identity_does_not_enable_executor_selected_code(node):
    checks = [type(check) for check in PipelineFactory.build_checks()]
    assert checks.index(NodeAccountCheck) < checks.index(StartGPUMonitorCheck)
    runner = LocalRunner(node.tmp)
    ctx = _ctx(node, runner)

    account = await NodeAccountCheck().run(ctx)
    assert account.passed and account.event.reason_code == "NODE_ACCOUNT_SKIPPED"  # compatibility kept
    await StartGPUMonitorCheck().run(ctx)
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    assert not _attacker_ran(node)


async def test_gpu_monitor_uses_trusted_payload_when_other_uploads_are_skipped(node):
    runner = LocalRunner(node.tmp)
    ctx = _ctx(node, runner, machine_scrape_source="print('scrape')")

    upload = await UploadFilesCheck().run(ctx)
    assert upload.event.reason_code == "UPLOAD_SKIPPED"
    assert (await StartGPUMonitorCheck().run(ctx)).passed
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    assert not _attacker_ran(node)


@pytest.mark.parametrize("run_dir", [None, ".verifyx-retry-0123456789abcdef"])
async def test_verifyx_does_not_execute_script_from_executor_root_dir(node, run_dir):
    command = VerifyXValidationService()._verifyx_command(
        node.executor, SimpleNamespace(seed=7, cipher_text="ab12"), "lib", run_dir=run_dir
    )
    done = subprocess.run(["sh", "-c", command], cwd=node.tmp, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert (node.tmp / "VALIDATOR_VERIFYX").read_text() == "--seed 7 --cipher_text ab12\n"
    assert not _attacker_ran(node)
    assert not list((node.tmp / "service").iterdir())  # the script goes with the run
    if run_dir:
        assert not (node.tmp / run_dir).exists()


async def test_inspector_does_not_execute_script_from_executor_root_dir(node):
    service = InspectorValidationService.__new__(InspectorValidationService)
    command = service._interactive_command(node.executor)
    done = subprocess.run(["sh", "-c", command], cwd=node.tmp, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert (node.tmp / "VALIDATOR_INSPECTOR").read_text() == "--interactive\n"
    assert not _attacker_ran(node)
    assert not list((node.tmp / "service").iterdir())


async def test_honest_gpu_monitor_still_starts(node):
    runner = LocalRunner(node.tmp)
    ctx = _ctx(node, runner)
    started = await StartGPUMonitorCheck().run(ctx)
    assert started.passed and started.event.reason_code == "MONITOR_STARTED"
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    args = (node.tmp / "VALIDATOR_MONITOR").read_text().split()
    assert args[0::2] == [
        "--program_id", "--signature", "--executor_id", "--validator_hotkey", "--compute_rest_app_url"
    ]
    assert args[5:] == ["node-1", "--validator_hotkey", "5Validator", "--compute_rest_app_url", "http://validator"]
    # the running monitor is found: no second copy
    again = await StartGPUMonitorCheck().run(ctx)
    assert again.passed and again.event.reason_code == "MONITOR_RUNNING"
    assert len((node.tmp / "VALIDATOR_MONITOR").read_text().splitlines()) == 1


def test_every_payload_is_found():
    # in the repository the executor sources; in the image the copies the Dockerfile takes from them
    for name in executor_payload.GPU_MONITOR + executor_payload.INSPECTOR + executor_payload.VERIFYX:
        assert (executor_payload.PAYLOAD_DIR / name).read_bytes() == (EXECUTOR_SOURCES / name).read_bytes(), name


def test_scripts_are_written_where_the_disk_reserve_holds():
    # a renter filling the host disk must not stop the validator writing its scripts
    reserve = (EXECUTOR_SOURCES.parent / "setup_disk_reserve.sh").read_text()
    assert f'workdir=$MNT/app-work" {executor_payload.SERVICE_DIR_PARENT}\n' in reserve
