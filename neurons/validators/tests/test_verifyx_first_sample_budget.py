"""DAH-5933: VerifyX's first sample gets only the task time the checks after it can spare (its 600 s cap once
starved the rental check, ad2382f9). Too little time, or a cut by that budget, is VERIFYX_NO_TASK_TIME: the
validator's budget, not the node, decided it. An express recheck's earlier deadline counts as the task's."""

import asyncio
import time

import pytest

from neurons.validators.src.services.task.checks import task_time
from neurons.validators.src.services.task.checks import verifyx as verifyx_check
from neurons.validators.src.services.task.checks.verifyx import (
    COLD_RETRY_MARGIN_SECONDS,
    MIN_COLD_RETRY_SECONDS,
    VERIFYX_LATER_CHECKS_RESERVE_SECONDS,
    VerifyXCheck,
)
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.verifyx_validation_service import VERIFYX_COMMAND_TIMEOUT_SECONDS

from tests.test_verifyx_cold_sample_retry import _ctx, _known, _probe


class _Probe:
    def __init__(self, response, run_s: float = 0.0):
        self.response = response
        self.run_s = run_s
        self.calls = []
        self.kills = []

    async def validate_verifyx_and_process_job(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.run_s)
        return self.response

    async def kill_remote_verifyx(self, shell, run_dir, default_extra):
        self.kills.append(run_dir)
        return True


def _left(monkeypatch, seconds):
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: seconds)


@pytest.mark.asyncio
async def test_too_little_time_for_verifyx_is_no_task_time_and_no_run(context_factory, monkeypatch):
    # under the first sample's minimum even with only the scoring margin kept
    _left(monkeypatch, COLD_RETRY_MARGIN_SECONDS + MIN_COLD_RETRY_SECONDS - 1)
    service = _Probe(_probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert result.passed is False
    assert result.event.reason_code == Msg.NO_TASK_TIME.reason
    assert service.calls == []


@pytest.mark.asyncio
async def test_a_full_budget_runs_verifyx_as_before(context_factory, monkeypatch):
    _left(monkeypatch, VERIFYX_LATER_CHECKS_RESERVE_SECONDS + VERIFYX_COMMAND_TIMEOUT_SECONDS)
    service = _Probe(_probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert result.passed is True
    assert "run_dir" not in service.calls[0]
    assert service.kills == []


@pytest.mark.asyncio
async def test_a_short_budget_cuts_verifyx_at_it_and_judges_a_finished_run(context_factory, monkeypatch):
    budget = MIN_COLD_RETRY_SECONDS + 10.0
    _left(monkeypatch, VERIFYX_LATER_CHECKS_RESERVE_SECONDS + budget)
    timeouts = []
    wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        timeouts.append(timeout)
        return await wait_for(aw, timeout)

    monkeypatch.setattr(verifyx_check.asyncio, "wait_for", spy)
    service = _Probe(_probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert result.passed is True
    assert timeouts == [pytest.approx(budget)]
    assert service.calls[0]["run_dir"].startswith(".verifyx-")
    assert service.kills == []


@pytest.mark.asyncio
async def test_a_run_cut_by_the_budget_is_no_task_time_and_killed_on_the_host(context_factory, monkeypatch):
    monkeypatch.setattr(verifyx_check, "MIN_COLD_RETRY_SECONDS", 0.05)
    _left(monkeypatch, VERIFYX_LATER_CHECKS_RESERVE_SECONDS + 0.05)
    service = _Probe(_probe(760.0), run_s=5.0)

    result = await asyncio.wait_for(VerifyXCheck().run(_ctx(context_factory, service, _known(500.0))), 2.0)

    assert result.passed is False
    assert result.event.reason_code == Msg.NO_TASK_TIME.reason
    assert service.kills == [service.calls[0]["run_dir"]]


def test_an_express_deadline_shortens_the_task_time(context_factory):
    ctx = context_factory()
    assert task_time.task_seconds_left(ctx) > 600
    ctx = ctx.model_copy(update={"deadline_monotonic": time.monotonic() + 110})

    assert 109 <= task_time.task_seconds_left(ctx) <= 110


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_due,left,budget",
    [(False, 320, 260), (True, 320, 260), (True, 359, 299), (True, 360, 300), (True, 390, 240), (False, 299, None)],
    ids=["probe-done", "probe-due-short", "probe-due-359", "probe-due-360", "probe-due-reserved", "no-time"],
)
async def test_the_reserve_is_what_can_still_run_after_verifyx(context_factory, monkeypatch, probe_due, left, budget):
    # review: the rental check already ran, so only a still-due rental probe keeps its share, and only while the
    # mandatory first sample still gets its minimum (F-15/F-23: at 360 s the reserve took the sample to zero attempts)
    _left(monkeypatch, left)

    async def due(ctx):
        return probe_due

    monkeypatch.setattr(verifyx_check, "rental_probe_due", due)
    timeouts = []
    wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        timeouts.append(timeout)
        return await wait_for(aw, timeout)

    monkeypatch.setattr(verifyx_check.asyncio, "wait_for", spy)
    service = _Probe(_probe(760.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    if budget is None:
        assert result.event.reason_code == Msg.NO_TASK_TIME.reason
        assert service.calls == []
    else:
        assert result.passed is True
        assert timeouts == [pytest.approx(budget)]
        # the bounded first sample carries its own deadline to the host (review F-24)
        assert service.calls[0]["deadline_monotonic"] == pytest.approx(time.monotonic() + budget, abs=5)


def test_a_bounded_run_on_a_host_without_timeout_k_runs_under_the_python_watchdog(tmp_path):
    import os
    import subprocess
    import sys
    from types import SimpleNamespace

    from neurons.validators.src.services.verifyx_validation_service import VerifyXValidationService

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "verifyx_executor.py").write_text("print('sample done')\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # busybox-style: `timeout -k` is refused
    (fake_bin / "timeout").write_text("#!/bin/sh\nexit 1\n")
    (fake_bin / "timeout").chmod(0o755)
    executor = SimpleNamespace(python_path=sys.executable, root_dir=str(tmp_path))
    challenge = SimpleNamespace(seed=1, cipher_text="x")
    service = VerifyXValidationService.__new__(VerifyXValidationService)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    def run(**kwargs):
        command = service._verifyx_command(executor, challenge, "lib", run_dir=".verifyx-t", **kwargs)
        return subprocess.run(["sh", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)

    bounded = run(python_watchdog=True)
    assert (bounded.returncode, bounded.stdout.strip()) == (0, "sample done")
    assert not (tmp_path / ".verifyx-t").exists()
    # the cold retry keeps its skip there: its first sample already stands
    retry = run()
    assert retry.returncode == 1 and "no timeout -k" in retry.stderr


def _preflight_passes(
    monkeypatch, service, module, *, preflight_s: float = 0.0, exit_status: int = 0, stderr: str | None = None
):
    from types import SimpleNamespace

    async def arch(shell):
        await asyncio.sleep(preflight_s)
        return "x86_64"

    async def checksum(shell, lib_name):
        return "same"

    monkeypatch.setattr(module, "executor_arch", arch)
    monkeypatch.setattr(module, "sha256_from_executor", checksum)
    monkeypatch.setattr(service, "gated_lib_name", lambda: "lib")
    monkeypatch.setattr(service, "lib_sha256", lambda source: "same")
    monkeypatch.setattr(service, "prepare_verifyx_challenge", lambda *a, **k: SimpleNamespace(log_extra={}))
    monkeypatch.setattr(service, "evaluate_verifyx_capture", lambda challenge, capture, extra: capture)
    commands = []

    async def ssh(shell, command):
        commands.append(command)
        return module.SSHCapture(stdout="captured", stderr=stderr, exit_status=exit_status)

    monkeypatch.setattr(service, "_run_ssh_command", ssh)
    monkeypatch.setattr(service, "_verifyx_command", lambda *a: f"timeout -k 10 {a[5]} run &")
    return commands


@pytest.mark.asyncio
async def test_the_remote_deadline_is_what_the_budget_leaves_after_the_preflight(monkeypatch):
    # review F-24: the host's watchdog ends with the validator's cut, not 600 s after the launch
    from neurons.validators.src.services import verifyx_validation_service as module

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    commands = _preflight_passes(monkeypatch, service, module)

    result = await service.validate_verifyx_and_process_job(
        shell=None, executor_info=None, default_extra={}, machine_spec={}, run_dir=".verifyx-t",
        deadline_monotonic=time.monotonic() + 250.5,
    )

    assert result.stdout == "captured"
    assert commands == ["timeout -k 10 250 run &"] or commands == ["timeout -k 10 249 run &"]


@pytest.mark.asyncio
async def test_no_launch_once_the_preflight_spent_the_budget(monkeypatch):
    from neurons.validators.src.services import verifyx_validation_service as module

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    commands = _preflight_passes(monkeypatch, service, module, preflight_s=0.05)

    result = await service.validate_verifyx_and_process_job(
        shell=None, executor_info=None, default_extra={}, machine_spec={}, run_dir=".verifyx-t",
        deadline_monotonic=time.monotonic() + 0.01,
    )

    assert result is None
    assert commands == []


def test_both_remote_watchdogs_take_the_given_seconds():
    from types import SimpleNamespace

    from neurons.validators.src.services.verifyx_validation_service import VerifyXValidationService

    service = VerifyXValidationService.__new__(VerifyXValidationService)
    executor = SimpleNamespace(python_path="/usr/bin/python3", root_dir="/root/app")
    challenge = SimpleNamespace(seed=1, cipher_text="x")

    bounded = service._verifyx_command(executor, challenge, "lib", ".verifyx-t", True, 137)
    retry = service._verifyx_command(executor, challenge, "lib", ".verifyx-t", False, 137)

    assert " 137 /usr/bin/python3 /root/app/src/verifyx_executor.py" in bounded
    assert "timeout -k 10 137 " in retry
    assert " 600 " not in bounded + retry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bounded,exit_status,marked,cut",
    [
        (True, 124, True, True),  # GNU timeout at the deadline
        (True, 143, True, True),  # busybox timeout: the child's status after its TERM (F-29)
        (True, 124, False, False),  # the child exited 124 itself before the deadline: judged (F-29)
        (True, 143, False, False),  # the child exited 143 itself before the deadline: judged (F-29)
        (False, 124, True, False),  # unbounded: judged as before
    ],
    ids=["gnu-deadline", "busybox-deadline", "early-child-124", "early-child-143", "unbounded"],
)
async def test_only_the_hosts_deadline_record_makes_a_run_the_cut(monkeypatch, bounded, exit_status, marked, cut):
    from neurons.validators.src.services import verifyx_validation_service as module

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    stderr = f"Terminated\n{module.VERIFYX_DEADLINE_MARKER}\n" if marked else "boom\n"
    _preflight_passes(monkeypatch, service, module, exit_status=exit_status, stderr=stderr)

    result = await service.validate_verifyx_and_process_job(
        shell=None, executor_info=None, default_extra={}, machine_spec={}, run_dir=".verifyx-t",
        deadline_monotonic=time.monotonic() + 250 if bounded else None,
    )

    # the cut is reported as no task time by the caller; anything else is judged
    assert (result is None) is cut


_BUSYBOX_TIMEOUT = """#!/bin/sh
[ "$1" = "-k" ] && shift 2
s=$1; shift
"$@" & p=$!
( sleep "$s"; kill -TERM $p 2>/dev/null ) &
wait $p
"""


@pytest.mark.parametrize("timeout_kind", ["busybox", "gnu", "none"])
@pytest.mark.parametrize(
    "child,marked",
    [
        ("import time; time.sleep(30)", True),  # stopped at the deadline
        ("import sys; sys.exit(124)", False),  # the child's own 124 (F-29)
        ("import sys; sys.exit(143)", False),  # the child's own 143 (F-29)
        ("import time; time.sleep(1.5)", False),  # done 0.5 s before the deadline: not cut, whatever the clock (F-32)
    ],
    ids=["stopped", "own-124", "own-143", "just-in-time"],
)
def test_only_the_watchdogs_stop_writes_the_deadline_marker(tmp_path, timeout_kind, child, marked):
    # F-29/F-32: the marker comes from the one actor that stops the run, whatever `timeout` the host has
    import os
    import shutil
    import subprocess
    import sys
    from types import SimpleNamespace

    from neurons.validators.src.services.verifyx_validation_service import (
        VERIFYX_DEADLINE_MARKER,
        VerifyXValidationService,
    )

    if timeout_kind == "gnu" and shutil.which("timeout") is None:
        pytest.skip("no timeout on this host")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "verifyx_executor.py").write_text(child + "\n")
    env = dict(os.environ)
    if timeout_kind != "gnu":
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        # none: a `timeout` that refuses everything
        (fake_bin / "timeout").write_text(_BUSYBOX_TIMEOUT if timeout_kind == "busybox" else "#!/bin/sh\nexit 1\n")
        (fake_bin / "timeout").chmod(0o755)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
    executor = SimpleNamespace(python_path=sys.executable, root_dir=str(tmp_path))
    challenge = SimpleNamespace(seed=1, cipher_text="x")
    service = VerifyXValidationService.__new__(VerifyXValidationService)

    command = service._verifyx_command(executor, challenge, "lib", ".verifyx-t", True, 2)
    done = subprocess.run(["sh", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)

    assert (VERIFYX_DEADLINE_MARKER in done.stderr) is marked
    assert not (tmp_path / ".verifyx-t").exists()


@pytest.mark.asyncio
async def test_no_probe_lookup_when_its_answer_cannot_change_the_reserve(context_factory, monkeypatch):
    # review at b49619d: at 305 s only the 60 s tail can apply, so a slow Redis read must not eat the sample's time
    _left(monkeypatch, 305)
    lookups = []

    async def due(ctx):
        lookups.append(ctx)
        return True

    monkeypatch.setattr(verifyx_check, "rental_probe_due", due)
    timeouts = []
    wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        timeouts.append(timeout)
        return await wait_for(aw, timeout)

    monkeypatch.setattr(verifyx_check.asyncio, "wait_for", spy)
    service = _Probe(_probe(760.0))

    await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert lookups == []
    assert timeouts == [pytest.approx(245)]


@pytest.mark.asyncio
async def test_the_budget_reads_the_clock_after_the_probe_lookup(context_factory, monkeypatch):
    # review at e79ac9c: 660.5 s before a 1 s lookup that says not due leaves 599.5 s: bounded, never the full run
    clock = {"left": 660.5}
    monkeypatch.setattr(verifyx_check, "task_seconds_left", lambda ctx: clock["left"])

    async def due(ctx):
        clock["left"] -= 1
        return False

    monkeypatch.setattr(verifyx_check, "rental_probe_due", due)
    timeouts = []
    wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        timeouts.append(timeout)
        return await wait_for(aw, timeout)

    monkeypatch.setattr(verifyx_check.asyncio, "wait_for", spy)
    service = _Probe(_probe(760.0))

    await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert timeouts == [pytest.approx(599.5)]
    assert "deadline_monotonic" in service.calls[0]
