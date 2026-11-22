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
async def test_a_full_budget_bounds_the_whole_call_at_the_time_left(context_factory, monkeypatch):
    # the preflight runs inside the budget too, so a full budget is a deadline as well, not an unbounded 600 s run
    _left(monkeypatch, COLD_RETRY_MARGIN_SECONDS + VERIFYX_COMMAND_TIMEOUT_SECONDS + 100)
    service = _Probe(_probe(760.0))

    started = time.monotonic()
    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert result.passed is True
    assert service.calls[0]["run_dir"].startswith(".verifyx-")
    deadline_in = service.calls[0]["deadline_monotonic"] - started
    assert VERIFYX_COMMAND_TIMEOUT_SECONDS + 100 <= deadline_in <= VERIFYX_COMMAND_TIMEOUT_SECONDS + 101
    assert service.kills == []


@pytest.mark.asyncio
async def test_a_full_budget_run_cut_by_the_time_left_is_no_task_time_and_killed(context_factory, monkeypatch):
    _left(monkeypatch, COLD_RETRY_MARGIN_SECONDS + VERIFYX_COMMAND_TIMEOUT_SECONDS)
    monkeypatch.setattr(verifyx_check, "COLD_RETRY_MARGIN_SECONDS", COLD_RETRY_MARGIN_SECONDS + VERIFYX_COMMAND_TIMEOUT_SECONDS - 0.05)
    monkeypatch.setattr(verifyx_check, "MIN_COLD_RETRY_SECONDS", 0.01)
    service = _Probe(_probe(760.0), run_s=1.0)

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert result.event.reason_code == Msg.NO_TASK_TIME.reason
    assert service.kills == [service.calls[0]["run_dir"]]


@pytest.mark.asyncio
async def test_a_short_budget_cuts_verifyx_at_it_and_judges_a_finished_run(context_factory, monkeypatch):
    budget = MIN_COLD_RETRY_SECONDS + 10.0
    _left(monkeypatch, COLD_RETRY_MARGIN_SECONDS + budget)
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
    _left(monkeypatch, COLD_RETRY_MARGIN_SECONDS + 0.1)
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
    "left,budget",
    [(320, 260), (390, 330), (600, 540), (299, None)],
    ids=["short", "390", "600", "no-time"],
)
async def test_the_first_sample_leaves_only_the_scoring_margin(context_factory, monkeypatch, left, budget):
    # review at 1749a1d: a rental-probe reserve below the probe's own 360 s never let it run and cut passing runs
    _left(monkeypatch, left)
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
    monkeypatch,
    service,
    module,
    *,
    preflight_s: float = 0.0,
    exit_status: int = 0,
    stderr: str | None = None,
    run_s: float = 0.0,
):
    from types import SimpleNamespace

    # the remote run takes `run_s` on the service's clock
    ran = [0.0]
    real = time.monotonic
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: real() + ran[0]))

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
        ran[0] += run_s
        return module.SSHCapture(stdout="captured", stderr=stderr, exit_status=exit_status)

    monkeypatch.setattr(service, "_run_ssh_command", ssh)
    monkeypatch.setattr(service, "_verifyx_command", lambda *a: f"timeout -k 10 {a[5]} run &")
    return commands


@pytest.mark.asyncio
async def test_the_hosts_watchdog_fires_after_the_validators_own_cut(monkeypatch):
    # review F-24: the host's watchdog follows the deadline, not 600 s after the launch. Review at ea74f37: it fires
    # only after the validator's own cut and that cut's kill, so a host clock running fast (0.04%) cannot stop an
    # honest run before the validator does, and the host's clock never decides a cut
    from neurons.validators.src.services import verifyx_validation_service as module

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    commands = _preflight_passes(monkeypatch, service, module)
    deadline_s = 250.8

    result = await service.validate_verifyx_and_process_job(
        shell=None, executor_info=None, default_extra={}, machine_spec={}, run_dir=".verifyx-t",
        deadline_monotonic=time.monotonic() + deadline_s,
    )

    watchdog_s = int(commands[0].split()[3])
    assert result.stdout == "captured"
    assert watchdog_s in (249 + module.VERIFYX_KILL_TIMEOUT_SECONDS, 250 + module.VERIFYX_KILL_TIMEOUT_SECONDS)
    assert watchdog_s / 1.0004 > deadline_s  # a host clock 0.04% fast still stops after the validator cut


@pytest.mark.asyncio
async def test_a_deadline_past_the_cap_still_runs_the_remote_for_the_cap(monkeypatch):
    # a full budget now passes its deadline too; the 600 s remote cap still applies after the preflight
    from neurons.validators.src.services import verifyx_validation_service as module

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    commands = _preflight_passes(monkeypatch, service, module)

    await service.validate_verifyx_and_process_job(
        shell=None, executor_info=None, default_extra={}, machine_spec={}, run_dir=".verifyx-t",
        deadline_monotonic=time.monotonic() + VERIFYX_COMMAND_TIMEOUT_SECONDS + 300,
    )

    assert commands == [f"timeout -k 10 {VERIFYX_COMMAND_TIMEOUT_SECONDS} run &"]


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
    "left,run_s,exit_status,marked",
    [
        (250, 249.92, 124, True),  # the host's watchdog early on its own clock: the validator's cut had not come
        (250, 10, 143, True),  # the marker printed long before the deadline: the host's word alone
        (250, 10, 124, False),  # the child exited 124 itself before the deadline (F-29)
        (None, 600, 124, True),  # unbounded
        (700, 600, 124, True),  # the cap, not the deadline, set the run
    ],
    ids=["host-watchdog-early", "early-marker", "early-child-124", "unbounded", "marker-at-cap"],
)
async def test_a_run_that_returns_is_judged_whatever_its_exit_status_or_marker(
    monkeypatch, left, run_s, exit_status, marked
):
    # review at ea74f37: only the caller's own wait, on the validator's clock, cuts a run (it returns None there and
    # kills the run); whatever comes back from the host is judged, never read as the cut
    from neurons.validators.src.services import verifyx_validation_service as module

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    stderr = f"Terminated\n{module.VERIFYX_DEADLINE_MARKER}\n" if marked else "boom\n"
    _preflight_passes(monkeypatch, service, module, exit_status=exit_status, stderr=stderr, run_s=run_s)

    result = await service.validate_verifyx_and_process_job(
        shell=None, executor_info=None, default_extra={}, machine_spec={}, run_dir=".verifyx-t",
        deadline_monotonic=None if left is None else time.monotonic() + left,
    )

    assert result is not None and result.exit_status == exit_status


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



def test_the_watchdog_stops_the_run_with_both_ssh_readers_gone(tmp_path):
    # review at 77549c7: with the session's readers closed the marker write fails (EPIPE); the kill must still land
    import subprocess
    import sys
    import time
    from types import SimpleNamespace

    from neurons.validators.src.services.verifyx_validation_service import VerifyXValidationService

    done = tmp_path / "done"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "verifyx_executor.py").write_text(f"import time; time.sleep(3); open({str(done)!r}, 'w')\n")
    executor = SimpleNamespace(python_path=sys.executable, root_dir=str(tmp_path))
    service = VerifyXValidationService.__new__(VerifyXValidationService)
    command = service._verifyx_command(executor, SimpleNamespace(seed=1, cipher_text="x"), "lib", ".verifyx-t", True, 1)

    run = subprocess.Popen(["sh", "-c", command], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    run.stdout.close()
    run.stderr.close()
    run.wait(timeout=30)
    time.sleep(3)

    assert not done.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "left,run_s",
    [(250, 10), (VERIFYX_COMMAND_TIMEOUT_SECONDS + 100, VERIFYX_COMMAND_TIMEOUT_SECONDS)],
    ids=["marker-before-the-deadline", "marker-at-the-cap"],
)
async def test_a_node_written_deadline_marker_never_turns_a_failing_check_into_a_carried_pass(
    context_factory, monkeypatch, left, run_s
):
    # the run's output is the node's: only the validator's own clock and budget may make a failing run a cut, which
    # the rollout would carry (NO_TASK_TIME_REASONS); a marker the validator's deadline did not cause is judged
    from neurons.validators.src.services import verifyx_validation_service as module
    from neurons.validators.src.services.task.messages import NO_TASK_TIME_REASONS

    service = module.VerifyXValidationService.__new__(module.VerifyXValidationService)
    _preflight_passes(
        monkeypatch, service, module, exit_status=1, stderr=f"{module.VERIFYX_DEADLINE_MARKER}\n", run_s=run_s
    )
    monkeypatch.setattr(
        service, "evaluate_verifyx_capture", lambda challenge, capture, extra: module.VerifyXResponse(error="rejected")
    )
    _left(monkeypatch, COLD_RETRY_MARGIN_SECONDS + left)

    result = await VerifyXCheck().run(_ctx(context_factory, service, _known(500.0)))

    assert result.passed is False
    assert result.event.reason_code not in NO_TASK_TIME_REASONS
