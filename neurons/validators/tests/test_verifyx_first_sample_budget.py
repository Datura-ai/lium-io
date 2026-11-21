"""DAH-5933: VerifyX's first sample gets only the task time the checks after it can spare (its 600 s cap once
starved the rental check, ad2382f9). Too little time, or a cut by that budget, is VERIFYX_NO_TASK_TIME: the
validator's budget, not the node, decided it. An express recheck's earlier deadline counts as the task's."""

import asyncio
import time

import pytest

from neurons.validators.src.services.task.checks import task_time
from neurons.validators.src.services.task.checks import verifyx as verifyx_check
from neurons.validators.src.services.task.checks.verifyx import (
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
    _left(monkeypatch, VERIFYX_LATER_CHECKS_RESERVE_SECONDS + MIN_COLD_RETRY_SECONDS - 1)
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
@pytest.mark.parametrize("probe_due,budget", [(False, 260), (True, None)], ids=["probe-done", "probe-due"])
async def test_the_reserve_is_what_can_still_run_after_verifyx(context_factory, monkeypatch, probe_due, budget):
    # review: 320 s left after the lanes; the rental check already ran, so only a still-due rental probe keeps its share
    _left(monkeypatch, 320)

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
        # the bounded first sample has no earlier result to stand: its deadline falls back to the executor's Python
        assert service.calls[0]["python_watchdog"] is True


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
