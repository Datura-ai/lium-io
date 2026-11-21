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
