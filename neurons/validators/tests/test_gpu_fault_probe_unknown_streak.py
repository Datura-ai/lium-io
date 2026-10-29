import logging

import pytest
from neurons.validators.src.services.task.checks import gpu_fault_probe as module
from neurons.validators.src.services.task.checks.gpu_fault_probe import (
    UNKNOWN_STREAK_LIMIT,
    GpuFaultProbeCheck,
)
from neurons.validators.src.services.task.messages import GpuFaultProbeMessages as Msg

from tests.test_gpu_fault_probe_check import FakeRunner, make_ctx, probe_gate, probe_stdout

NO_VERDICT = dict(stdout="", exit_code=-1, error_type="timeout")
OK = dict(stdout=probe_stdout("ok"))


async def _cycles(context_factory, answers: list[dict], **gate) -> list:
    results = []
    for answer in answers:
        answer = dict(answer)
        ctx = make_ctx(context_factory, FakeRunner(answer.pop("stdout"), **answer))
        with probe_gate(**gate):
            results.append(await GpuFaultProbeCheck().run(ctx))
    return results


def _streak_logs(caplog) -> int:
    return sum("GPU_FAULT_PROBE_UNKNOWN_STREAK" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_no_verdict_streak_is_logged_from_the_limit_and_still_passes(
    context_factory, caplog
):
    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        results = await _cycles(
            context_factory, [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT + 1), enforce=True
        )

    assert all(result.passed for result in results)
    assert {result.event.reason_code for result in results} == {Msg.UNKNOWN.reason}
    assert _streak_logs(caplog) == 2
    assert results[-1].event.what_we_saw["unknown_streak"] == UNKNOWN_STREAK_LIMIT + 1


@pytest.mark.asyncio
async def test_a_verdict_ends_the_streak(context_factory, caplog):
    answers = (
        [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT - 1) + [OK] + [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT - 1)
    )

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        await _cycles(context_factory, answers)

    assert _streak_logs(caplog) == 0


@pytest.mark.asyncio
async def test_under_both_switches_a_no_verdict_streak_fails_like_a_fault(context_factory):
    results = await _cycles(
        context_factory,
        [NO_VERDICT] * UNKNOWN_STREAK_LIMIT,
        enforce=True,
        enforce_unknown_streak=True,
    )

    assert [result.passed for result in results] == [True] * (UNKNOWN_STREAK_LIMIT - 1) + [False]
    assert results[-1].event.reason_code == Msg.PROBE_FAILED.reason
    assert "no verdict" in results[-1].event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_the_streak_switch_alone_stays_in_shadow(context_factory):
    results = await _cycles(
        context_factory,
        [NO_VERDICT] * UNKNOWN_STREAK_LIMIT,
        enforce=False,
        enforce_unknown_streak=True,
    )

    assert all(result.passed for result in results)
    assert results[-1].event.reason_code == Msg.PROBE_FAILED.reason


def test_unknown_streak_enforcement_is_off_by_default():
    from core.config import settings

    fields = type(settings).model_fields
    assert fields["GPU_FAULT_PROBE_UNKNOWN_STREAK_ENFORCEMENT_ENABLED"].default is False
