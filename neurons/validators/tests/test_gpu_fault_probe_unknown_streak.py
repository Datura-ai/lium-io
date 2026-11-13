import logging

import pytest
from neurons.validators.src.services.task.checks import gpu_fault_probe as module
from neurons.validators.src.services.task.checks.gpu_fault_probe import (
    BUSY_VRAM_ERROR_PREFIX,
    UNKNOWN_STREAK_LIMIT,
    GpuFaultProbeCheck,
    probe_source,
)
from neurons.validators.src.services.task.messages import GpuFaultProbeMessages as Msg

from tests.test_gpu_fault_probe_check import FakeRunner, make_ctx, probe_gate, probe_stdout

NO_VERDICT = dict(stdout="", exit_code=-1, error_type="timeout")
OK = dict(stdout=probe_stdout("ok"))
BUSY_ERROR = (
    f"{BUSY_VRAM_ERROR_PREFIX}: 227 MB free, 1024 MB reserved, the smallest working set is 16 MB"
)
# a split node's rented card (aa801f3e, 31 Oct-3 Nov): the probe could not get memory on it, the others are fine
BUSY = dict(
    stdout=probe_stdout(
        "error",
        error=f"gpu 0: {BUSY_ERROR}",
        devices=[
            {"index": 0, "status": "error", "error": BUSY_ERROR},
            {"index": 1, "status": "ok"},
        ],
    )
)
BUSY_AND_HUNG = dict(
    stdout=probe_stdout(
        "error",
        error="gpu 0: ...; gpu 1: hung in kernels: no result after 39s",
        devices=[
            {"index": 0, "status": "error", "error": BUSY_ERROR},
            {"index": 1, "status": "error", "error": "hung in kernels: no result after 39s"},
        ],
    )
)


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
            context_factory, [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT + 1)
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
async def test_under_the_streak_switch_a_no_verdict_streak_fails_like_a_fault(context_factory):
    results = await _cycles(
        context_factory,
        [NO_VERDICT] * UNKNOWN_STREAK_LIMIT,
        enforce_unknown_streak=True,
    )

    assert [result.passed for result in results] == [True] * (UNKNOWN_STREAK_LIMIT - 1) + [False]
    assert results[-1].event.reason_code == Msg.PROBE_FAILED.reason
    assert "no verdict" in results[-1].event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_a_card_too_busy_to_probe_does_not_count_toward_the_streak(context_factory, caplog):
    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        results = await _cycles(
            context_factory,
            [BUSY] * (UNKNOWN_STREAK_LIMIT + 1),
            enforce_unknown_streak=True,
        )

    assert all(result.passed for result in results)
    assert {result.event.reason_code for result in results} == {Msg.UNKNOWN.reason}
    assert _streak_logs(caplog) == 0


@pytest.mark.asyncio
async def test_a_busy_cycle_neither_counts_nor_ends_a_streak(context_factory):
    results = await _cycles(
        context_factory,
        [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT - 1) + [BUSY, NO_VERDICT],
        enforce_unknown_streak=True,
    )

    assert [result.passed for result in results] == [True] * UNKNOWN_STREAK_LIMIT + [False]


@pytest.mark.asyncio
async def test_a_busy_card_beside_a_silent_one_still_counts(context_factory):
    results = await _cycles(
        context_factory,
        [BUSY_AND_HUNG] * UNKNOWN_STREAK_LIMIT,
        enforce_unknown_streak=True,
    )

    assert results[-1].passed is False


def test_the_busy_prefix_is_the_probes_own_setup_error():
    assert f'f"{BUSY_VRAM_ERROR_PREFIX}: ' in probe_source()


def test_unknown_streak_enforcement_is_off_by_default():
    from core.config import settings

    fields = type(settings).model_fields
    assert fields["GPU_FAULT_PROBE_UNKNOWN_STREAK_ENFORCEMENT_ENABLED"].default is False
