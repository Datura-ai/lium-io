from unittest.mock import AsyncMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.const import FILLER_CONTAINER_PREFIX
from neurons.validators.src.services.task.checks import gpu_fault_probe
from neurons.validators.src.services.task.checks.rented_machine import (
    PodRunningAndAuthorizedKeys,
    TenantEnforcementCheck,
)
from neurons.validators.src.services.task.checks.gpu_fault_probe import (
    BUSY_VRAM_ERROR_PREFIX,
    UNKNOWN_STREAK_LIMIT,
    GpuFaultProbeCheck,
    probe_source,
)
from neurons.validators.src.services.task.messages import GpuFaultProbeMessages as Msg

from tests.helpers import default_executor
from tests.test_gpu_fault_probe_check import FakeRunner, make_ctx, probe_stdout
from tests.test_gpu_usage_check import _run_the_renter_ending_mid_run, _scored_ctx, _snapshot_with_pod

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


async def _cycles(context_factory, answers: list[dict]) -> list:
    results = []
    for answer in answers:
        answer = dict(answer)
        ctx = make_ctx(context_factory, FakeRunner(answer.pop("stdout"), **answer))
        results.append(await GpuFaultProbeCheck().run(ctx))
    return results


@pytest.mark.asyncio
async def test_a_verdict_ends_the_streak(context_factory):
    answers = (
        [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT - 1) + [OK] + [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT - 1)
    )

    results = await _cycles(context_factory, answers)

    assert all(result.passed for result in results)


@pytest.mark.asyncio
async def test_a_no_verdict_streak_fails_like_a_fault(context_factory):
    results = await _cycles(context_factory, [NO_VERDICT] * UNKNOWN_STREAK_LIMIT)

    assert [result.passed for result in results] == [True] * (UNKNOWN_STREAK_LIMIT - 1) + [False]
    assert results[-1].event.reason_code == Msg.PROBE_FAILED.reason
    assert "no verdict" in results[-1].event.what_we_saw["error"]


@pytest.mark.asyncio
async def test_a_card_too_busy_to_probe_does_not_count_toward_the_streak(context_factory):
    results = await _cycles(context_factory, [BUSY] * (UNKNOWN_STREAK_LIMIT + 1))

    assert all(result.passed for result in results)
    assert {result.event.reason_code for result in results} == {Msg.UNKNOWN.reason}


@pytest.mark.asyncio
async def test_a_busy_cycle_neither_counts_nor_ends_a_streak(context_factory):
    results = await _cycles(
        context_factory,
        [NO_VERDICT] * (UNKNOWN_STREAK_LIMIT - 1) + [BUSY, NO_VERDICT],
    )

    assert [result.passed for result in results] == [True] * UNKNOWN_STREAK_LIMIT + [False]


@pytest.mark.asyncio
async def test_a_busy_card_beside_a_silent_one_still_counts(context_factory):
    results = await _cycles(context_factory, [BUSY_AND_HUNG] * UNKNOWN_STREAK_LIMIT)

    assert results[-1].passed is False


def test_the_busy_prefix_is_the_probes_own_setup_error():
    assert f'f"{BUSY_VRAM_ERROR_PREFIX}: ' in probe_source()



FILLER = "filler"


async def _cycles_with_fillers(context_factory, answers: list) -> list:
    """FILLER is a cycle a filler holds the node: the probe is skipped."""
    results = []
    for answer in answers:
        if answer == FILLER:
            rented_data = RentedExecutorsResponse(
                executors={},
                banned_guids=[],
                filler_containers_by_executor={default_executor().uuid: f"{FILLER_CONTAINER_PREFIX}active"},
            )
            ctx = make_ctx(context_factory, FakeRunner(probe_stdout("ok")), rented_data=rented_data)
        else:
            answer = dict(answer)
            ctx = make_ctx(context_factory, FakeRunner(answer.pop("stdout"), **answer))
        results.append(await GpuFaultProbeCheck().run(ctx))
    return results


@pytest.mark.asyncio
async def test_a_filler_cycle_between_no_verdicts_ends_the_streak(context_factory):
    # 7b3527db, 14-15 Nov: a healthy card hung once on the first idle cycle after each rental, a filler between each
    answers = [NO_VERDICT, FILLER] * UNKNOWN_STREAK_LIMIT

    results = await _cycles_with_fillers(context_factory, answers)

    assert all(result.passed for result in results)


async def _rented_cycle_with_a_running_pod(ctx):
    rented_machine = "neurons.validators.src.services.task.checks.rented_machine"
    with (
        patch(
            f"{rented_machine}._check_pod_running_and_read_authorized_keys",
            AsyncMock(return_value=PodRunningAndAuthorizedKeys(running=True, authorized_keys=[])),
        ),
        patch(f"{rented_machine}._restore_missing_volume_mount", AsyncMock(return_value=True)),
    ):
        return await TenantEnforcementCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_rented_cycle_ends_the_streak(context_factory):
    ctx = _scored_ctx(context_factory, rented_data=_snapshot_with_pod(last_verifyx_download_speed=800.0))
    key = (ctx.miner_hotkey, ctx.executor.uuid)
    gpu_fault_probe._unknown_streak[key] = UNKNOWN_STREAK_LIMIT - 1

    result = await _rented_cycle_with_a_running_pod(ctx)

    assert result.halt is True and result.updates["rented"] is True
    assert key not in gpu_fault_probe._unknown_streak


@pytest.mark.asyncio
async def test_a_stale_pod_run_goes_on_as_idle_and_keeps_the_streak(context_factory):
    # every listed pod stale: the run continues unrented and reaches the probe, so it is a probe cycle
    ctx = _scored_ctx(context_factory, rented_data=_snapshot_with_pod(last_verifyx_download_speed=800.0))
    key = (ctx.miner_hotkey, ctx.executor.uuid)
    gpu_fault_probe._unknown_streak[key] = UNKNOWN_STREAK_LIMIT - 1

    await _run_the_renter_ending_mid_run(ctx)

    assert gpu_fault_probe._unknown_streak[key] == UNKNOWN_STREAK_LIMIT - 1


@pytest.mark.asyncio
async def test_the_streak_has_its_own_advice_not_reseat_the_card(context_factory):
    results = await _cycles(context_factory, [NO_VERDICT] * UNKNOWN_STREAK_LIMIT)

    remediation = results[-1].event.remediation
    assert "libcuda" in remediation
    assert remediation != Msg.PROBE_FAILED.remediation
