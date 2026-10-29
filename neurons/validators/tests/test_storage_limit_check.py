"""StorageLimitCheck: a node whose Docker cannot cap a container's disk (18 Oct 2026, serdkey node 4837fec9).

A renter filled the disk of a node without a per-container storage limit and took the host down. Off by default:
a warning only. With REQUIRE_STORAGE_LIMIT_SUPPORTED an unrented node is held from new rentals after one failed
probe and scores 0 after two in a row; a probe without a verdict neither counts nor resets (Mikhail, 29 Oct).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from neurons.validators.tests.helpers import build_state
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from services.task.availability import availability_errors
from services.task.checks.rented_machine import TenantEnforcementCheck
from services.task.checks.storage_limit import StorageLimitCheck
from services.task.messages import StorageLimitMessages as Msg
from services.task.pipeline_factory import PipelineFactory


async def run(context_factory, specs: dict, *, rented: bool = False):
    rented_data = None
    if rented:
        executor = RentedExecutor.model_validate(
            {
                "miner_hotkey": "hk",
                "executor_ip_address": "1.2.3.4",
                "executor_ip_port": "8080",
                "pods": [{"pod_id": "pod-1", "container_name": "c", "gpu_count": 1}],
            }
        )
        rented_data = RentedExecutorsResponse(executors={context_factory().executor.uuid: executor})
    ctx = context_factory(state=build_state(specs=specs, rented_data=rented_data))
    # two cycles with the same scrape: REQUIRED needs two conclusive failures in a row
    await StorageLimitCheck().run(ctx)
    return await StorageLimitCheck().run(ctx)


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setattr(
        "services.task.checks.storage_limit.settings.REQUIRE_STORAGE_LIMIT_SUPPORTED", True
    )


async def run_with(ctx, specs):
    # ContextState is frozen; a cycle's scrape refills the same specs dict
    ctx.state.specs.clear()
    ctx.state.specs.update(specs)
    return await StorageLimitCheck().run(ctx)


PASS = {"storage_limit_supported": True}
NO_LIMIT = {
    "storage_limit_supported": False,
    "storage_limit_scrape_error": "Storage limit is not supported: /var/lib/docker is on ext4, not xfs.",
}
UNKNOWN = {"storage_limit_supported": None, "storage_limit_scrape_error": "Test command timed out."}


def held(result) -> bool:
    """Hidden from new rentals with the score untouched."""
    return result.passed is True and bool(availability_errors([result.event])) and not result.updates


def zeroed(result) -> bool:
    return (
        result.passed is False
        and result.event.reason_code == Msg.REQUIRED.reason
        and result.updates["score"] == 0.0
        and bool(availability_errors([result.event]))
    )


def warned(result) -> bool:
    return result.passed is True and not availability_errors([result.event]) and not result.updates


def inconclusive_since(ctx, delta: timedelta) -> None:
    # backdate the node's first probe without a verdict in its stored verdict
    key = next(k for k in ctx.services.redis.store if k.startswith("storage_limit_verdict:"))
    state = json.loads(ctx.services.redis.store[key])
    state["inconclusive_since"] = (datetime.now(UTC) - delta).isoformat()
    ctx.services.redis.store[key] = json.dumps(state)


def new_ctx(context_factory, capable=True):
    ctx = context_factory(state=build_state(specs={}))
    return ctx, ({} if capable is None else {"storage_limit_capable": capable})


@pytest.mark.asyncio
async def test_a_node_with_the_limit_passes(context_factory, enforced):
    result = await run(context_factory, PASS)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_without_enforcement_a_node_without_the_limit_only_warns(context_factory):
    result = await run(context_factory, NO_LIMIT)

    assert warned(result)
    assert result.event.reason_code == Msg.MISSING.reason
    assert "ext4" in result.event.what_we_saw["detail"]


@pytest.mark.asyncio
async def test_the_first_failure_holds_and_the_second_in_a_row_zeros(context_factory, enforced):
    ctx, capable = new_ctx(context_factory)

    first = await run_with(ctx, {**NO_LIMIT, **capable})
    second = await run_with(ctx, {**NO_LIMIT, **capable})
    passed = await run_with(ctx, PASS)
    again = await run_with(ctx, {**NO_LIMIT, **capable})

    assert held(first)
    assert first.event.reason_code == Msg.MISSING.reason
    assert zeroed(second)
    assert passed.event.reason_code == Msg.OK.reason
    # a conclusive pass resets the count
    assert held(again)


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [False, True])
async def test_enforced_only_an_unrented_node_is_hidden_from_new_rentals(
    context_factory, enforced, rented
):
    result = await run(context_factory, {**NO_LIMIT, "storage_limit_capable": True}, rented=rented)

    codes = [error.reason_code for error in availability_errors([result.event])]
    assert codes == ([] if rented else ["STORAGE_LIMIT_REQUIRED"])
    if rented:
        assert warned(result)


@pytest.mark.parametrize(
    "build",
    [
        PipelineFactory.build_checks,
        PipelineFactory.build_fast_path_checks,
        PipelineFactory.build_dry_run_checks,
    ],
)
def test_the_check_runs_before_the_rented_halt_in_every_pipeline(build):
    ids = [check.check_id for check in build()]

    assert ids.index(StorageLimitCheck.check_id) < ids.index(TenantEnforcementCheck.check_id)


@pytest.mark.asyncio
async def test_enforced_a_single_disk_host_only_warns(context_factory, enforced):
    """DAH-1997 (Fish 21 Oct): required only of hosts that can build the limit."""
    result = await run(context_factory, {**NO_LIMIT, "storage_limit_capable": False})

    assert warned(result)
    assert result.event.reason_code == Msg.MISSING.reason


@pytest.mark.asyncio
async def test_an_unknown_capability_adds_no_consequence(context_factory, enforced):
    ctx, unknown = new_ctx(context_factory, capable=None)

    first = await run_with(ctx, {**NO_LIMIT, **unknown})
    second = await run_with(ctx, {**NO_LIMIT, **unknown})

    assert warned(first)
    assert warned(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("failures, check", [(1, held), (2, zeroed)])
async def test_an_unknown_capability_clears_no_existing_hold_or_zero(
    context_factory, enforced, failures, check
):
    ctx, capable = new_ctx(context_factory)
    for _ in range(failures):
        await run_with(ctx, {**NO_LIMIT, **capable})

    result = await run_with(ctx, NO_LIMIT)

    assert check(result)


@pytest.mark.asyncio
async def test_unknown_after_a_pass_keeps_the_pass_and_publishes_it(context_factory, enforced):
    ctx, capable = new_ctx(context_factory)
    await run_with(ctx, PASS)

    result = await run_with(ctx, {**UNKNOWN, **capable})

    assert warned(result)
    assert result.event.reason_code == Msg.OK.reason
    assert result.event.what_we_saw["reused_last_pass"] is True
    # the published specs keep the limit, so the backend never offers this host uncapped
    assert ctx.state.specs["storage_limit_supported"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failures, check", [(1, held), (2, zeroed)])
async def test_unknown_after_failures_keeps_what_they_caused(context_factory, enforced, failures, check):
    ctx, capable = new_ctx(context_factory)
    for _ in range(failures):
        await run_with(ctx, {**NO_LIMIT, **capable})

    results = [await run_with(ctx, {**UNKNOWN, **capable}) for _ in range(3)]
    after = await run_with(ctx, {**NO_LIMIT, **capable})

    # unknown never creates a zero, and never restores pay after two failures
    assert all(check(result) for result in results)
    assert ctx.state.specs["storage_limit_supported"] is False
    # and it neither counts nor resets: one failure, three unknowns, one failure is two in a row
    assert zeroed(after)


@pytest.mark.asyncio
@pytest.mark.parametrize("specs", [UNKNOWN, {}])  # no verdict, or an older scrape without the field
async def test_no_verdict_on_record_holds_without_a_zero(context_factory, enforced, specs):
    ctx, capable = new_ctx(context_factory)

    results = [await run_with(ctx, {**specs, **capable}) for _ in range(2)]

    assert all(held(result) for result in results)
    assert results[-1].event.reason_code == Msg.INCONCLUSIVE.reason
    # published as unknown, never as "no limit"
    assert ctx.state.specs["storage_limit_supported"] is None


@pytest.mark.asyncio
async def test_unenforced_no_verdict_is_published_as_unknown(context_factory):
    ctx, capable = new_ctx(context_factory)

    result = await run_with(ctx, {**UNKNOWN, **capable})

    assert warned(result)
    assert result.event.reason_code == Msg.MISSING.reason
    assert ctx.state.specs["storage_limit_supported"] is None


@pytest.mark.asyncio
async def test_unknown_for_over_a_day_after_a_pass_holds_without_a_score_change(
    context_factory, enforced
):
    ctx, capable = new_ctx(context_factory)
    await run_with(ctx, PASS)
    await run_with(ctx, {**UNKNOWN, **capable})
    inconclusive_since(ctx, timedelta(hours=23))
    within = await run_with(ctx, {**UNKNOWN, **capable})
    inconclusive_since(ctx, timedelta(hours=25))

    result = await run_with(ctx, {**UNKNOWN, **capable})

    assert warned(within)
    assert held(result)
    assert result.event.reason_code == Msg.INCONCLUSIVE_LONG.reason
    assert ctx.state.specs["storage_limit_supported"] is True


@pytest.mark.asyncio
async def test_unknown_for_over_a_day_after_one_failure_never_becomes_a_zero(
    context_factory, enforced
):
    ctx, capable = new_ctx(context_factory)
    await run_with(ctx, {**NO_LIMIT, **capable})
    await run_with(ctx, {**UNKNOWN, **capable})
    inconclusive_since(ctx, timedelta(hours=25))

    result = await run_with(ctx, {**UNKNOWN, **capable})

    assert held(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["read", "write", "expired"])
@pytest.mark.parametrize("probe", [NO_LIMIT, UNKNOWN])
async def test_a_lost_verdict_memory_holds_without_a_zero(context_factory, enforced, fault, probe):
    # after a pass: a timeout would otherwise reuse the pass
    ctx, capable = new_ctx(context_factory)
    await run_with(ctx, PASS)
    key = f"storage_limit_verdict:{ctx.miner_hotkey}:{ctx.executor.uuid}"
    if fault == "read":
        ctx.services.redis.failing = True
    elif fault == "write":
        ctx.services.redis.fail_next_set_of.add(key)
    else:
        ctx.services.redis.store.pop(key)

    result = await run_with(ctx, {**probe, **capable})

    assert held(result)
    if fault != "write":  # a write error still read the last verdict
        assert ctx.state.specs["storage_limit_supported"] is not True
