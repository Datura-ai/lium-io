"""StorageLimitCheck: a node whose Docker cannot cap a container's disk (18 Oct 2026, serdkey node 4837fec9).

A renter filled the disk of a node without a per-container storage limit and took the host down. Off by default:
a warning only. With REQUIRE_STORAGE_LIMIT_SUPPORTED an unrented node without the limit scores 0.
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


@pytest.mark.asyncio
async def test_a_node_with_the_limit_passes(context_factory, enforced):
    result = await run(context_factory, {"storage_limit_supported": True})

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_without_enforcement_a_node_without_the_limit_only_warns(context_factory):
    result = await run(
        context_factory,
        {"storage_limit_supported": False, "storage_limit_scrape_error": "pquota missing"},
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert result.event.what_we_saw["detail"] == "pquota missing"
    assert not result.updates


@pytest.mark.asyncio
@pytest.mark.parametrize("specs", [{"storage_limit_supported": False}, {}])
async def test_enforced_an_unrented_node_without_the_limit_scores_zero(
    context_factory, enforced, specs
):
    result = await run(context_factory, specs)

    assert result.passed is False
    assert result.event.reason_code == Msg.REQUIRED.reason
    assert result.updates["score"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [False, True])
async def test_enforced_only_an_unrented_node_is_hidden_from_new_rentals(
    context_factory, enforced, rented
):
    # the backend hides a node whose cycle carried an availability error at once, not at the hourly sweep
    result = await run(context_factory, {"storage_limit_supported": False}, rented=rented)

    codes = [error.reason_code for error in availability_errors([result.event])]
    assert codes == ([] if rented else ["STORAGE_LIMIT_REQUIRED"])


@pytest.mark.asyncio
async def test_enforced_a_rented_node_without_the_limit_only_warns(context_factory, enforced):
    result = await run(context_factory, {"storage_limit_supported": False}, rented=True)

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason


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
@pytest.mark.parametrize("capable", [None, True])
async def test_enforced_a_capable_or_unknown_host_scores_zero(context_factory, enforced, capable):
    specs = {"storage_limit_supported": False}
    if capable is not None:
        specs["storage_limit_capable"] = capable

    result = await run(context_factory, specs)

    assert result.passed is False
    assert result.event.reason_code == Msg.REQUIRED.reason


@pytest.mark.asyncio
async def test_enforced_a_single_disk_host_only_warns(context_factory, enforced):
    """DAH-1997 (Fish 21 Oct): required only of hosts that can build the limit."""
    result = await run(
        context_factory, {"storage_limit_supported": False, "storage_limit_capable": False}
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert "score" not in result.updates


INCONCLUSIVE = ["Test command timed out.", "An unexpected error occurred: boom"]


async def run_with(ctx, specs):
    # ContextState is frozen; a cycle's scrape refills the same specs dict
    ctx.state.specs.clear()
    ctx.state.specs.update(specs)
    return await StorageLimitCheck().run(ctx)


TIMEOUT = {"storage_limit_supported": None, "storage_limit_scrape_error": "Test command timed out."}
NO_LIMIT = {
    "storage_limit_supported": False,
    "storage_limit_scrape_error": "Storage limit is not supported.",
}


def inconclusive_since(ctx, delta: timedelta) -> None:
    # backdate the node's first inconclusive probe in its stored verdict
    key = next(k for k in ctx.services.redis.store if k.startswith("storage_limit_verdict:"))
    state = json.loads(ctx.services.redis.store[key])
    state["inconclusive_since"] = (datetime.now(UTC) - delta).isoformat()
    ctx.services.redis.store[key] = json.dumps(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", INCONCLUSIVE)
async def test_a_probe_without_a_verdict_reuses_a_recent_pass_and_publishes_it(
    context_factory, enforced, error
):
    ctx = context_factory(state=build_state(specs={}))
    await run_with(ctx, {"storage_limit_supported": True})

    result = await run_with(
        ctx, {"storage_limit_supported": False, "storage_limit_scrape_error": error}
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert result.event.what_we_saw["reused_last_pass"] is True
    # the published specs keep the limit, so the backend never offers this host uncapped
    assert ctx.state.specs["storage_limit_supported"] is True
    assert not result.updates


@pytest.mark.asyncio
@pytest.mark.parametrize("error", INCONCLUSIVE)
@pytest.mark.parametrize("scraped", [None, False])  # the scrape sends None; an older one sent False
async def test_a_probe_without_a_verdict_and_no_recent_pass_holds_the_node(
    context_factory, enforced, error, scraped
):
    ctx = context_factory(state=build_state(specs={}))
    result = await run_with(
        ctx, {"storage_limit_supported": scraped, "storage_limit_scrape_error": error}
    )

    assert result.passed is False
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["conclusive"] is False
    assert result.event.what_we_saw["storage_limit_supported"] is None
    assert result.updates["score"] == 0.0
    assert availability_errors([result.event])
    # published as unknown, never as "no limit"
    assert ctx.state.specs["storage_limit_supported"] is None


@pytest.mark.asyncio
async def test_unenforced_a_probe_without_a_verdict_is_published_as_unknown(context_factory):
    ctx = context_factory(state=build_state(specs={}))
    result = await run_with(
        ctx,
        {"storage_limit_supported": None, "storage_limit_scrape_error": "Test command timed out."},
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert ctx.state.specs["storage_limit_supported"] is None


@pytest.mark.asyncio
async def test_a_timeout_after_a_pass_keeps_the_pass_after_two_hours(context_factory, enforced):
    ctx = context_factory(state=build_state(specs={}))
    await run_with(ctx, {"storage_limit_supported": True})
    await run_with(ctx, TIMEOUT)
    inconclusive_since(ctx, timedelta(hours=2))

    result = await run_with(ctx, TIMEOUT)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert ctx.state.specs["storage_limit_supported"] is True


@pytest.mark.asyncio
async def test_a_timeout_after_a_fail_keeps_the_fail(context_factory, enforced):
    ctx = context_factory(state=build_state(specs={}))
    await run_with(ctx, NO_LIMIT)
    await run_with(ctx, NO_LIMIT)

    result = await run_with(ctx, TIMEOUT)

    assert result.passed is False
    assert result.event.reason_code == Msg.REQUIRED.reason
    assert result.updates["score"] == 0.0
    # published as the last verdict, so a whole-host rental cannot start uncapped
    assert ctx.state.specs["storage_limit_supported"] is False


@pytest.mark.asyncio
async def test_one_unsupported_after_a_pass_does_not_fire_required_the_second_does(
    context_factory, enforced
):
    ctx = context_factory(state=build_state(specs={}))
    await run_with(ctx, {"storage_limit_supported": True})

    first = await run_with(ctx, NO_LIMIT)
    # a timeout in between neither resets nor counts
    await run_with(ctx, TIMEOUT)
    second = await run_with(ctx, NO_LIMIT)

    assert first.passed is True
    assert first.event.reason_code == Msg.MISSING.reason
    assert "score" not in first.updates
    assert second.passed is False
    assert second.event.reason_code == Msg.REQUIRED.reason


@pytest.mark.asyncio
async def test_inconclusive_for_over_a_day_warns_without_a_score_change(context_factory, enforced):
    ctx = context_factory(state=build_state(specs={}))
    await run_with(ctx, {"storage_limit_supported": True})
    await run_with(ctx, TIMEOUT)
    inconclusive_since(ctx, timedelta(hours=25))

    result = await run_with(ctx, TIMEOUT)

    assert result.passed is True
    assert result.event.reason_code == Msg.INCONCLUSIVE_LONG.reason
    assert result.event.severity == "warning"
    assert not result.updates
    assert ctx.state.specs["storage_limit_supported"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["read", "write"])
@pytest.mark.parametrize("probe", [NO_LIMIT, TIMEOUT])
async def test_a_redis_error_on_the_verdict_memory_holds_the_node(
    context_factory, enforced, fault, probe
):
    # after a pass, "unsupported" would be a first warning and a timeout would reuse the pass
    ctx = context_factory(state=build_state(specs={}))
    await run_with(ctx, {"storage_limit_supported": True})
    if fault == "read":
        ctx.services.redis.failing = True
    else:
        ctx.services.redis.fail_next_set_of.add(
            f"storage_limit_verdict:{ctx.miner_hotkey}:{ctx.executor.uuid}"
        )

    result = await run_with(ctx, probe)

    assert result.passed is False
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.updates["score"] == 0.0
    assert ctx.state.specs["storage_limit_supported"] is not True
