"""REGISTRY_PULL_FAILED: an idle node whose Docker daemon cannot pull a Docker Hub image through its own registry path.

ticket-0361 root cause (reported 23 Sep 18:46Z): a node failed 16 rents in 24 h, every one of
them a template image the node did not have cached. Its dockerd pulls through the registry mirror
docker.m.daocloud.io, whose DNS lookup times out; cached templates started fine. No check pulled anything, so the
node passed every one.
"""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from services.task.checks import registry_pull as module

from tests.helpers import FakeRedis, build_services, build_state, make_context

# Docker 29.1.3 (containerd image store) with that mirror configured and its DNS answers dropped: the 2.4 KB
# pull took 51 s on one run and over 60 s on another, and the 30 s bound ended it on both runs after; with the resolver refusing, the pull failed at once
# Docker 29.8.1, 23 Sep 2026, a REJECT rule in OUTPUT for every address of registry-1.docker.io (or the daemon's
# proxy), the committed hello-world digest pulled; the text is the daemon's own, verbatim


def test_defaults_bounded():
    """Regression: the check is on with no enforcement switch, pulls every 6 h and retries a failure after 30 min."""
    fields = type(module.settings).model_fields
    assert fields["REGISTRY_PULL_CHECK_ENABLED"].default is True
    assert "REGISTRY_PULL_ENFORCEMENT_ENABLED" not in fields
    assert fields["REGISTRY_PULL_PROBE_INTERVAL_HOURS"].default == 6.0
    assert fields["REGISTRY_PULL_PROBE_RETRY_MINUTES"].default == 30.0
    assert not [name for name in fields if name.startswith("REGISTRY_PULL_FLEET")]


# ------------------------------------------------------------------ the script under sh, docker stubbed


# ------------------------------------------------------------------ the check


def _pull_output(driver: str) -> str:
    return f"lium_pull driver={driver} mirrors=null\nlium_pull cached=no cache=removed\nlium_pull exit=0 seconds=2\n"


def _ctx(stdout: str):
    services = build_services()
    runner = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(error_type=None, stdout=stdout))
    )
    return make_context(
        services=services, state=build_state(specs={"gpu": {"count": 1}}), runner=runner
    )


async def _pull_now(ctx):
    # a scheduled state already due, so this cycle pulls
    state = module._ProbeState(at=time.time() - 10 * 24 * 3600, outcome="ok")
    await ctx.services.redis.set(f"{module._REDIS_PREFIX}:{ctx.executor.uuid}", state.as_json())
    return await module.RegistryPullCheck().run(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("driver, store", [("overlayfs", "containerd"), ("overlay2", "classic")])
async def test_the_image_store_rides_the_specs(driver, store):
    """DAH-5159: the backend moves a Sysbox many-layers failure only to a node whose specs say "containerd"."""
    result = await _pull_now(_ctx(_pull_output(driver)))
    assert result.passed is True
    assert result.updates["state"].specs == {"gpu": {"count": 1}, "docker_image_store": store}


@pytest.mark.asyncio
async def test_an_unknown_store_publishes_no_key():
    result = await _pull_now(_ctx(_pull_output("")))
    assert "state" not in result.updates


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [False, True])
async def test_a_cycle_between_pulls_publishes_the_last_reading(rented):
    ctx = _ctx(_pull_output("overlayfs"))
    await _pull_now(ctx)
    ctx.runner.run.reset_mock()
    if rented:  # a split node with a pod on it is still a move target
        rented_data = SimpleNamespace(
            executors={ctx.executor.uuid: SimpleNamespace(pods=[object()])}
        )
        ctx = ctx.model_copy(update={"state": replace(ctx.state, rented_data=rented_data)})

    result = await module.RegistryPullCheck().run(ctx)

    ctx.runner.run.assert_not_called()
    assert result.updates["state"].specs["docker_image_store"] == "containerd"


@pytest.mark.asyncio
async def test_a_rented_cycle_renews_the_records_ttl_without_a_pull():
    """Review of #1864: rented cycles never re-saved the record, so a rental past 7 days lost the store."""
    ctx = _ctx(_pull_output("overlayfs"))
    await _pull_now(ctx)
    key = module._state_key(ctx)
    redis = ctx.services.redis
    before = redis.store[key]
    redis.ttl[key] = 60  # nearly expired
    ctx.runner.run.reset_mock()
    rented_data = SimpleNamespace(executors={ctx.executor.uuid: SimpleNamespace(pods=[object()])})
    ctx = ctx.model_copy(update={"state": replace(ctx.state, rented_data=rented_data)})

    result = await module.RegistryPullCheck().run(ctx)

    ctx.runner.run.assert_not_called()
    assert redis.ttl[key] == module._REDIS_TTL_SECONDS
    assert redis.store[key] == before
    assert result.updates["state"].specs["docker_image_store"] == "containerd"


@pytest.mark.asyncio
async def test_a_pull_that_read_no_driver_keeps_the_last_store():
    """Review of #1864: a pull that did not run saved a record with no driver over the last one, so the node
    published no store for 3-9 h."""
    ctx = _ctx("")
    ctx.runner.run.return_value = SimpleNamespace(
        error_type="TimeoutError", error_message="ssh", stdout=""
    )
    last = module._ProbeState(
        at=time.time() - 10 * 24 * 3600,
        outcome="ok",
        reading={"driver": "overlayfs", "image_store": "containerd"},
    )
    await ctx.services.redis.set(f"{module._REDIS_PREFIX}:{ctx.executor.uuid}", last.as_json())

    result = await module.RegistryPullCheck().run(ctx)

    ctx.runner.run.assert_called_once()
    assert result.updates["state"].specs["docker_image_store"] == "containerd"
    saved = await module.RegistryPullCheck()._load(ctx)
    assert (saved.outcome, saved.reading["image_store"]) == ("not_run", "containerd")


# ------------------------------------------------------------------ the Docker Hub control


# ------------------------------------------------------------------ fleets over many cycles


@pytest.mark.asyncio
async def test_a_node_copying_the_uuid_keeps_its_own_pull_state(context_factory):
    redis = FakeRedis()
    owner = context_factory(services=build_services(redis=redis))
    copycat = context_factory(services=build_services(redis=redis), miner_hotkey="copycat-hotkey")
    check = module.RegistryPullCheck()

    await check._save(copycat, module._ProbeState(at=1.0, outcome=module._SCHEDULED))

    assert owner.executor.uuid == copycat.executor.uuid
    assert await check._load(owner) is None


@pytest.mark.asyncio
async def test_a_deploy_keeps_the_open_streak_saved_under_the_uuid_alone(context_factory):
    redis = FakeRedis()
    ctx = context_factory(services=build_services(redis=redis))
    legacy = module._ProbeState(at=1.0, outcome="pull_failed", failures_in_a_row=1)
    redis.store[f"{module._REDIS_PREFIX}:{ctx.executor.uuid}"] = legacy.as_json()

    loaded = await module.RegistryPullCheck()._load(ctx)

    assert loaded is not None and loaded.failures_in_a_row == 1
    assert redis.store[module._state_key(ctx)] == legacy.as_json()
