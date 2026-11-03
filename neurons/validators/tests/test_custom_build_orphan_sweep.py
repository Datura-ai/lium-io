"""The custom-build orphan sweep spares a build that is still in flight: its pod is in the
pending-pods set, not yet in rented_data (prod 30 Oct: the sweep `docker rm -fv`'d a running
build's DinD one second into the build)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.task.checks.custom_build_orphan_sweep import (
    CustomBuildOrphanSweepCheck,
)
from services.redis_service import PENDING_PODS_PREFIX, RedisService

from tests.helpers import build_services

INFLIGHT = "lium-dind-build-pod-inflight"
ORPHAN = "lium-dind-build-pod-gone"


def _ssh():
    calls: list[str] = []

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        stdout = f"{INFLIGHT}\n{ORPHAN}\n" if "docker ps -a" in cmd else ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")

    return SimpleNamespace(run=run, calls=calls)


def _removed(ssh) -> list[str]:
    return [c.split()[3] for c in ssh.calls if c.startswith("/usr/bin/docker rm -fv")]


@pytest.mark.asyncio
async def test_a_pending_pods_build_dind_is_not_removed(context_factory):
    redis = AsyncMock()
    redis.get_pending_pods.return_value = [{"time": 0, "pod_id": "pod-inflight"}]
    ssh = _ssh()
    ctx = context_factory(ssh=ssh, services=build_services(redis=redis))

    result = await CustomBuildOrphanSweepCheck().run(ctx)

    assert result.passed is True
    assert _removed(ssh) == [ORPHAN]
    redis.get_pending_pods.assert_awaited_once_with(
        ctx.miner_hotkey, ctx.executor.uuid, max_age_seconds=None
    )


@pytest.mark.asyncio
async def test_unreadable_pending_pods_removes_nothing(context_factory):
    redis = AsyncMock()
    redis.get_pending_pods.side_effect = ConnectionError("redis down")
    ssh = _ssh()
    ctx = context_factory(ssh=ssh, services=build_services(redis=redis))

    result = await CustomBuildOrphanSweepCheck().run(ctx)

    assert result.passed is True
    assert ssh.calls == []


@pytest.mark.asyncio
async def test_a_build_pending_past_30_minutes_is_still_listed_for_the_sweep():
    # setup + build (1200 s) + export can run past the 30 min other readers allow
    entry = {"time": int(time.time()) - 40 * 60, "pod_id": "pod-long-build"}
    service = RedisService.__new__(RedisService)
    service.hget = AsyncMock(return_value=json.dumps([entry]))

    assert await service.get_pending_pods("miner", "exec") == []
    assert await service.get_pending_pods("miner", "exec", max_age_seconds=None) == [entry]
    service.hget.assert_awaited_with(PENDING_PODS_PREFIX, "miner:exec")


@pytest.mark.asyncio
async def test_a_later_pending_pod_write_keeps_a_build_past_30_minutes():
    entry = {"time": int(time.time()) - 40 * 60, "pod_id": "pod-long-build"}
    service = RedisService.__new__(RedisService)
    service.hget = AsyncMock(return_value=json.dumps([entry, {"time": 0, "pod_id": "pod-other"}]))
    service.hset = AsyncMock()
    service.hdel = AsyncMock()
    service._remove_unacked_container_created_unless = AsyncMock()  # the saved-report hash, not this one

    await service.add_pending_pod("miner", "exec", "pod-new")
    assert [i["pod_id"] for i in json.loads(service.hset.await_args.args[2])][:2] == [
        "pod-long-build",
        "pod-other",
    ]

    await service.remove_pending_pod("miner", "exec", "pod-other")
    assert json.loads(service.hset.await_args.args[2]) == [entry]
