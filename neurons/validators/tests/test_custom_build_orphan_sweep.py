"""The custom-build orphan sweep spares a build that is still in flight: its pod is in the
pending-pods set, not yet in rented_data (prod 30 Oct: the sweep `docker rm -fv`'d a running
build's DinD one second into the build)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.task.checks.custom_build_orphan_sweep import (
    CustomBuildOrphanSweepCheck,
)

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
    redis.get_pending_pods.assert_awaited_once_with(ctx.miner_hotkey, ctx.executor.uuid)


@pytest.mark.asyncio
async def test_unreadable_pending_pods_removes_nothing(context_factory):
    redis = AsyncMock()
    redis.get_pending_pods.side_effect = ConnectionError("redis down")
    ssh = _ssh()
    ctx = context_factory(ssh=ssh, services=build_services(redis=redis))

    result = await CustomBuildOrphanSweepCheck().run(ctx)

    assert result.passed is True
    assert ssh.calls == []
