"""The custom-build orphan sweep spares a build that is still in flight: its pod is in the
pending-pods set, not yet in rented_data (prod 30 Oct: the sweep `docker rm -fv`'d a running
build's DinD one second into the build)."""

from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from neurons.validators.src.services.task.checks.custom_build_orphan_sweep import (
    CustomBuildOrphanSweepCheck,
)
from protocol.vc_protocol.compute_requests import PodRentalActiveResponse
from services.redis_service import PENDING_PODS_PREFIX, RedisService

from tests.helpers import build_services

INFLIGHT = "lium-dind-build-pod-inflight"
ORPHAN = "lium-dind-build-pod-gone"


def _ssh(*extra_dind: str):
    calls: list[str] = []

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        stdout = "".join(f"{n}\n" for n in (INFLIGHT, ORPHAN, *extra_dind)) if "docker ps -a" in cmd else ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")

    return SimpleNamespace(run=run, calls=calls)


def _removed(ssh) -> list[str]:
    return [c.split()[3] for c in ssh.calls if c.startswith("/usr/bin/docker rm -fv")]


@pytest.mark.asyncio
async def test_a_pending_pods_build_dind_is_not_removed(context_factory):
    redis = RedisService.__new__(RedisService)
    redis.redis = FakeRedis(server=FakeServer())
    redis._remove_unacked_container_created_unless = AsyncMock()
    # another registration on the same node is creating a pod: only the backend knows that pod
    live_build_name = "lium-poddind-pod-other-registration.deadbeef"
    unreadable_build_name = "lium-poddind-pod-unreadable.deadbeef"
    await redis.add_pending_pod("miner-b", "exec-b", "pod-other-registration")
    ssh = _ssh(live_build_name, unreadable_build_name)
    ctx = context_factory(ssh=ssh, services=build_services(redis=redis))
    await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod-inflight")
    redis.get_pending_pods = AsyncMock(wraps=redis.get_pending_pods)

    async def rental_active(pod_id):
        if pod_id == "pod-unreadable":
            raise TimeoutError
        return PodRentalActiveResponse(active=pod_id == "pod-other-registration")

    ctx.services.backend.get_pod_rental_active.side_effect = rental_active

    result = await CustomBuildOrphanSweepCheck().run(ctx)

    assert result.passed is True
    removed_dind_names = _removed(ssh)
    assert removed_dind_names == [ORPHAN]
    assert await redis.get_pending_pods("miner-b", "exec-b")
    assert live_build_name not in removed_dind_names
    redis.get_pending_pods.assert_any_await(ctx.miner_hotkey, ctx.executor.uuid, max_age_seconds=None)


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
    service.redis = FakeRedis(server=FakeServer())
    await service.redis.hset(PENDING_PODS_PREFIX, "miner:exec", json.dumps([entry, {"time": 0, "pod_id": "pod-other"}]))
    service._remove_unacked_container_created_unless = AsyncMock()  # the saved-report hash, not this one

    async def stored() -> list[dict]:
        return json.loads(await service.redis.hget(PENDING_PODS_PREFIX, "miner:exec"))

    await service.add_pending_pod("miner", "exec", "pod-new")
    assert [i["pod_id"] for i in await stored()][:2] == ["pod-long-build", "pod-other"]

    await service.remove_pending_pod("miner", "exec", "pod-other")
    assert [i["pod_id"] for i in await stored()] == ["pod-long-build", "pod-new"]


def test_an_attempt_named_build_artifact_reads_as_its_pods():
    # review on lium-io#2394: build artifacts are named `<pod id>.<create attempt>`; the sweep keeps an active pod's
    from neurons.validators.src.services.task.checks.custom_build_orphan_sweep import (
        dind_build_pod_id,
        dind_container_pod_id,
    )

    assert dind_container_pod_id("lium-poddind-pod-1.ab12-fw-apply") == "pod-1"
    assert dind_build_pod_id("lium-poddind-pod-1.ab12cd34") == "pod-1"
    assert dind_build_pod_id("lium-poddind-pod-1.zzzz-not-hex") is None
    assert dind_build_pod_id("lium-poddind-pod-1.ab12") is None
    assert dind_build_pod_id("lium-poddind-pod-1.ab12-fw-remove") is None
    assert dind_build_pod_id("lium-dind-build-pod-1") == "pod-1"
    assert dind_container_pod_id("lium-dind-build-pod-1-fw-apply") == "pod-1"


def _listings(pod_id: str, attempt: str | None) -> dict[str, list[str]]:
    from services.docker_service import DockerService

    if attempt is None:  # the names a validator before per-attempt names writes
        image, scratch, dind = f"lium-build-{pod_id}", f"/tmp/lium-build-{pod_id}", f"lium-dind-build-{pod_id}"
    else:
        image = DockerService._custom_build_image_tag(pod_id, attempt)
        scratch = DockerService._custom_build_scratch_dir(pod_id, attempt)
        dind = DockerService._dind_container_name(pod_id, attempt)
    return {"images": [f"{image}:latest"], "scratch": [scratch], "dind": [dind, f"{dind}-fw-apply"]}


def _legacy_orphans(active_pods: set[str], listings: dict[str, list[str]]) -> tuple[list, list, list]:
    """The sweep of a validator released before per-attempt names (eec20430b), as it selects and reads: docker's
    `reference=lium-build-*` and `ls /tmp/lium-build-*` globs, `docker ps --filter name=lium-dind-build-` (a
    substring match), and the pod id is everything after its prefix."""
    from fnmatch import fnmatchcase

    def pod(rest: str) -> str:
        for suffix in ("-fw-apply", "-fw-remove"):
            rest = rest.removesuffix(suffix)
        return rest

    images = [i for i in listings["images"] if fnmatchcase(i.split(":")[0], "lium-build-*")]
    scratch = [s for s in listings["scratch"] if fnmatchcase(s, "/tmp/lium-build-*")]
    dind = [d for d in listings["dind"] if "lium-dind-build-" in d]
    return (
        [i for i in images if pod(i.split(":")[0][len("lium-build-"):]) not in active_pods],
        [s for s in scratch if pod(s[len("/tmp/lium-build-"):]) not in active_pods],
        [d for d in dind if pod(d[len("lium-dind-build-"):]) not in active_pods],
    )


async def _current_orphans(active_pods: set[str], listings: dict[str, list[str]]) -> tuple[list, list, list]:
    from fnmatch import fnmatchcase

    async def run(cmd, *args, **kwargs):
        # the node answers each listing with what its filters select, as docker and the shell would
        if "docker images" in cmd:
            pats = re.findall(r'reference=([^"]+)"', cmd)
            rows = [i for i in listings["images"] if any(fnmatchcase(i.split(":")[0], p) for p in pats)]
        elif "ls -1d" in cmd:
            globs = [w for w in cmd.split() if w.startswith("/tmp/")]
            rows = [s for s in listings["scratch"] if any(fnmatchcase(s, g) for g in globs)]
        else:
            subs = re.findall(r'name=([^"]+)"', cmd)
            rows = [d for d in listings["dind"] if any(s in d for s in subs)]
        return SimpleNamespace(exit_status=0, stdout="".join(f"{r}\n" for r in rows), stderr="")

    ssh = SimpleNamespace(run=run)
    check = CustomBuildOrphanSweepCheck.__new__(CustomBuildOrphanSweepCheck)
    return (
        await check._list_orphan_images(ssh, active_pods),
        await check._list_orphan_scratch_dirs(ssh, active_pods),
        await check._list_orphan_dind_containers(ssh, active_pods),
    )


@pytest.mark.asyncio
async def test_an_older_validators_sweep_never_selects_a_running_builds_attempt_names():
    # review on lium-io#2394: the validator and the connector roll out apart, so an older validator's sweep meets
    # the names this one writes; it would read `<pod id>.<attempt>` as an unknown pod and `docker rm -fv` the build
    current = _listings("pod-1", "ab12cd34")
    assert _legacy_orphans({"pod-1"}, current) == ([], [], [])
    assert await _current_orphans({"pod-1"}, current) == ([], [], [])
    # and a build of a pod no longer rented is still found, under either name
    legacy = _listings("pod-gone", None)
    assert await _current_orphans({"pod-1"}, legacy) == (legacy["images"], legacy["scratch"], legacy["dind"])
    gone = _listings("pod-gone", "ab12cd34")
    assert await _current_orphans({"pod-1"}, gone) == (gone["images"], gone["scratch"], gone["dind"])
    assert await _current_orphans({"pod-gone"}, legacy) == ([], [], [])
