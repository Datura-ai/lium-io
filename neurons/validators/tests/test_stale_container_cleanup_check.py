"""Tests for StaleContainerCleanupCheck (DAH-2164 follow-up).

The check reaps orphaned non-rented rental containers BEFORE the port checks so an
orphan that outlived its rental (e.g. a BROKEN_BY_PROVIDER pod whose container the
platform does not tear down) cannot keep binding the rental port range and deadlock
port verification.
"""

import asyncio

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from helpers import build_services, build_state, default_executor, make_context
from neurons.validators.src.services.task.checks import StaleContainerCleanupCheck
from neurons.validators.src.services.task.checks.stale_container_cleanup import (
    StaleContainerCleanupCheck as DirectImport,
)
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)


class RecordingContainerCleanup:
    """Records cleanup() calls and returns a configurable (count, names) result."""

    def __init__(self, result=(0, []), reclaimed=0, swept=0):
        self._result = result
        self._reclaimed = reclaimed
        self._swept = swept
        self.calls = []
        self.reclaim_calls = []
        self.sweep_calls = []

    async def cleanup(self, ssh_client, rented_data, executor_uuid):
        self.calls.append(
            {
                "ssh_client": ssh_client,
                "rented_data": rented_data,
                "executor_uuid": executor_uuid,
            }
        )
        return self._result

    async def reclaim_dphn_cache_when_disk_is_tight(self, ssh_client, executor_uuid):
        self.reclaim_calls.append({"ssh_client": ssh_client, "executor_uuid": executor_uuid})
        return self._reclaimed

    async def sweep_abandoned_download_temporaries(self, ssh_client, executor_uuid):
        self.sweep_calls.append({"ssh_client": ssh_client, "executor_uuid": executor_uuid})
        return self._swept


class SeenExecutorsRedis:
    """The one Redis call the check makes: SADD on the seen-executors set (DAH-1932).

    Starts with every UUID already seen, so the existing tests exercise a node the
    validator knows; ``seen=set()`` makes the next visit a first sight.
    """

    def __init__(self, seen=None, error=None):
        self.seen = set(seen) if seen is not None else None  # None = knows everyone
        self.error = error
        self.calls = []

    async def sadd(self, key, elem):
        self.calls.append((key, elem))
        if self.error:
            raise self.error
        if self.seen is None:
            return 0
        added = 0 if elem in self.seen else 1
        self.seen.add(elem)
        return added


def _make_ctx(cleanup, rented_data=None, redis=None):
    services = build_services(container_cleanup=cleanup, redis=redis or SeenExecutorsRedis())
    state = build_state(rented_data=rented_data)
    return make_context(services=services, state=state, ssh="ssh-conn-sentinel")


@pytest.mark.asyncio
async def test_runs_cleanup_and_passes():
    cleanup = RecordingContainerCleanup(result=(2, ["pod_orphan", "filler_old"]))
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert result.event.check_id == "executor.cleanup.stale_containers"
    assert result.event.reason_code == "STALE_CLEANUP_DONE"
    assert result.event.what_we_saw["removed_count"] == 2
    assert result.event.what_we_saw["removed_containers"] == ["pod_orphan", "filler_old"]


@pytest.mark.asyncio
async def test_passes_cleanup_args_through():
    cleanup = RecordingContainerCleanup()
    executor = default_executor()
    rented_data = RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=[RentedPod(pod_id="p1", container_name="pod_p1")],
                owner_flag=False,
            )
        },
        banned_guids=[],
    )
    services = build_services(container_cleanup=cleanup, redis=SeenExecutorsRedis())
    state = build_state(rented_data=rented_data)
    ctx = make_context(executor=executor, services=services, state=state, ssh="ssh-conn-sentinel")

    await StaleContainerCleanupCheck().run(ctx)

    assert len(cleanup.calls) == 1
    call = cleanup.calls[0]
    assert call["ssh_client"] == "ssh-conn-sentinel"
    assert call["executor_uuid"] == executor.uuid
    assert call["rented_data"] is rented_data


@pytest.mark.asyncio
async def test_noop_cleanup_still_passes():
    cleanup = RecordingContainerCleanup(result=(0, []))
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["removed_count"] == 0


def test_check_is_not_fatal():
    # Cleanup must never change the executor's verdict on its own.
    assert StaleContainerCleanupCheck.fatal is False
    assert StaleContainerCleanupCheck is DirectImport


def test_cleanup_runs_before_port_checks_in_production_pipeline():
    """Regression guard for the deadlock: cleanup must precede the port checks so a
    fatal PortCountCheck can never halt the pipeline before the orphan is reaped."""
    check_ids = [type(c).__name__ for c in PipelineFactory.build_checks()]

    assert "StaleContainerCleanupCheck" in check_ids
    idx_cleanup = check_ids.index("StaleContainerCleanupCheck")
    idx_conn = check_ids.index("PortConnectivityCheck")
    idx_count = check_ids.index("PortCountCheck")
    idx_tenant = check_ids.index("TenantEnforcementCheck")

    assert idx_cleanup < idx_conn < idx_count < idx_tenant


@pytest.mark.asyncio
async def test_also_reclaims_the_dphn_cache_when_disk_is_tight():
    # DAH-2475: this check is the ONLY caller of the reclaim backstop — the create-time sweep
    # deliberately never reclaims, so a node stranded under the listing floor is rescued here.
    cleanup = RecordingContainerCleanup(result=(0, []), reclaimed=2)
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert cleanup.reclaim_calls == [{"ssh_client": "ssh-conn-sentinel", "executor_uuid": ctx.executor.uuid}]
    assert result.passed
    assert result.event.what_we_saw["reclaimed_cache_volumes"] == 2


@pytest.mark.asyncio
async def test_also_sweeps_abandoned_download_temporaries():
    # DAH-2805: the sweep rides this same per-cycle visit, so a node is cleaned whatever image it
    # currently runs — including one whose filler already filled the disk and stopped mining.
    cleanup = RecordingContainerCleanup(swept=7)
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert cleanup.sweep_calls == [{"ssh_client": "ssh-conn-sentinel", "executor_uuid": ctx.executor.uuid}]
    assert result.passed
    assert result.event.what_we_saw["swept_download_temporaries"] == 7


@pytest.mark.asyncio
async def test_the_sweep_is_throttled_per_executor():
    # DAH-2805: the check runs every pipeline cycle (~15 min) but a temporary cannot become eligible
    # until it is hours old, so a per-executor cadence keeps the walk off every cycle.
    cleanup = RecordingContainerCleanup(swept=1)
    check = StaleContainerCleanupCheck()

    await check.run(_make_ctx(cleanup))
    second_result = await check.run(_make_ctx(cleanup))

    assert len(cleanup.sweep_calls) == 1
    assert second_result.event.what_we_saw["swept_download_temporaries"] is None


# DAH-1932: a re-registered executor (new subnet UUID, same box) shows up with empty rented_data
# because the tenant is still filed under the old UUID; the backend remaps only after this
# cycle's spec publish. Killing "orphans" on that first visit killed paying customers' pods.


@pytest.mark.asyncio
async def test_first_sight_of_an_executor_uuid_skips_container_removal():
    cleanup = RecordingContainerCleanup(result=(1, ["pod_tenant"]))
    redis = SeenExecutorsRedis(seen=set())
    ctx = _make_ctx(cleanup, redis=redis)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert cleanup.calls == []
    assert result.passed is True
    assert result.event.what_we_saw["removed_count"] == 0
    assert result.event.what_we_saw["first_sight_grace"] is True
    assert redis.calls == [("cleanup_seen_executors", ctx.executor.uuid)]
    # The disk sweeps do not depend on rented state and still run.
    assert len(cleanup.sweep_calls) == 1 and len(cleanup.reclaim_calls) == 1


@pytest.mark.asyncio
async def test_second_sight_of_the_same_uuid_cleans_up():
    cleanup = RecordingContainerCleanup(result=(1, ["pod_orphan"]))
    redis = SeenExecutorsRedis(seen=set())
    check = StaleContainerCleanupCheck()

    await check.run(_make_ctx(cleanup, redis=redis))
    second = await check.run(_make_ctx(cleanup, redis=redis))

    assert len(cleanup.calls) == 1
    assert second.event.what_we_saw["removed_count"] == 1
    assert second.event.what_we_saw["first_sight_grace"] is False


@pytest.mark.asyncio
async def test_a_known_uuid_is_cleaned_on_its_first_visit_after_a_validator_restart():
    """The set lives in Redis, not on the check instance: a fresh check object must not
    hand every executor another grace cycle."""
    cleanup = RecordingContainerCleanup(result=(1, ["pod_orphan"]))
    redis = SeenExecutorsRedis(seen=set())

    await StaleContainerCleanupCheck().run(_make_ctx(cleanup, redis=redis))
    await StaleContainerCleanupCheck().run(_make_ctx(cleanup, redis=redis))

    assert len(cleanup.calls) == 1


@pytest.mark.asyncio
async def test_redis_trouble_skips_removal_and_stays_non_fatal():
    cleanup = RecordingContainerCleanup(result=(1, ["pod_tenant"]))
    ctx = _make_ctx(cleanup, redis=SeenExecutorsRedis(error=ConnectionError("redis down")))

    result = await StaleContainerCleanupCheck().run(ctx)

    assert cleanup.calls == []
    assert result.passed is True
    assert result.event.what_we_saw["first_sight_grace"] is True


@pytest.mark.asyncio
async def test_redis_service_sadd_returns_one_only_for_a_new_member():
    # the grace rests on `RedisService.sadd` returning what Redis returns: 1 the first time, 0 after
    from neurons.validators.src.services.redis_service import CLEANUP_SEEN_EXECUTORS_SET, RedisService

    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()

    assert await service.sadd(CLEANUP_SEEN_EXECUTORS_SET, "uuid-b") == 1
    assert await service.sadd(CLEANUP_SEEN_EXECUTORS_SET, "uuid-b") == 0
    assert await service.sadd(CLEANUP_SEEN_EXECUTORS_SET, "uuid-c") == 1
