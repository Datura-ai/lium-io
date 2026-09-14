"""Tests for StaleContainerCleanupCheck (DAH-2164 follow-up).

The check reaps orphaned non-rented rental containers BEFORE the port checks so an
orphan that outlived its rental (e.g. a BROKEN_BY_PROVIDER pod whose container the
platform does not tear down) cannot keep binding the rental port range and deadlock
port verification.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis as FakeRedisClient
from helpers import build_services, build_state, default_executor, make_context
from neurons.validators.src.services.task.checks import StaleContainerCleanupCheck
from neurons.validators.src.services.task.checks.stale_container_cleanup import (
    REAPED_POD_STATE_RETENTION_SECONDS,
    REAPED_POD_STATES_KEY_PREFIX,
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

    def __init__(self, result=(0, [], []), reclaimed=0, swept=0):
        self._result = result
        self._reclaimed = reclaimed
        self._swept = swept
        self.calls = []
        self.reclaim_calls = []
        self.sweep_calls = []

    async def cleanup(self, ssh_client, rented_data, executor_uuid, on_before_remove=None):
        self.calls.append(
            {
                "ssh_client": ssh_client,
                "rented_data": rented_data,
                "executor_uuid": executor_uuid,
            }
        )
        # as the real cleanup does: the hook runs before every removal attempt, the failed ones too
        if on_before_remove is not None:
            for name in [*self._result[1], *self._result[2]]:
                await on_before_remove(name)
        return self._result

    async def reclaim_dphn_cache_when_disk_is_tight(self, ssh_client, executor_uuid):
        self.reclaim_calls.append({"ssh_client": ssh_client, "executor_uuid": executor_uuid})
        return self._reclaimed

    async def sweep_abandoned_download_temporaries(self, ssh_client, executor_uuid):
        self.sweep_calls.append({"ssh_client": ssh_client, "executor_uuid": executor_uuid})
        return self._swept


def _make_ctx(cleanup, rented_data=None, redis=None):
    services = build_services(container_cleanup=cleanup, redis=redis or FakeRedis())
    state = build_state(rented_data=rented_data)
    return make_context(services=services, state=state, ssh="ssh-conn-sentinel")


class FakeRedis:
    """What the check asks of redis: SADD on the seen-executors set (DAH-1932) and the three hash
    calls of the reaped-pod queue (DAH-3338), over one dict per key, bytes in and out like the
    real client without decode_responses.

    Starts with every UUID already seen, so most tests exercise a node the validator knows;
    ``seen=set()`` makes the next visit a first sight, ``error=`` makes SADD raise.
    """

    def __init__(self, hashes: dict[str, dict[str, str]] | None = None, seen=None, error=None):
        self.hashes: dict[str, dict[str, str]] = hashes or {}
        self.hset_calls: list[tuple[str, str, str]] = []
        self.hdel_calls: list[tuple[str, tuple[str, ...]]] = []
        self.expire_calls: list[tuple[str, int]] = []
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

    async def hset(self, key, field, value):
        self.hset_calls.append((key, field, value))
        self.hashes.setdefault(key, {})[field] = value

    async def hgetall(self, key):
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}

    async def hdel(self, key, *fields):
        self.hdel_calls.append((key, fields))
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)

    async def expire(self, key, seconds):
        self.expire_calls.append((key, seconds))


REAPED_POD_ID = "11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
EARLIER_POD_ID = "2b2b2b2b-0000-4000-8000-000000000002"


@pytest.mark.asyncio
async def test_runs_cleanup_and_passes():
    cleanup = RecordingContainerCleanup(result=(2, ["pod_orphan", "filler_old"], []))
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
    services = build_services(container_cleanup=cleanup, redis=FakeRedis())
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
    cleanup = RecordingContainerCleanup(result=(0, [], []))
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
    cleanup = RecordingContainerCleanup(result=(0, [], []), reclaimed=2)
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


@pytest.mark.asyncio
async def test_unremovable_orphan_is_recorded_in_state_for_the_port_check():
    """DAH-2991: an orphan that survives removal is named in the event and handed to PortCountCheck."""
    cleanup = RecordingContainerCleanup(result=(0, [], ["pod_orphan"]))
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["unremovable_containers"] == ["pod_orphan"]
    assert result.updates["state"].orphaned_containers == ["pod_orphan"]


# DAH-1932: a re-registered executor (new subnet UUID, same box) shows up with empty rented_data
# because the tenant is still filed under the old UUID; the backend remaps only after this
# cycle's spec publish. Killing "orphans" on that first visit killed paying customers' pods.


@pytest.mark.asyncio
async def test_first_sight_of_an_executor_uuid_skips_container_removal():
    cleanup = RecordingContainerCleanup(result=(1, ["pod_tenant"], []))
    redis = FakeRedis(seen=set())
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
    cleanup = RecordingContainerCleanup(result=(1, ["pod_orphan"], []))
    redis = FakeRedis(seen=set())
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
    cleanup = RecordingContainerCleanup(result=(1, ["pod_orphan"], []))
    redis = FakeRedis(seen=set())

    await StaleContainerCleanupCheck().run(_make_ctx(cleanup, redis=redis))
    await StaleContainerCleanupCheck().run(_make_ctx(cleanup, redis=redis))

    assert len(cleanup.calls) == 1


@pytest.mark.asyncio
async def test_redis_trouble_skips_removal_and_stays_non_fatal():
    cleanup = RecordingContainerCleanup(result=(1, ["pod_tenant"], []))
    ctx = _make_ctx(cleanup, redis=FakeRedis(error=ConnectionError("redis down")))

    result = await StaleContainerCleanupCheck().run(ctx)

    assert cleanup.calls == []
    assert result.passed is True
    assert result.event.what_we_saw["first_sight_grace"] is True


@pytest.mark.asyncio
async def test_redis_service_sadd_returns_one_only_for_a_new_member():
    # the grace rests on `RedisService.sadd` returning what Redis returns: 1 the first time, 0 after
    from neurons.validators.src.services.redis_service import CLEANUP_SEEN_EXECUTORS_SET, RedisService

    service = RedisService.__new__(RedisService)
    service.redis = FakeRedisClient(server=FakeServer())
    service.lock = asyncio.Lock()

    assert await service.sadd(CLEANUP_SEEN_EXECUTORS_SET, "uuid-b") == 1
    assert await service.sadd(CLEANUP_SEEN_EXECUTORS_SET, "uuid-b") == 0
    assert await service.sadd(CLEANUP_SEEN_EXECUTORS_SET, "uuid-c") == 1


# ---------------------------------------------------------------------------
# DAH-3338: what the sweep reaped is reported to the backend as container_state = reaped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaped_pod_containers_are_reported_as_reaped_pod_states():
    """Only pod_* names carry a pod id; a reaped health-check container belongs to no rental."""
    cleanup = RecordingContainerCleanup(
        result=(2, ["pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7", "container_healthcheck_abc"], [])
    )
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    states = result.updates["state"].pod_states
    assert [(s.pod_id, s.container_state.value) for s in states] == [
        ("11655dc5-53ba-4a8d-a341-fe6c9d12bda7", "reaped")
    ]
    assert states[0].observed_at.tzinfo is not None
    # DAH-2991's orphan list is carried in the same state object, untouched.
    assert result.updates["state"].orphaned_containers == []


@pytest.mark.asyncio
async def test_nothing_reaped_and_nothing_unremovable_leaves_the_state_alone():
    cleanup = RecordingContainerCleanup(result=(1, ["container_healthcheck_abc"], []))
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.updates == {}


# ---------------------------------------------------------------------------
# DAH-3338 (review): `reaped` leaves the validator once over a lossy pub/sub, so it is queued in
# redis before the removal and re-sent every cycle inside the retention window.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaped_pod_is_queued_in_redis_before_the_container_is_removed():
    cleanup = RecordingContainerCleanup(result=(1, [f"pod_{REAPED_POD_ID}"], []))
    redis = FakeRedis()
    ctx = _make_ctx(cleanup, redis=redis)

    result = await StaleContainerCleanupCheck().run(ctx)

    key = f"{REAPED_POD_STATES_KEY_PREFIX}{ctx.executor.uuid}"
    # queued by the hook the cleanup awaits before the `docker rm`, once — the post-removal add
    # is the same id in the same cycle and writes nothing new
    assert [call[:2] for call in redis.hset_calls] == [(key, REAPED_POD_ID)]
    # the hash of a node that leaves the fleet is never read again: the TTL is what removes it
    assert redis.expire_calls == [(key, REAPED_POD_STATE_RETENTION_SECONDS)]
    assert [s.pod_id for s in result.updates["state"].pod_states] == [REAPED_POD_ID]


@pytest.mark.asyncio
async def test_a_reaped_pod_from_an_earlier_cycle_is_sent_again():
    """Regression: with nothing reaped this cycle the earlier report used to be gone for good —
    a cycle whose pub/sub had no subscriber lost it, and the container was no longer there to reap."""
    an_hour_ago = datetime.now(UTC) - timedelta(hours=1)
    executor = default_executor()
    key = f"{REAPED_POD_STATES_KEY_PREFIX}{executor.uuid}"
    redis = FakeRedis({key: {EARLIER_POD_ID: an_hour_ago.isoformat()}})
    cleanup = RecordingContainerCleanup(result=(0, [], []))
    services = build_services(container_cleanup=cleanup, redis=redis)
    ctx = make_context(executor=executor, services=services, state=build_state(), ssh="ssh-conn-sentinel")

    result = await StaleContainerCleanupCheck().run(ctx)

    states = result.updates["state"].pod_states
    assert [(s.pod_id, s.container_state.value, s.observed_at) for s in states] == [
        (EARLIER_POD_ID, "reaped", an_hour_ago)
    ]
    assert redis.hdel_calls == []


@pytest.mark.asyncio
async def test_a_first_sight_grace_cycle_still_sends_the_queued_reaps():
    """Regression: DAH-1932's grace skips the removal, and a queue built only next to the removal
    would skip the re-send too, so an id reaped before the grace cycle went missing for a cycle."""
    an_hour_ago = datetime.now(UTC) - timedelta(hours=1)
    executor = default_executor()
    key = f"{REAPED_POD_STATES_KEY_PREFIX}{executor.uuid}"
    redis = FakeRedis({key: {EARLIER_POD_ID: an_hour_ago.isoformat()}}, seen=set())
    cleanup = RecordingContainerCleanup(result=(1, ["pod_tenant"], []))
    services = build_services(container_cleanup=cleanup, redis=redis)
    ctx = make_context(executor=executor, services=services, state=build_state(), ssh="ssh-conn-sentinel")

    result = await StaleContainerCleanupCheck().run(ctx)

    assert cleanup.calls == []
    assert result.event.what_we_saw["first_sight_grace"] is True
    assert [(s.pod_id, s.container_state.value) for s in result.updates["state"].pod_states] == [
        (EARLIER_POD_ID, "reaped")
    ]


@pytest.mark.asyncio
async def test_a_reaped_pod_older_than_the_retention_window_is_dropped():
    two_days_ago = datetime.now(UTC) - timedelta(days=2)
    executor = default_executor()
    key = f"{REAPED_POD_STATES_KEY_PREFIX}{executor.uuid}"
    redis = FakeRedis({key: {EARLIER_POD_ID: two_days_ago.isoformat()}})
    cleanup = RecordingContainerCleanup(result=(0, [], []))
    services = build_services(container_cleanup=cleanup, redis=redis)
    ctx = make_context(executor=executor, services=services, state=build_state(), ssh="ssh-conn-sentinel")

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.updates == {}
    assert redis.hdel_calls == [(key, (EARLIER_POD_ID,))]
    assert redis.hashes[key] == {}


@pytest.mark.asyncio
async def test_an_unremovable_pod_is_taken_off_the_queue_and_not_reported_as_reaped():
    """Queued before the attempt (the hook runs first), still on the host after it: not reaped."""
    cleanup = RecordingContainerCleanup(result=(0, [], [f"pod_{REAPED_POD_ID}"]))
    redis = FakeRedis()
    ctx = _make_ctx(cleanup, redis=redis)

    result = await StaleContainerCleanupCheck().run(ctx)

    key = f"{REAPED_POD_STATES_KEY_PREFIX}{ctx.executor.uuid}"
    assert redis.hashes[key] == {}
    assert result.updates["state"].pod_states == []
    assert result.updates["state"].orphaned_containers == [f"pod_{REAPED_POD_ID}"]


@pytest.mark.asyncio
async def test_a_pod_container_whose_suffix_is_not_a_uuid_is_not_reported():
    """The host names its containers; the backend refuses a pod id over 64 chars and skips one it
    cannot read as a UUID, so such a name never goes on the wire (and is never queued)."""
    cleanup = RecordingContainerCleanup(result=(2, ["pod_not-a-uuid", "pod_" + "a" * 70], []))
    redis = FakeRedis()
    ctx = _make_ctx(cleanup, redis=redis)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.updates == {}
    assert redis.hset_calls == []


@pytest.mark.asyncio
async def test_a_redis_error_does_not_stop_the_report_of_this_cycle():
    class BrokenRedis(FakeRedis):
        async def hset(self, key, field, value):
            raise ConnectionError("redis down")

        async def hgetall(self, key):
            raise ConnectionError("redis down")

    cleanup = RecordingContainerCleanup(result=(1, [f"pod_{REAPED_POD_ID}"], []))
    ctx = _make_ctx(cleanup, redis=BrokenRedis())

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert [s.pod_id for s in result.updates["state"].pod_states] == [REAPED_POD_ID]
