from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.executor_rollout import ExecutorRolloutTracker
from services.redis_service import PENDING_PODS_PREFIX, RENTAL_CHANGE_STATE_KEY, RedisService, create_started_at_key
from services.task.checks import rented_mid_cycle as rented_mid_cycle_module
from services.task.checks.rented_mid_cycle import RentedMidCycleCheck
from services.task.messages import TenantEnforcementMessages as Msg
from services.task.pipeline_factory import PipelineFactory

from tests.helpers import FakeRedis, build_context_config, build_services, build_state, make_context
from test_executor_rollout_grace import _failed, _scored, _validator_process

NODE = "miner-hotkey:executor-123"


def mark(redis: FakeRedis, pod: str, age: float = 0, renter: bool = True) -> None:
    """A pending-pod entry of a create on the node, as RedisService.add_pending_pod writes it."""
    entries = json.loads(redis.hashes.get(PENDING_PODS_PREFIX, {}).get(NODE) or "[]")
    entry = {"time": int(time.time() - age), "pod_id": pod, **({"renter": True} if renter else {})}
    redis.hashes.setdefault(PENDING_PODS_PREFIX, {})[NODE] = json.dumps([*entries, entry])


def stamp(redis: FakeRedis, at: float) -> None:
    redis.store[create_started_at_key("miner-hotkey", "executor-123")] = str(at)


def rented(*pods: RentedPod) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=list(pods),
            )
        },
        banned_guids=[],
    )


UNRENTED = RentedExecutorsResponse(executors={}, banned_guids=[])
OLD = RentedPod(pod_id="pod-old", container_name="tenant-old", rented_ports=[])


class NoSSH:
    """The check must not read the host: a starting pod is not judged."""

    async def run(self, command: str):
        raise AssertionError(f"unexpected SSH call: {command}")


def redis_service(redis: FakeRedis | None = None) -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = redis or FakeRedis()
    service.lock = asyncio.Lock()
    return service


def make_ctx(marked: str | None = None, at_start: RentedExecutorsResponse = UNRENTED, redis: FakeRedis | None = None):
    # the check reads through RedisService.get_pending_pods, so the fake client sits under the real service
    services = build_services(redis=redis_service(redis or FakeRedis()))
    if marked:
        mark(services.redis.redis, marked)
    return make_context(services=services, state=build_state(rented_data=at_start), ssh=NoSSH())


@pytest.mark.asyncio
async def test_a_rent_since_cycle_start_ends_the_run_with_no_verdict():
    result = await RentedMidCycleCheck().run(make_ctx(marked="pod-new"))

    assert not result.passed and not result.halt
    assert result.event.reason_code == Msg.RENTED_SINCE_CYCLE_START.reason
    # no score, no rented flag, no reset of the verified job
    assert not result.updates


@pytest.mark.parametrize(
    "marked, at_start, redis",
    [
        # no mark, or one that expired
        (None, UNRENTED, None),
        # the pod was listed at start: TenantEnforcementCheck judged it, nothing new here
        ("pod-old", rented(OLD), None),
        # Redis down: no evidence of a rent, the run goes on
        (None, UNRENTED, FakeRedis(failing=True)),
    ],
)
@pytest.mark.asyncio
async def test_no_rent_since_cycle_start_goes_on(marked, at_start, redis):
    result = await RentedMidCycleCheck().run(make_ctx(marked, at_start, redis))

    assert result.passed and not result.halt
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert not result.updates


@pytest.mark.asyncio
async def test_a_starved_redis_pool_does_not_hold_the_check(monkeypatch):
    """The GET is bounded: a read stuck waiting for a pool connection counts as no rent."""
    ctx = make_ctx()

    async def stuck(_key):
        await asyncio.sleep(3600)

    ctx.services.redis.hget = lambda *_args: stuck(None)
    monkeypatch.setattr(rented_mid_cycle_module, "RENT_MARK_READ_TIMEOUT_SECONDS", 0.01)
    result = await RentedMidCycleCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.NOT_RENTED.reason


@pytest.mark.parametrize(
    "write, stops",
    [
        # a renter's create that finished after the cycle started: its entry is gone, the stamp is newer
        (lambda redis: stamp(redis, time.time()), True),
        # a stamp from before the cycle started: that create's pod is in the cycle-start list or it failed
        (lambda redis: stamp(redis, time.time() - 3600), False),
        # a filler's or the probe's create: an entry without the renter flag and no stamp
        (lambda redis: mark(redis, "pod-filler", renter=False), False),
        # a crashed create's entry, older than get_pending_pods keeps
        (lambda redis: mark(redis, "pod-crashed", age=31 * 60), False),
    ],
)
@pytest.mark.asyncio
async def test_the_marks_the_verifyx_cold_retry_reads_decide(write, stops):
    redis = FakeRedis()
    ctx = make_ctx(redis=redis)
    write(redis)

    result = await RentedMidCycleCheck().run(ctx)

    assert result.passed is not stops


# the create writes the marks: a renter's entry and the stamp; a filler's and the probe's neither


@pytest.mark.parametrize(
    "kwargs, renter",
    [({}, True), ({"is_filler_create": True}, False), ({"synthetic_probe": True}, False), ({"kind": "gpu_reset"}, False)],
)
@pytest.mark.asyncio
async def test_only_a_renters_create_writes_the_marks(kwargs, renter):
    from fakeredis import FakeServer
    from fakeredis.aioredis import FakeRedis as RealFakeRedis

    service = RedisService.__new__(RedisService)
    service.redis = RealFakeRedis(server=FakeServer())

    await service.add_pending_pod("miner-hotkey", "executor-123", "pod-new", **kwargs)

    entries = json.loads(await service.redis.hget(PENDING_PODS_PREFIX, NODE))
    assert [bool(e.get("renter")) for e in entries] == [renter]
    assert (await service.redis.get(create_started_at_key("miner-hotkey", "executor-123")) is not None) is renter


@pytest.mark.asyncio
async def test_the_run_ended_by_a_new_pod_gets_no_verdict_this_cycle():
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    held = _failed("node-1", Msg.RENTED_SINCE_CYCLE_START.reason)
    healthy = _scored("node-2")

    kept, withheld = await validator_process.withhold_verdicts_for_rollout(
        {"5Miner": [held, healthy]}, 7500, "batch-1", MagicMock()
    )

    assert kept == {"5Miner": [healthy]}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.parametrize("fast_path", [False, True])
def test_rechecked_right_before_the_gpu_and_network_tests(fast_path):
    names = [type(check).__name__ for check in PipelineFactory.build_checks(fast_path=fast_path)]
    before = [names[i + 1] for i, name in enumerate(names) if name == "RentedMidCycleCheck"]
    # serial: once before the rental check and once after it, since its probe can take up to 300 s
    serial = ["RentalVerificationCheck", "LocalVerifyCheck", "CapabilityCheck"]
    expected = ["LocalVerifyCheck", "CapabilityCheck", "VerifyXCheck"] if fast_path else serial
    assert before == expected
    assert names.index("TenantEnforcementCheck") < names.index("RentedMidCycleCheck")


@pytest.mark.asyncio
async def test_a_create_stamped_between_the_rented_list_read_and_the_task_start_stops_the_run():
    """The stamp is compared with the rented list's read time, not the task start."""
    redis = FakeRedis()
    listed = UNRENTED.model_copy(update={"read_at": time.time() - 120})
    ctx = make_ctx(at_start=listed, redis=redis)
    stamp(redis, time.time() - 60)  # after the list was read, before this task started

    result = await RentedMidCycleCheck().run(ctx)

    assert not result.passed and result.event.reason_code == Msg.RENTED_SINCE_CYCLE_START.reason


def test_a_hold_for_a_create_mid_cycle_keeps_no_miner_active():
    """A provider can start a create on his own node, so that hold earns no eligibility floor."""
    from services.executor_rollout import session_kept_hotkeys, withhold_renter_gpu_reset_verdicts

    held = _failed("executor-123", Msg.RENTED_SINCE_CYCLE_START.reason)
    _, withheld = withhold_renter_gpu_reset_verdicts({"5Miner": [held]})

    assert len(withheld) == 1 and session_kept_hotkeys(withheld) == set()


def batch_ctx(batch: str, redis: FakeRedis):
    services = build_services(redis=redis_service(redis))
    mark(redis, "pod-new")
    return make_context(
        services=services,
        state=build_state(rented_data=UNRENTED),
        ssh=NoSSH(),
        config=build_context_config(job_batch_id=batch),
    )


@pytest.mark.asyncio
async def test_a_passing_node_rented_in_consecutive_cycles_still_gets_a_verdict():
    """DAH-7343 (details on the ticket)."""
    redis = FakeRedis()
    first = await RentedMidCycleCheck().run(batch_ctx("2026-11-28 11:00:00", redis))
    again = await RentedMidCycleCheck().run(batch_ctx("2026-11-28 11:00:00", redis))  # the express lane asks again
    second = await RentedMidCycleCheck().run(batch_ctx("2026-11-28 11:15:00", redis))
    third = await RentedMidCycleCheck().run(batch_ctx("2026-11-28 11:30:00", redis))

    assert [first.passed, again.passed, second.passed] == [False, False, False]
    assert second.event.reason_code == Msg.RENTED_SINCE_CYCLE_START.reason
    assert third.passed
    assert json.loads(redis.hashes[RENTAL_CHANGE_STATE_KEY][NODE])["batches"] == [
        "2026-11-28 11:00:00",
        "2026-11-28 11:15:00",
    ]


@pytest.mark.asyncio
async def test_the_pause_and_the_rental_change_rule_share_one_per_node_state():
    redis = FakeRedis()
    redis.hashes[RENTAL_CHANGE_STATE_KEY] = {NODE: json.dumps({"batches": ["b1", "b2"], "at": time.time()})}
    assert (await RentedMidCycleCheck().run(batch_ctx("b3", redis))).passed

    redis.hashes[RENTAL_CHANGE_STATE_KEY].pop(NODE)  # what withhold_rental_change_verdicts does on a real pass
    assert not (await RentedMidCycleCheck().run(batch_ctx("b4", redis))).passed
