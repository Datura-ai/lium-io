from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from payload_models.payloads import ContainerCreateRequest, ContainerDeleteRequest, FailedContainerRequest, WorkloadKind
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.executor_rollout import ExecutorRolloutTracker
from services.redis_service import RENT_REQUESTED_TTL_SECONDS, RedisService, rent_requested_key
from services.task import pipeline as pipeline_module
from services.task.checks.gpu_fault_probe import GpuFaultProbeCheck
from services.task.checks.nvlink_fault import NvlinkFaultCheck
from services.task.checks.pinned_memory_probe import PinnedMemoryProbeCheck
from services.task.checks.rented_mid_cycle import RentedMidCycleCheck
from services.task.checks.verifyx import VerifyXCheck
from services.task.messages import TenantEnforcementMessages as Msg, render_message
from services.task.pipeline import CheckResult, Pipeline, ParallelStage
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult

from clients.compute_client import ComputeClient
from core.config import settings
from tests.helpers import FakeRedis, build_services, build_state, make_context
from test_compute_client_restart_cut_creates import _client
from test_executor_rollout_grace import _failed, _scored, _validator_process

KEY = rent_requested_key("miner-hotkey", "executor-123")


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


def make_ctx(marked: str | None = None, at_start: RentedExecutorsResponse = UNRENTED, redis: FakeRedis | None = None):
    services = build_services(redis=redis or FakeRedis())
    if marked:
        services.redis.store[KEY] = marked
    return make_context(services=services, state=build_state(rented_data=at_start), ssh=NoSSH())


def redis_service(redis: FakeRedis | None = None) -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = redis or FakeRedis()
    service.lock = asyncio.Lock()
    return service


@pytest.mark.asyncio
async def test_a_rent_requested_since_cycle_start_ends_the_run_with_no_verdict():
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


def test_the_mark_outlives_one_validation_run_and_no_more():
    # a run that began before the rent is over within JOB_TIME_OUT; the mark must not reach far into the next cycle
    assert settings.JOB_TIME_OUT < RENT_REQUESTED_TTL_SECONDS <= settings.JOB_TIME_OUT + 120


@pytest.mark.asyncio
async def test_a_mark_left_by_a_crash_expires_on_its_own():
    redis = FakeRedis()
    await redis_service(redis).mark_rent_requested("miner-hotkey", "executor-123", "pod-new")

    assert redis.store[KEY] == "pod-new"
    assert redis.ttl[KEY] == RENT_REQUESTED_TTL_SECONDS


@pytest.mark.asyncio
async def test_clearing_one_pod_keeps_a_later_pods_mark():
    redis = FakeRedis()
    service = redis_service(redis)
    await service.mark_rent_requested("miner-hotkey", "executor-123", "pod-2")

    await service.clear_rent_requested("miner-hotkey", "executor-123", "pod-1")

    assert redis.store[KEY] == "pod-2"


# the connector: marks a renter's create on receipt, clears it when the create fails or the pod is deleted


def _create(kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL) -> ContainerCreateRequest:
    return ContainerCreateRequest.model_construct(
        pod_id="pod-new", miner_hotkey="miner-hotkey", executor_id="executor-123", workload_kind=kind
    )


def _connector(drive) -> tuple[MagicMock, FakeRedis]:
    redis = FakeRedis()
    client = _client(redis_service(redis))
    client.miner_service.redis_service = redis_service(redis)
    client._drive_miner_job = drive
    return client, redis


@pytest.mark.asyncio
async def test_a_renters_create_is_marked_while_it_runs_and_the_mark_stays_once_it_succeeds():
    seen = []

    async def drive(request):
        seen.append(redis.store.get(KEY))

    client, redis = _connector(drive)

    await client._run_miner_job(_create())

    assert seen == ["pod-new"]
    # the pod is up: the next cycle's rented list carries it, the mark ends with its TTL
    assert redis.store[KEY] == "pod-new"


@pytest.mark.asyncio
async def test_a_failed_create_clears_its_mark():
    client, redis = _connector(AsyncMock())
    client.miner_service.handle_container = AsyncMock(
        return_value=FailedContainerRequest.model_construct(
            miner_hotkey="miner-hotkey", executor_id="executor-123", pod_id="pod-new", msg="no"
        )
    )
    client._drive_miner_job = ComputeClient._drive_miner_job.__get__(client)
    client.get_miner_axon_info = AsyncMock(return_value=MagicMock(ip="miner", port=8000))
    client.lock = asyncio.Lock()

    await client._run_miner_job(_create())

    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_a_create_that_raises_clears_its_mark():
    client, redis = _connector(AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        await client._run_miner_job(_create())

    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_a_create_cancelled_mid_way_clears_its_mark():
    client, redis = _connector(AsyncMock(side_effect=asyncio.CancelledError))

    with pytest.raises(asyncio.CancelledError):
        await client._run_miner_job(_create())

    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_deleting_the_pod_clears_its_mark():
    client, redis = _connector(AsyncMock())
    redis.store[KEY] = "pod-new"

    await client._run_miner_job(
        ContainerDeleteRequest.model_construct(pod_id="pod-new", miner_hotkey="miner-hotkey", executor_id="executor-123")
    )

    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_a_filler_create_is_not_marked():
    client, redis = _connector(AsyncMock())

    await client._run_miner_job(_create(WorkloadKind.FILLER))

    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_a_redis_outage_does_not_stop_the_create():
    drive = AsyncMock()
    client, _ = _connector(drive)
    client.miner_service.redis_service = redis_service(FakeRedis(failing=True))

    await client._run_miner_job(_create())

    drive.assert_awaited_once()


# the pipeline: a rent that arrives while the GPU or network tests run stops them


class HeavyStep:
    check_id = "executor.heavy"
    fatal = True
    stops_on_rent = True

    def __init__(self, redis: FakeRedis | None = None):
        self.cancelled = False
        self.redis = redis
        self.finished = False

    async def run(self, ctx):
        try:
            if self.redis is not None:
                # the renter's create lands while the test runs
                self.redis.store[KEY] = "pod-new"
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return CheckResult(passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={}))


@pytest.fixture(autouse=True)
def fast_rent_poll(monkeypatch):
    monkeypatch.setattr(pipeline_module, "RENT_POLL_SECONDS", 0.01)


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.asyncio
async def test_a_rent_during_the_gpu_and_network_tests_stops_them_with_no_verdict(staged):
    ctx = make_ctx()
    heavy = HeavyStep(ctx.services.redis)

    ok, events, _ = await Pipeline([ParallelStage([[heavy]]) if staged else heavy], AsyncMock()).run(ctx)

    assert not ok and heavy.cancelled and not heavy.finished
    assert events[-1].reason_code == Msg.RENTED_SINCE_CYCLE_START.reason


@pytest.mark.asyncio
async def test_an_expired_mark_does_not_stop_the_tests():
    ctx = make_ctx()
    heavy = HeavyStep()

    ok, events, _ = await Pipeline([heavy], AsyncMock()).run(ctx)

    assert ok and heavy.finished
    assert events[-1].reason_code == Msg.NOT_RENTED.reason


@pytest.mark.asyncio
async def test_a_mark_for_a_pod_listed_at_cycle_start_does_not_stop_the_tests():
    ctx = make_ctx(marked="pod-old", at_start=rented(OLD))
    heavy = HeavyStep()

    ok, _, _ = await Pipeline([heavy], AsyncMock()).run(ctx)

    assert ok and heavy.finished


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
    assert before == ["LocalVerifyCheck", "ParallelStage" if fast_path else "CapabilityCheck"]
    assert names.index("TenantEnforcementCheck") < names.index("RentedMidCycleCheck")


# a step the rent poll cancels must not leave its remote work holding the renter's GPUs


class BlockingVerifyX:
    def __init__(self):
        self.run_dir = None
        self.started = asyncio.Event()
        self.kill_remote_verifyx = AsyncMock(return_value=True)

    async def validate_verifyx_and_process_job(self, *, run_dir=None, deadline_optional=False, **_):
        # the first sample falls back to the old run on a host without `timeout -k`
        assert deadline_optional
        self.run_dir = run_dir
        self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_cancelled_verifyx_run_is_killed_on_the_host():
    verifyx = BlockingVerifyX()
    ctx = make_context(services=build_services(verifyx=verifyx), state=build_state(specs={"gpu": {"count": 1}}))
    task = asyncio.ensure_future(VerifyXCheck().run(ctx))
    await asyncio.wait_for(verifyx.started.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert verifyx.run_dir
    verifyx.kill_remote_verifyx.assert_awaited_once()
    assert verifyx.kill_remote_verifyx.await_args.args[1] == verifyx.run_dir


class BlockingRunner:
    """The probe's run never ends; every later command (the kill) answers at once."""

    def __init__(self):
        self.calls: list[str] = []
        self.started = asyncio.Event()

    async def run(self, cmd, **_):
        self.calls.append(cmd)
        if len(self.calls) == 1:
            self.started.set()
            await asyncio.sleep(3600)
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd, command_id="k", exit_code=0, stdout="", stderr="", duration_ms=0,
            started_at=now, finished_at=now, success=True,
        )


@pytest.mark.parametrize(
    "check, specs",
    [
        (GpuFaultProbeCheck(enforce=True), {"gpu": {"count": 1}}),
        (PinnedMemoryProbeCheck(), {"gpu": {"count": 1}}),
        (NvlinkFaultCheck(), {"interconnect": {"nvlink": True, "nvlink_links": 18, "nvlink_active_links": 18}}),
    ],
)
@pytest.mark.asyncio
async def test_a_cancelled_gpu_probe_kills_its_remote_process(check, specs, monkeypatch):
    monkeypatch.setattr(settings, "PINNED_MEMORY_PROBE_CHECK_ENABLED", True)
    runner = BlockingRunner()
    ctx = make_context(services=build_services(), state=build_state(specs=specs), runner=runner)
    task = asyncio.ensure_future(check.run(ctx))
    await asyncio.wait_for(runner.started.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert check.stops_on_rent
    probe, kill = runner.calls
    tag = re.search(r"lium-probe-[0-9a-f]+", probe).group(0)
    assert tag in kill and "kill -KILL" in kill


class BlockingCardShell:
    """Each card's matmul never ends; the kills answer at once."""

    def __init__(self, cards: int):
        self.calls: list[str] = []
        self.cards = cards
        self.started = asyncio.Event()

    async def run(self, cmd, **_):
        self.calls.append(cmd)
        if "kill -KILL" in cmd:
            return MagicMock(exit_status=0, stdout="")
        if sum("kill -KILL" not in c for c in self.calls) == self.cards:
            self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_cancelled_all_cards_matmul_kills_every_card_run():
    from services.matrix_validation_service import ValidationService

    service = ValidationService.__new__(ValidationService)
    shell = BlockingCardShell(cards=2)
    executor = MagicMock(python_path="/usr/bin/python3")
    runs = asyncio.ensure_future(
        asyncio.gather(*(service._run_one_card(shell, executor, "/tmp/s.py", i, "params") for i in range(2)))
    )
    await asyncio.wait_for(shell.started.wait(), 5)

    runs.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runs

    card_runs = [c for c in shell.calls if "kill -KILL" not in c]
    kills = [c for c in shell.calls if "kill -KILL" in c]
    tags = {re.search(r"lium-probe-[0-9a-f]+", c).group(0) for c in card_runs}
    assert len(tags) == 2 and all(any(tag in kill for kill in kills) for tag in tags)


def test_the_post_rental_recheck_stops_on_a_rent():
    from services.task.checks.post_rental_gpu_recheck import PostRentalGpuRecheckCheck

    assert PostRentalGpuRecheckCheck.stops_on_rent
