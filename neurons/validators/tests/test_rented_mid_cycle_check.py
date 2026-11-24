from __future__ import annotations

import asyncio
import json
import re
import signal
import sys
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.executor_rollout import ExecutorRolloutTracker
from services.redis_service import PENDING_PODS_PREFIX, RedisService, create_started_at_key
from services.task import pipeline as pipeline_module
from services.task.checks.gpu_fault_probe import GpuFaultProbeCheck
from services.task.checks.nvlink_fault import NvlinkFaultCheck
from services.task.checks.pinned_memory_probe import PinnedMemoryProbeCheck
from services.task.checks import rented_mid_cycle as rented_mid_cycle_module
from services.task.checks.rented_mid_cycle import RentedMidCycleCheck
from services.task.checks.verifyx import VerifyXCheck
from services.task.messages import TenantEnforcementMessages as Msg, render_message
from services.task.pipeline import CheckResult, Pipeline, ParallelStage
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult, SSHCommandRunner, run_killed_on_cancel, run_to_end

from core.config import settings
from tests.helpers import FakeRedis, build_services, build_state, make_context
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
    [({}, True), ({"filler": True}, False), ({"synthetic_probe": True}, False), ({"kind": "gpu_reset"}, False)],
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
                mark(self.redis, "pod-new")
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
    heavy = HeavyStep(ctx.services.redis.redis)

    ok, events, _ = await Pipeline([ParallelStage([[heavy]]) if staged else heavy], AsyncMock()).run(ctx)

    assert not ok and heavy.cancelled and not heavy.finished
    assert events[-1].reason_code == Msg.RENTED_SINCE_CYCLE_START.reason


class SlowCleanupStep(HeavyStep):
    """Its cancel cleanup (the remote kill) takes an await of its own."""

    def __init__(self):
        super().__init__()
        self.cleaned_up = False

    async def run(self, ctx):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await run_to_end(self._clean_up())
            raise

    async def _clean_up(self):
        await asyncio.sleep(0.05)
        self.cleaned_up = True


@pytest.mark.asyncio
async def test_an_outside_cancel_returns_only_after_the_steps_own_cleanup_ran():
    # JOB_TIME_OUT cancels the run; SSH closes right after, so the remote kill must have run before the run returns
    step = SlowCleanupStep()
    run = asyncio.ensure_future(Pipeline([step], AsyncMock()).run(make_ctx()))
    await asyncio.sleep(0.05)

    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    assert step.cleaned_up


class HeldReadRedis(FakeRedis):
    """The first rent-mark read waits until the test answers it; later reads find no mark."""

    def __init__(self):
        super().__init__()
        self.read = asyncio.get_running_loop().create_future()
        self.called = asyncio.Event()

    async def hget(self, key, field):
        if self.called.is_set():
            return None
        self.called.set()
        return await self.read


class SubprocessStep(HeavyStep):
    """Local GPU-work stand-in: a 2 s subprocess, killed on cancel."""

    async def run(self, ctx):
        self.proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(2)")
        try:
            await self.proc.wait()
        except asyncio.CancelledError:
            self.proc.kill()
            await self.proc.wait()
            raise
        self.finished = True
        return CheckResult(passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={}))


@pytest.mark.parametrize("read_done_same_turn", [False, True])
@pytest.mark.asyncio
async def test_an_outside_cancel_is_never_lost_to_a_rent_read_that_completes_in_the_same_turn(read_done_same_turn):
    redis = HeldReadRedis()
    step = SubprocessStep()
    run = asyncio.ensure_future(Pipeline([step], AsyncMock()).run(make_ctx(redis=redis)))
    await redis.called.wait()

    if read_done_same_turn:
        redis.read.set_result(None)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, 1.5)
    assert step.proc.returncode == -signal.SIGKILL and not step.finished


class LosesTheFirstCancelStep(HeavyStep):
    """Like a 3.11 wait_for (asyncssh's own `run(timeout=)`) reached as its command completes: the first cancel is
    lost and the step goes on to its next remote command."""

    def __init__(self, redis: FakeRedis | None = None):
        super().__init__(redis)
        self.started = asyncio.Event()
        self.went_on = False

    async def run(self, ctx):
        self.started.set()
        try:
            if self.redis is not None:
                mark(self.redis, "pod-new")
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        self.went_on = True
        return await self._next_command(ctx)

    async def _next_command(self, ctx):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return CheckResult(passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={}))


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.asyncio
async def test_a_rent_stops_a_step_that_lost_the_first_cancel(staged):
    ctx = make_ctx()
    step = LosesTheFirstCancelStep(ctx.services.redis.redis)

    ok, events, _ = await asyncio.wait_for(
        Pipeline([ParallelStage([[step]]) if staged else step], AsyncMock()).run(ctx), 2
    )

    assert not ok and step.went_on and step.cancelled and not step.finished
    assert events[-1].reason_code == Msg.RENTED_SINCE_CYCLE_START.reason


@pytest.mark.asyncio
async def test_an_outside_cancel_stops_a_step_that_lost_the_first_cancel():
    step = LosesTheFirstCancelStep()
    run = asyncio.ensure_future(Pipeline([step], AsyncMock()).run(make_ctx()))
    await step.started.wait()

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, 2)

    assert step.went_on and step.cancelled and not step.finished


class SlowKillRunner:
    """The probe never ends; the kill takes a moment, during which the pipeline cancels the step again."""

    def __init__(self):
        self.started = asyncio.Event()
        self.killed = False

    async def run(self, cmd, **_):
        if not self.started.is_set():
            self.started.set()
            await asyncio.sleep(3600)
        await asyncio.sleep(0.1)
        self.killed = True
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd, command_id="k", exit_code=0, stdout="", stderr="", duration_ms=0,
            started_at=now, finished_at=now, success=True,
        )


@pytest.mark.asyncio
async def test_a_probes_remote_kill_finishes_through_a_second_cancel():
    runner = SlowKillRunner()
    task = asyncio.ensure_future(run_killed_on_cancel(runner, "probe", seconds=60))
    await runner.started.wait()

    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runner.killed


class HeldSSH:
    def __init__(self):
        self.result = asyncio.get_running_loop().create_future()
        self.called = asyncio.Event()

    async def run(self, cmd, input=None):
        self.called.set()
        return await self.result


@pytest.mark.asyncio
async def test_a_cancel_that_lands_as_an_ssh_command_completes_is_not_lost():
    ssh = HeldSSH()
    task = asyncio.ensure_future(SSHCommandRunner(ssh).run("true"))
    await ssh.called.wait()

    ssh.result.set_result(SimpleNamespace(exit_status=0, stdout="", stderr=""))
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


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
        (GpuFaultProbeCheck(is_post_rental_recheck=True), {"gpu": {"count": 1}}),
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
    """Each card's matmul never ends and keeps its session; like sshd's MaxSessions 10, a command that would
    open an 11th session at once is refused."""

    MAX_SESSIONS = 10

    def __init__(self, cards: int):
        self.calls: list[str] = []
        self.refused: list[str] = []
        self.cards = cards
        self.open = 0
        self.started = asyncio.Event()

    async def run(self, cmd, **_):
        self.calls.append(cmd)
        if "kill -KILL" in cmd:
            if self.open + 1 > self.MAX_SESSIONS:
                self.refused.append(cmd)
                raise ConnectionError("open failed: administratively prohibited")
            self.open += 1
            await asyncio.sleep(0.01)  # the kill's own session, held while it runs
            self.open -= 1
            return MagicMock(exit_status=0, stdout="")
        self.open += 1  # a cancelled card run keeps its session until its own command ends
        if sum("kill -KILL" not in c for c in self.calls) == self.cards:
            self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_cancelled_all_cards_matmul_kills_every_card_run_in_one_session():
    """review 4523441629: on an 8-card host the live card runs leave two sessions, so one kill carries every card."""
    from services.matrix_validation_service import ValidationService

    service = ValidationService.__new__(ValidationService)
    shell = BlockingCardShell(cards=8)
    executor = MagicMock(python_path="/usr/bin/python3")
    runs = asyncio.ensure_future(service._run_cards(shell, executor, "/tmp/s.py", [MagicMock(params="params")] * 8))
    await asyncio.wait_for(shell.started.wait(), 5)

    runs.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runs

    card_runs = [c for c in shell.calls if "kill -KILL" not in c]
    kills = [c for c in shell.calls if "kill -KILL" in c]
    tags = {re.search(r"lium-probe-[0-9a-f]+", c).group(0) for c in card_runs}
    assert len(tags) == 8 and len(kills) == 1 and shell.refused == []
    assert all(tag in kills[0] for tag in tags)


def test_the_post_rental_recheck_stops_on_a_rent():
    from services.task.checks.post_rental_gpu_recheck import PostRentalGpuRecheckCheck

    assert PostRentalGpuRecheckCheck.stops_on_rent


# review 6038245162 (taiberium, 24 Nov)


@pytest.mark.asyncio
async def test_a_create_stamped_between_the_rented_list_read_and_the_task_start_stops_the_run():
    """The stamp is compared with the rented list's read time, not the task start."""
    redis = FakeRedis()
    listed = UNRENTED.model_copy(update={"read_at": time.time() - 120})
    ctx = make_ctx(at_start=listed, redis=redis)
    stamp(redis, time.time() - 60)  # after the list was read, before this task started

    result = await RentedMidCycleCheck().run(ctx)

    assert not result.passed and result.event.reason_code == Msg.RENTED_SINCE_CYCLE_START.reason


class FailsAsTheRentLandsStep(HeavyStep):
    """A watched step that fails before the next rent poll, as a VerifyX run failing in the poll's last interval."""

    async def run(self, ctx):
        mark(self.redis, "pod-new")
        return CheckResult(passed=False, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={}))


@pytest.mark.asyncio
async def test_a_watched_step_that_fails_as_a_create_lands_is_read_once_more_and_gives_no_verdict(monkeypatch):
    monkeypatch.setattr(pipeline_module, "RENT_POLL_SECONDS", 60)
    ctx = make_ctx()
    step = FailsAsTheRentLandsStep(ctx.services.redis.redis)

    ok, events, _ = await asyncio.wait_for(Pipeline([step], AsyncMock()).run(ctx), 5)

    assert not ok and events[-1].reason_code == Msg.RENTED_SINCE_CYCLE_START.reason


def test_a_hold_for_a_create_mid_cycle_keeps_no_miner_active():
    """A provider can start a create on his own node, so that hold earns no eligibility floor."""
    from services.executor_rollout import session_kept_hotkeys, withhold_renter_gpu_reset_verdicts

    held = _failed("executor-123", Msg.RENTED_SINCE_CYCLE_START.reason)
    _, withheld = withhold_renter_gpu_reset_verdicts({"5Miner": [held]})

    assert len(withheld) == 1 and session_kept_hotkeys(withheld) == set()


@pytest.mark.asyncio
async def test_the_port_check_container_removal_is_bounded_on_a_stuck_docker_daemon():
    import asyncssh

    from services.executor_connectivity.container_runner import CLEANUP_TIMEOUT_SECONDS, ContainerRunner

    seen = {}

    class StuckDocker:
        async def run(self, command, **kwargs):
            seen.update(kwargs)
            raise asyncssh.TimeoutError(None, None, None, None, None, None, None, None)

    await ContainerRunner().cleanup(StuckDocker(), "nc-test")

    assert seen == {"timeout": CLEANUP_TIMEOUT_SECONDS}


def test_pid_tagged_names_its_parts():
    from services.task.runner import pid_tagged

    tagged = pid_tagged("sleep 1")

    assert "sleep 1" in tagged.command and tagged.kill_command.startswith("p=$(cat ")
