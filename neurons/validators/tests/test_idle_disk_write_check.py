"""DAH-5320: IdleDiskWriteCheck, the daily shadow disk write on idle and filler nodes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.task.checks import idle_disk_write as module
from services.task.checks import task_time
from services.task.checks.idle_disk_write import IdleDiskWriteCheck
from services.task.runner import SSHCommandResult
from tests.helpers import build_services, build_state, default_executor, make_context

EXECUTOR = default_executor()
pytestmark = pytest.mark.asyncio
KEY = f"idle_disk_write:miner-hotkey:{EXECUTOR.uuid}"


def snap(disk_mb_per_s: float) -> str:
    """Two snapshots 2 s apart of the docker root disk: sectors moved at disk_mb_per_s."""
    sectors = int(disk_mb_per_s * 1e6 * 2 / 512)
    return (
        "lium_tp_device name=nvme0n1\n"
        "lium_tp t=0.00 rx=0 tx=0 io=nvme0n1:0,\n"
        f"lium_tp t=2.00 rx=0 tx=0 io=nvme0n1:{sectors},\n"
    )


def write(mb_per_s: float) -> str:
    """The dd line: 256 MiB at mb_per_s."""
    n = 256 * 1024 * 1024
    return f"lium_disk bytes={n} start=0.00 end={n / 1e6 / mb_per_s:.2f} rc=0\n"


IDLE = snap(1)
POD = RentedExecutorsResponse(
    executors={
        EXECUTOR.uuid: RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address=EXECUTOR.address,
            executor_ip_port=str(EXECUTOR.port),
            pods=[RentedPod(pod_id="p1", container_name="pod_p1")],
        )
    }
)
FILLER = RentedExecutorsResponse(executors={}, all_filler_containers_by_executor={EXECUTOR.uuid: ["filler_1"]})


class FakeRunner:
    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.commands: list[str] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):  # noqa: ASYNC109
        self.commands.append(cmd)
        now = datetime.now(UTC)
        out = self.answers.pop(0) if self.answers else ""
        return SSHCommandResult(
            command="", command_id="c", exit_code=0, stdout=out, stderr="", duration_ms=10,
            started_at=now, finished_at=now, success=True,
        )


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.renting_in_progress = AsyncMock(return_value=False)

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


class Clock:
    now = 1_790_000_000.0
    mono = 0.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.mono


@contextmanager
def flags(enabled=True):
    fake = SimpleNamespace(
        IDLE_DISK_WRITE_CHECK_ENABLED=enabled,
        RENTED_THROUGHPUT_DISK_MIN_MB_PER_S=50.0,
        RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S=200.0,
        IDLE_DISK_WRITE_FILLER_BUSY_MB_PER_S=5.0,
        RENTED_THROUGHPUT_INTERVAL_HOURS=3.0,
        JOB_TIME_OUT=900,
    )
    clock = Clock()
    with (
        patch.object(module, "settings", fake),
        patch.object(module, "time", clock),
        patch.object(task_time, "time", clock),
    ):
        yield clock


def make_ctx(clock, *answers: str, rented_data=None, fresh=None, due=True):
    runner, redis = FakeRunner(*answers), FakeRedis()
    ctx = make_context(
        state=build_state(rented_data=rented_data, sysbox_runtime=True, gpu_model="NVIDIA B200"),
        runner=runner,
        services=build_services(redis=redis),
    )
    clock.mono = ctx.started_at_monotonic
    ctx.services.backend.get_rented_executors_now.return_value = fresh or RentedExecutorsResponse(executors={})
    if due:
        redis.store[KEY] = str(clock.now - 2 * 86400)
    return ctx, runner, redis


def writes(runner) -> int:
    return sum("lium_disk" in c and "conv=fsync" in c for c in runner.commands)


@pytest.mark.parametrize(("mb_per_s", "verdict"), [(12.8, "fail"), (400, "pass")])
async def test_logs_the_write_speed_card_and_provider_and_always_passes(caplog, mb_per_s, verdict):
    with flags() as clock, caplog.at_level("INFO"):
        ctx, runner, _ = make_ctx(clock, IDLE, write(mb_per_s))
        res = await IdleDiskWriteCheck().run(ctx)
    assert res.passed and not IdleDiskWriteCheck.fatal and writes(runner) == 1
    extra = next(r for r in caplog.records if r.getMessage() == "IDLE_DISK_WRITE").msg.extra
    assert extra["shadow_verdict"] == verdict and extra["mb_per_s"] == pytest.approx(mb_per_s, rel=0.01)
    assert extra["gpu_model"] == "NVIDIA B200" and extra["miner_hotkey"] == "miner-hotkey" and not extra["enforced"]


async def test_measures_a_filler_node_but_not_while_its_docker_root_disk_is_busy():
    with flags() as clock:
        ctx, runner, _ = make_ctx(clock, IDLE, write(400), rented_data=FILLER)
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 1

        ctx, runner, redis = make_ctx(clock, snap(300), rented_data=FILLER)
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 0 and len(runner.commands) == 1  # the 2 s read only
        ctx.services.backend.get_rented_executors_now.assert_not_called()
        assert float(redis.store[KEY]) < clock.now - 86400  # not stamped: the next cycle tries again


async def test_a_filler_writing_under_the_rented_busy_line_still_skips_the_probe():
    with flags() as clock:
        ctx, runner, _ = make_ctx(clock, snap(20), rented_data=FILLER)
        res = await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 0 and res.event.what_we_saw["reason"] == "disk busy (20 MB/s)"

        ctx, runner, _ = make_ctx(clock, snap(20), write(400))  # no filler: 20 MB/s is the provider's, not busy
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 1


async def test_the_last_reading_goes_to_the_backend_every_cycle_until_the_next():
    """DAH-5385: the backend keeps only the latest specs, so a day without a probe still carries the reading."""
    with flags() as clock:
        ctx, _, redis = make_ctx(clock, due=False)
        assert not (await IdleDiskWriteCheck().run(ctx)).updates  # never measured: nothing sent

        ctx, _, redis = make_ctx(clock, IDLE, write(13))
        res = await IdleDiskWriteCheck().run(ctx)
        assert res.updates["state"].specs[module.SPECS_KEY] == pytest.approx(13, rel=0.01)

        ctx2, runner, _ = make_ctx(clock, due=False)  # not due: the stored reading still rides along
        ctx2.services.redis.store.update(redis.store)
        res = await IdleDiskWriteCheck().run(ctx2)
        assert writes(runner) == 0
        assert res.updates["state"].specs[module.SPECS_KEY] == pytest.approx(13, rel=0.01)


async def test_a_stored_reading_is_not_refreshed_on_read_so_it_expires_after_its_probe():
    with flags() as clock:
        ctx, _, redis = make_ctx(clock, due=False)
        key = f"{module._READING_PREFIX}:miner-hotkey:{EXECUTOR.uuid}"
        redis.store[key] = f"13.0:{clock.now - 3600}"
        redis.set = AsyncMock()
        res = await IdleDiskWriteCheck().run(ctx)
    specs = res.updates["state"].specs
    assert specs[module.SPECS_KEY] == 13.0 and specs[module.SPECS_MEASURED_AT_KEY] == clock.now - 3600
    assert all(c.args[0] != key for c in redis.set.await_args_list)


async def test_the_reading_is_sent_even_when_redis_fails_to_keep_it():
    with flags() as clock:
        ctx, runner, redis = make_ctx(clock, IDLE, write(13))
        real_set = redis.set

        async def set_(k, v, ex=None):
            if k.startswith(module._READING_PREFIX):
                raise ConnectionError
            await real_set(k, v, ex)

        redis.set = set_
        res = await IdleDiskWriteCheck().run(ctx)
    assert writes(runner) == 1
    assert res.updates["state"].specs[module.SPECS_KEY] == pytest.approx(13, rel=0.01)
    assert res.updates["state"].specs[module.SPECS_MEASURED_AT_KEY] == clock.now
    assert res.event.what_we_saw["mb_per_s_sent"] == pytest.approx(13, rel=0.01)


async def test_a_filler_started_since_cycle_start_keeps_its_download_from_being_measured():
    """lio#1953 review: the snapshot has no filler, the fresh list has one; 20 MB/s is its download."""
    with flags() as clock:
        ctx, runner, _ = make_ctx(clock, snap(20), write(400), fresh=FILLER)
        res = await IdleDiskWriteCheck().run(ctx)
    assert writes(runner) == 0 and res.event.what_we_saw["reason"] == "disk busy (20 MB/s)"
    assert res.event.what_we_saw["disk_busy_mb_per_s"] == pytest.approx(20, rel=0.01)


async def test_a_write_that_timed_out_with_nothing_written_sends_no_reading():
    """lio#1953 review: bytes=0 parses as 0.0 MB/s, which the backend would hold as a slow disk for a day."""
    with flags() as clock:
        ctx, runner, redis = make_ctx(clock, IDLE, "lium_disk bytes=0 start=0.00 end=150.00 rc=124\n")
        res = await IdleDiskWriteCheck().run(ctx)
    assert writes(runner) == 1 and not res.updates
    assert not any(k.startswith(module._READING_PREFIX) for k in redis.store)


async def test_skips_a_node_rented_since_cycle_start():
    with flags() as clock:
        ctx, runner, _ = make_ctx(clock, IDLE, fresh=POD)
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 0

        ctx, runner, redis = make_ctx(clock, IDLE)
        redis.renting_in_progress.return_value = True
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 0


async def test_once_a_day_at_the_node_phase():
    with flags() as clock:
        ctx, runner, redis = make_ctx(clock, due=False)
        with patch.object(module, "_node_fraction_of_day", lambda c: 1.0):
            await IdleDiskWriteCheck().run(ctx)  # first sight: stamped, nothing run
        assert not runner.commands and KEY in redis.store
        runner.answers = [IDLE, write(400)]
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 1
        clock.now += 23 * 3600
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 1
        clock.now += 3600
        runner.answers = [IDLE, write(400)]
        await IdleDiskWriteCheck().run(ctx)
        assert writes(runner) == 2


@pytest.mark.parametrize(("elapsed", "runs"), [(500, 1), (520, 0)])
async def test_runs_only_while_its_own_reads_fit(elapsed, runs):
    with flags() as clock:
        ctx, runner, _ = make_ctx(clock, IDLE, write(400))
        clock.mono = ctx.started_at_monotonic + elapsed  # 900 - 120 - elapsed against 30 + 30 + 150 + 60 slack
        await IdleDiskWriteCheck().run(ctx)
    assert writes(runner) == runs


async def test_a_failure_or_the_flag_off_never_changes_the_result():
    with flags() as clock:
        ctx, runner, redis = make_ctx(clock)
        redis.store[KEY] = "not-a-number"
        assert (await IdleDiskWriteCheck().run(ctx)).passed and not runner.commands
    with flags(enabled=False) as clock:
        ctx, runner, _ = make_ctx(clock)
        assert (await IdleDiskWriteCheck().run(ctx)).passed and not runner.commands


async def test_an_unresolvable_docker_root_disk_is_its_own_reason_and_stamps_the_day():
    with flags() as clock:
        ctx, runner, redis = make_ctx(clock, "lium_tp t=0.00 io=\nlium_tp t=2.00 io=\n")
        res = await IdleDiskWriteCheck().run(ctx)
    assert writes(runner) == 0 and float(redis.store[KEY]) == clock.now
    assert res.event.what_we_saw["reason"] == "docker root disk not resolvable"
