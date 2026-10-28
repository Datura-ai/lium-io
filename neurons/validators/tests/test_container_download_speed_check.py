"""CONTAINER_DOWNLOAD_SLOW: the host-side speed test read a node at 297-588 Mbps while a renter's container on it
fetched from PyPI at under 1 Mbps. The check measures from inside a container, the way a renter downloads."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.task.checks import container_download_speed as module
from services.task.checks.container_download_speed import (
    ContainerDownloadSustainedCheck,
    CAP_BYTES,
    ContainerDownloadSpeedCheck,
    download_speed_command,
    parse_download_speed,
)
from services.task.messages import ContainerDownloadSpeedMessages as Msg
from services.task.runner import SSHCommandResult
from tests.helpers import build_services, build_state, default_executor, make_context

EXECUTOR = default_executor()
pytestmark = pytest.mark.asyncio


def line(ep: str, got: int, secs: float, rc: int, err: str = "") -> str:
    return f"lium_dl ep={ep} bytes={got} start=100.00 end={100 + secs:.2f} rc={rc}\nlium_dl_err ep={ep} {err}\n"


RENTALS = "lium_dl_net name=lium-rentals\n"
BRIDGE = "lium_dl_net name=bridge\n"  # a node with no rental yet has no lium-rentals network
# 16 MiB in 1.0 s = 134 Mbps on both
FAST = RENTALS + line("pypi", CAP_BYTES, 1.0, 0) + line("hf", CAP_BYTES, 1.0, 0)
# the reported node: PyPI crawls, 1.8 MB in the 30 s bound
SLOW = RENTALS + line("pypi", 1_800_000, 30.0, 143) + line("hf", CAP_BYTES, 1.0, 0)
SLOW_ON_BRIDGE = SLOW.replace(RENTALS, BRIDGE)
DNS_DOWN = (
    RENTALS + line("pypi", 0, 0.05, 1, "wget: bad address 'files.pythonhosted.org'") + line("hf", CAP_BYTES, 1.0, 0)
)


def result(stdout: str = "", *, exit_code: int = 0, error_type: str | None = None) -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command="",
        command_id="c",
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_ms=10,
        started_at=now,
        finished_at=now,
        success=exit_code == 0 and error_type is None,
        error_type=error_type,
        error_message="timed out" if error_type else None,
    )


class FakeRunner:
    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.commands: list[str] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        return result(self.answers.pop(0) if self.answers else FAST)


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def pttl(self, key):
        return -1 if key in self.store else -2


class Clock:
    now = 1_790_000_000.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return time.monotonic()


@contextmanager
def flags(*, reachable=True, sustained=False):
    fake = SimpleNamespace(
        CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED=True,
        CONTAINER_DOWNLOAD_SPEED_MIN_MBPS=10.0,
        CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS=6.0,
        CONTAINER_DOWNLOAD_SUSTAINED_ENABLED=sustained,
        CONTAINER_DOWNLOAD_SUSTAINED_URL="https://cdn.example/public/big.bin",
        CONTAINER_DOWNLOAD_SUSTAINED_MAX_BYTES=2 * 1024**3,
        RENTED_THROUGHPUT_BUSY_MBPS=200.0,
        JOB_TIME_OUT=900,
    )
    clock = Clock()

    async def reaches(url):
        return reachable

    with (
        patch.object(module, "settings", fake),
        patch.object(module, "time", clock),
        patch.object(module, "validator_reaches", reaches),
        # the node's phase is now, so its first cycle measures
        patch.object(module, "phase_offset_seconds", lambda uuid: 6 * 3600),
    ):
        yield clock


def make_ctx(*answers: str, redis=None, rented_data=None):
    runner = FakeRunner(*answers)
    redis = redis or FakeRedis()
    ctx = make_context(
        state=build_state(rented_data=rented_data, sysbox_runtime=True),
        runner=runner,
        services=build_services(redis=redis),
    )
    return ctx, runner, redis


def test_defaults_are_bounded():
    fields = type(module.settings).model_fields
    assert fields["CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED"].default is True
    assert fields["CONTAINER_DOWNLOAD_SPEED_MIN_MBPS"].default == 10.0
    assert fields["CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS"].default == 6.0


def test_command_runs_in_a_container_with_the_renter_runtime_and_a_size_cap():
    cmd = download_speed_command(True)
    # the rentals bridge a pod runs on, or the default bridge before the node's first rental
    assert "docker network inspect lium-rentals" in cmd and "net=lium-rentals" in cmd and "net=bridge" in cmd
    assert '/usr/bin/docker run --rm --network "$net" --runtime=sysbox-runc ' in cmd
    assert "Range" not in cmd and f"head -c {CAP_BYTES}" in cmd  # busybox wget fails on 206
    assert all(url in cmd for _, url in module.ENDPOINTS)  # both pinned files, in full
    assert "--runtime" not in download_speed_command(False)


def test_parse_reports_the_network_on_every_reading_failed_or_not():
    assert [r.network for r in parse_download_speed(DNS_DOWN)] == ["lium-rentals", "lium-rentals"]
    assert parse_download_speed(DNS_DOWN)[0].error == "dns_failed"
    assert parse_download_speed(FAST.replace(RENTALS, ""))[0].network is None


def test_parse_measures_mbps_and_names_errors():
    fast = parse_download_speed(FAST)
    assert [r.endpoint for r in fast] == ["pypi", "hf"]
    assert fast[0].mbps == pytest.approx(CAP_BYTES * 8 / 1e6, rel=1e-3) and fast[0].error is None

    slow = parse_download_speed(SLOW)[0]
    assert slow.error is None and slow.mbps == pytest.approx(0.48)  # a timed-out read is the slow reading

    assert parse_download_speed(DNS_DOWN)[0].error == "dns_failed"
    assert parse_download_speed(line("pypi", 0, 30, 143))[0].error == "timeout"
    assert parse_download_speed(line("pypi", 0, 1, 1, "wget: server returned error: HTTP/1.1 403"))[0].error == "http_error"
    assert parse_download_speed(line("pypi", 0, 1, 1, "wget: can't connect: Connection refused"))[0].error == "unreachable"
    # head stopped the read at the cap and wget died of SIGPIPE
    assert parse_download_speed(line("pypi", CAP_BYTES, 2.0, 1))[0].mbps is not None
    assert parse_download_speed(line("hf", CAP_BYTES, 1.7, 141))[1].mbps is not None  # staging node, 20 Oct
    assert parse_download_speed("")[1].error == "no_output"


async def test_fast_passes_and_logs_the_reading(caplog):
    with flags(), caplog.at_level("INFO"):
        ctx, runner, redis = make_ctx(FAST)
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OK.reason
    assert len(runner.commands) == 1
    record = next(r for r in caplog.records if "CONTAINER_DOWNLOAD_SPEED" in r.getMessage())
    assert record is not None


async def test_slow_twice_fails_the_node():
    with flags() as clock:
        ctx, runner, redis = make_ctx(SLOW, SLOW)
        check = ContainerDownloadSpeedCheck()
        first = await check.run(ctx)
        assert first.passed and first.event.reason_code == Msg.SLOW_ONCE.reason
        clock.now += 31 * 60  # the confirming reading comes after RETRY, not an interval later
        second = await check.run(ctx)
        assert len(runner.commands) == 2
        assert not second.passed and second.event.reason_code == Msg.SLOW.reason
        # the finding stands between readings
        clock.now += 60
        third = await check.run(ctx)
        assert len(runner.commands) == 2 and third.passed is False


async def test_a_streak_with_a_default_bridge_reading_is_never_enforced():
    for answers in ((SLOW_ON_BRIDGE, SLOW_ON_BRIDGE), (SLOW_ON_BRIDGE, SLOW)):
        with flags() as clock:
            ctx, runner, redis = make_ctx(*answers)
            check = ContainerDownloadSpeedCheck()
            await check.run(ctx)
            clock.now += 31 * 60
            second = await check.run(ctx)
        assert second.passed and second.event.reason_code == Msg.SLOW_OBSERVED.reason


async def test_rentals_failures_after_a_bridge_failure_are_enforced():
    with flags() as clock:
        ctx, runner, redis = make_ctx(SLOW_ON_BRIDGE, SLOW, SLOW)
        check = ContainerDownloadSpeedCheck()
        for _ in range(2):
            await check.run(ctx)
            clock.now += 31 * 60
        third = await check.run(ctx)
    assert not third.passed and third.event.reason_code == Msg.SLOW.reason


async def test_a_streak_saved_without_the_rentals_count_is_not_enforced():
    with flags() as clock:
        redis = FakeRedis()
        # a failing reading saved by an earlier release: no failing_readings_on_rentals_network
        redis.store[f"{module._REDIS_PREFIX}:{EXECUTOR.uuid}"] = json.dumps(
            {"at": clock.now - 31 * 60, "failures_in_a_row": 1, "readings": [], "measured": True}
        )
        ctx, runner, _ = make_ctx(SLOW, redis=redis)
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.SLOW_OBSERVED.reason


async def test_slow_then_fast_resets():
    with flags() as clock:
        ctx, runner, redis = make_ctx(SLOW, FAST)
        check = ContainerDownloadSpeedCheck()
        await check.run(ctx)
        clock.now += 31 * 60
        res = await check.run(ctx)
    assert res.passed and res.event.reason_code == Msg.OK.reason


async def test_endpoint_unreachable_counts_only_when_the_validator_reaches_it():
    with flags(reachable=True):
        ctx, _, redis = make_ctx(DNS_DOWN)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason
        assert res.event.what_we_saw["endpoints"][0]["error"] == "dns_failed"
    with flags(reachable=False):
        ctx, _, redis = make_ctx(DNS_DOWN)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.passed and res.event.reason_code == Msg.UNMEASURED.reason
        assert json.loads(next(iter(redis.store.values())))["failures_in_a_row"] == 0


async def test_container_that_did_not_run_is_no_verdict():
    with flags():
        ctx, _, _ = make_ctx("docker: Error response from daemon: unknown runtime")
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.UNMEASURED.reason


async def test_cadence_once_per_interval_and_skips_rented_and_filler():
    with flags() as clock:
        ctx, runner, _ = make_ctx(FAST, FAST)
        check = ContainerDownloadSpeedCheck()
        await check.run(ctx)
        clock.now += 5 * 3600
        res = await check.run(ctx)
        assert len(runner.commands) == 1 and res.event.reason_code == Msg.SKIPPED.reason
        clock.now += 3600
        await check.run(ctx)
        assert len(runner.commands) == 2

        pod = RentedExecutorsResponse(
            executors={
                EXECUTOR.uuid: RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address=EXECUTOR.address,
                    executor_ip_port=str(EXECUTOR.port),
                    pods=[RentedPod(pod_id="p1", container_name="pod_p1")],
                )
            }
        )
        filler = RentedExecutorsResponse(
            executors={}, all_filler_containers_by_executor={EXECUTOR.uuid: ["filler_1"]}
        )
        for rented_data in (pod, filler):
            ctx, runner, _ = make_ctx(rented_data=rented_data)
            res = await check.run(ctx)
            assert not runner.commands and res.event.reason_code == Msg.SKIPPED.reason


async def test_first_sight_waits_for_the_node_phase():
    with flags() as clock, patch.object(module, "phase_offset_seconds", lambda uuid: 3600.0):
        ctx, runner, _ = make_ctx()
        check = ContainerDownloadSpeedCheck()
        await check.run(ctx)
        assert not runner.commands
        clock.now += 5 * 3600
        await check.run(ctx)
        assert len(runner.commands) == 1


async def test_a_node_copying_the_uuid_cannot_fail_the_owner():
    with flags() as clock:
        redis = FakeRedis()
        copycat = make_context(
            state=build_state(sysbox_runtime=True),
            runner=FakeRunner(SLOW, SLOW),
            services=build_services(redis=redis),
            miner_hotkey="copycat-hotkey",
        )
        check = ContainerDownloadSpeedCheck()
        await check.run(copycat)
        clock.now += 31 * 60
        assert not (await check.run(copycat)).passed

        ctx, _, _ = make_ctx(FAST, redis=redis)
        res = await check.run(ctx)
    assert res.passed and res.event.reason_code == Msg.OK.reason


async def test_a_deploy_keeps_the_open_streak_saved_under_the_uuid_alone():
    with flags() as clock:
        redis = FakeRedis()
        redis.store[f"{module._REDIS_PREFIX}:{EXECUTOR.uuid}"] = json.dumps(
            {
                "at": clock.now - 31 * 60,
                "failures_in_a_row": 1,
                "readings": [],
                "measured": True,
                "failing_readings_on_rentals_network": 1,
            }
        )
        ctx, _, _ = make_ctx(SLOW, redis=redis)
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert not res.passed and res.event.reason_code == Msg.SLOW.reason


# DAH-5320 sustained phase: cumulative received bytes every 10 s, then each stream's exit


def sustained(mbps: list[float], rcs=("0", "0", "0", "0"), alive_after: int | None = None) -> str:
    rx, out = 0, [f"lium_dls t=0.00 rx=0 alive={4 if mbps else 0}"]
    for i, m in enumerate(mbps, 1):
        rx += int(m * 1e6 / 8 * 10)
        alive = 0 if alive_after is not None and i >= alive_after else 4
        out.append(f"lium_dls t={i * 10}.00 rx={rx} alive={alive}")
    out += [f"lium_dls_rc s={s} rc={rc} err=" for s, rc in enumerate(rcs, 1)]
    return RENTALS + "\n".join(out) + "\n"


def nic(mbps: float) -> str:
    """Two host NIC snapshots 2 s apart, as RentedThroughputCheck reads them."""
    return f"lium_tp t=0.00 rx=0 tx=0 io=\nlium_tp t=2.00 rx={int(mbps * 1e6 / 8 * 2)} tx=0 io=\n"


IDLE_NIC = nic(5)
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


def due_ctx(clock, *answers: str, rented_data=None, fresh=None):
    """A node whose daily sustained read is due; the backend's re-read answers `fresh` (idle by default)."""
    ctx, runner, redis = make_ctx(*answers, rented_data=rented_data)
    redis.store[f"container_download_sustained:{ctx.miner_hotkey}:{EXECUTOR.uuid}"] = str(clock.now - 2 * 86400)
    redis.renting_in_progress = AsyncMock(return_value=False)
    ctx.services.backend.get_rented_executors_now.return_value = fresh or RentedExecutorsResponse(executors={})
    return ctx, runner, redis


def sustained_runs(runner) -> int:
    return sum("wget -q -O -" in c and "lium_dls" in c for c in runner.commands)


@pytest.mark.parametrize(
    ("mbps", "verdict"),
    [
        ([400, 380, 420, 410, 390, 405], "pass"),
        ([400, 90, 80, 95, 85, 400], "fail"),  # median 92.5
        ([900, 900, 0.2, 0.1, 0.0, 0.3], "fail"),  # 40 s stalled after a fast start
    ],
)
def test_sustained_verdict_on_the_window_series(mbps, verdict):
    reading = module.parse_sustained(sustained(mbps))
    assert reading.errors == [] and len(reading.windows) == 6
    assert module.sustained_verdict(reading.windows) == verdict


def test_sustained_stall_of_30_s_fails_and_20_s_does_not():
    assert module.sustained_verdict([500, 0.1, 0.1, 500, 500, 500]) == "pass"
    assert module.sustained_verdict([500, 500, 0.1, 0.1, 0.1, 500, 500]) == "fail"  # exactly 30 s


def test_sustained_byte_cap_reached_early_drops_the_idle_windows_and_a_stream_error_is_no_verdict():
    reading = module.parse_sustained(sustained([3000, 3000, 0, 0, 0, 0], alive_after=2))
    assert reading.windows == [3000.0, 3000.0] and not reading.errors
    assert module.sustained_verdict(reading.windows) == "pass"
    reading = module.parse_sustained(sustained([0.0], rcs=("1", "1", "short", "1"), alive_after=1))
    assert len(reading.errors) == 4  # e.g. the object answers 403, or is shorter than a chunk: not the node's fault
    assert module.parse_sustained(RENTALS).errors == ["no_output"]


def test_sustained_script_reads_chunks_up_to_the_byte_cap():
    script = module.sustained_script("https://cdn.example/x.bin", 2 * 1024**3)
    # 2 GiB in 32 MiB requests over 4 streams: 16 a stream, 64 a run
    assert f"head -c {module.CHUNK_BYTES}" in script and "[ $k -lt 16 ]" in script
    assert "Range" not in script  # busybox wget fails on a 206


async def test_sustained_runs_once_a_day_at_the_node_phase_and_never_changes_the_result(caplog):
    with flags(sustained=True) as clock, caplog.at_level("INFO"):
        ctx, runner, redis = make_ctx(IDLE_NIC, sustained([5, 5, 5, 5, 5, 5]))
        redis.renting_in_progress = AsyncMock(return_value=False)
        ctx.services.backend.get_rented_executors_now.return_value = RentedExecutorsResponse(executors={})
        check = ContainerDownloadSustainedCheck()
        key = f"container_download_sustained:{ctx.miner_hotkey}:{EXECUTOR.uuid}"
        with patch.object(module, "_phase_fraction", lambda uuid: 1.0):
            first = await check.run(ctx)  # first sight: due only at the node's phase
        assert not runner.commands and key in redis.store
        res = await check.run(ctx)  # due now: the shadow fail is logged, the check still passes
        assert sustained_runs(runner) == 1
        assert first.passed and res.passed and not check.fatal
        record = next(r for r in caplog.records if r.getMessage() == "CONTAINER_DOWNLOAD_SUSTAINED")
        extra = record.msg.extra
        assert extra["shadow_verdict"] == "fail" and extra["windows_mbps"] == [5.0] * 6 and not extra["enforced"]

        clock.now += 23 * 3600
        await check.run(ctx)
        assert sustained_runs(runner) == 1
        clock.now += 3600
        runner.answers = [IDLE_NIC, sustained([400] * 6)]
        await check.run(ctx)  # a day after the last one
        assert sustained_runs(runner) == 2


async def test_sustained_measures_a_filler_node_but_not_while_its_nic_is_busy():
    with flags(sustained=True) as clock:
        ctx, runner, _ = due_ctx(clock, IDLE_NIC, sustained([400] * 6), rented_data=FILLER)
        res = await ContainerDownloadSustainedCheck().run(ctx)
        assert sustained_runs(runner) == 1 and res.passed

        ctx, runner, redis = due_ctx(clock, nic(900), rented_data=FILLER)
        await ContainerDownloadSustainedCheck().run(ctx)
        assert sustained_runs(runner) == 0 and len(runner.commands) == 1  # the NIC read only
        # read first: a busy node, due every cycle, never pulls the backend's rented list
        ctx.services.backend.get_rented_executors_now.assert_not_called()
        # not stamped: the next cycle tries again
        assert float(redis.store[f"container_download_sustained:{ctx.miner_hotkey}:{EXECUTOR.uuid}"]) < clock.now


async def test_sustained_skips_a_node_rented_at_cycle_start_or_since():
    with flags(sustained=True) as clock:
        ctx, runner, _ = due_ctx(clock, rented_data=POD)
        await ContainerDownloadSustainedCheck().run(ctx)
        assert not runner.commands

        ctx, runner, _ = due_ctx(clock, IDLE_NIC, fresh=POD)  # rented after the cycle's snapshot
        await ContainerDownloadSustainedCheck().run(ctx)
        assert len(runner.commands) == 1 and sustained_runs(runner) == 0

        ctx, runner, redis = due_ctx(clock, IDLE_NIC)
        redis.renting_in_progress = AsyncMock(return_value=True)
        await ContainerDownloadSustainedCheck().run(ctx)
        assert sustained_runs(runner) == 0


@pytest.mark.parametrize(("elapsed", "runs"), [(590, 1), (610, 0)])
async def test_sustained_runs_only_while_its_own_reads_fit(elapsed, runs):
    with flags(sustained=True) as clock:
        ctx, runner, _ = due_ctx(clock, IDLE_NIC, sustained([400] * 6))
        # 900 - 120 - elapsed left against the 30 s NIC read, the 30 s re-read and the 120 s read
        with patch.object(clock, "monotonic", lambda: ctx.started_at_monotonic + elapsed):
            await ContainerDownloadSustainedCheck().run(ctx)
        assert sustained_runs(runner) == runs


async def test_sustained_failure_never_changes_the_result():
    with flags(sustained=True):
        redis = FakeRedis()
        ctx, runner, _ = make_ctx(redis=redis)
        redis.store[f"container_download_sustained:{ctx.miner_hotkey}:{EXECUTOR.uuid}"] = "not-a-number"
        res = await ContainerDownloadSustainedCheck().run(ctx)
    assert res.passed and not runner.commands


def test_sustained_default_byte_cap_is_512_mib():
    assert type(module.settings).model_fields["CONTAINER_DOWNLOAD_SUSTAINED_MAX_BYTES"].default == 512 * 1024**2
