"""CONTAINER_DOWNLOAD_SLOW: the host-side speed test read a node at 297-588 Mbps while a renter's container on it
fetched from PyPI at under 1 Mbps. The check measures from inside a container, the way a renter downloads."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from services.task.checks import container_download_speed as module
from services.task.checks.container_download_speed import (
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


class Clock:
    now = 1_790_000_000.0

    def time(self) -> float:
        return self.now


@contextmanager
def flags(*, reachable=True):
    fake = SimpleNamespace(
        CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED=True,
        CONTAINER_DOWNLOAD_SPEED_MIN_MBPS=10.0,
        CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS=6.0,
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
