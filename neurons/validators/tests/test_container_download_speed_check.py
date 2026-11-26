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
# the pod's upload: 16 MiB posted in 1.0 s = 134 Mbps
UP = line("cf_up", CAP_BYTES, 1.0, 0)
BRIDGE = "lium_dl_net name=bridge\n"  # a node with no rental yet has no lium-rentals network
# 16 MiB in 1.0 s = 134 Mbps on both
FAST = RENTALS + line("pypi", CAP_BYTES, 1.0, 0) + line("hf", CAP_BYTES, 1.0, 0) + UP
# a slow link: both endpoints crawl, 1.8 MB in the 30 s bound
SLOW = RENTALS + line("pypi", 1_800_000, 30.0, 143) + line("hf", 1_800_000, 30.0, 143) + UP
# only PyPI's CDN crawls while Hugging Face reads fast: the CDN path, not the node
PYPI_ONLY_SLOW = RENTALS + line("pypi", 1_800_000, 30.0, 143) + line("hf", CAP_BYTES, 1.0, 0) + UP
SLOW_ON_BRIDGE = SLOW.replace(RENTALS, BRIDGE)
DNS_DOWN = (
    RENTALS
    + line("pypi", 0, 0.05, 1, "wget: bad address 'files.pythonhosted.org'")
    + line("hf", CAP_BYTES, 1.0, 0)
    + UP
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


def member(hotkey: str, uuid: str, at: float, bad: int) -> str:
    """One fleet-window entry as the check writes it."""
    return json.dumps([hotkey, uuid, round(at, 3), bad])


class FleetRedis(FakeRedis):
    """FakeRedis with the endpoint fleet window, preloaded with other providers' readings."""

    def __init__(self, **slow_share: tuple[int, int]):
        super().__init__()
        self.fleet: dict[str, list[tuple[float, str]]] = {}
        for endpoint, (slow, total) in slow_share.items():
            for i in range(total):
                self.fleet.setdefault(endpoint, []).append(
                    (Clock.now - 60, member(f"5Other{i % 7}", f"node-{i}", Clock.now - 60, int(i < slow)))
                )

    async def record_endpoint_speed(self, endpoint, member, now, window_seconds):
        self.fleet.setdefault(endpoint, []).append((now, member))

    async def endpoint_speeds_since(self, endpoint, since):
        return [m for t, m in self.fleet.get(endpoint, []) if t >= since]


class Clock:
    now = 1_790_000_000.0

    def time(self) -> float:
        return self.now


@contextmanager
def flags(*, reachable=True, enforced=True):
    fake = SimpleNamespace(
        CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED=True,
        CONTAINER_DOWNLOAD_SPEED_MIN_MBPS=10.0,
        CONTAINER_UPLOAD_SPEED_MIN_MBPS=10.0,
        CONTAINER_SPEED_FLEET_RULE_ENFORCED=enforced,
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
    assert [r.network for r in parse_download_speed(DNS_DOWN)] == ["lium-rentals"] * 3
    assert parse_download_speed(DNS_DOWN)[0].error == "dns_failed"
    assert parse_download_speed(FAST.replace(RENTALS, ""))[0].network is None


def test_parse_measures_mbps_and_names_errors():
    fast = parse_download_speed(FAST)
    assert [r.endpoint for r in fast] == ["pypi", "hf", "cf_up"]
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


async def test_one_slow_cdn_while_the_other_endpoint_is_fast_passes():
    # DAH-5979: PyPI's CDN alone crawled on unrelated networks at once while Hugging Face read 20-80 Mbps
    with flags() as clock:
        burst = FleetRedis(pypi=(15, 40), hf=(0, 40), cf_up=(0, 40))
        ctx, runner, _ = make_ctx(PYPI_ONLY_SLOW, PYPI_ONLY_SLOW, redis=burst)
        check = ContainerDownloadSpeedCheck()
        res = await check.run(ctx)
        assert res.passed and res.event.reason_code == Msg.OK.reason
        clock.now += 31 * 60
        res = await check.run(ctx)  # a pass waits the full interval: no confirming retry
        assert res.passed and len(runner.commands) == 1


async def test_one_slow_cdn_that_only_this_provider_reads_slow_fails_the_node():
    # one provider's pods read PyPI slow on many readings while HF reads fast and other providers read PyPI fast
    with flags() as clock:
        fleet = FleetRedis(pypi=(2, 40), hf=(0, 40), cf_up=(0, 40))
        ctx, runner, redis = make_ctx(PYPI_ONLY_SLOW, PYPI_ONLY_SLOW, redis=fleet)
        check = ContainerDownloadSpeedCheck()
        first = await check.run(ctx)
        assert first.passed and first.event.reason_code == Msg.SLOW_ONCE.reason
        clock.now += 31 * 60
        second = await check.run(ctx)
        assert second.passed is False and second.event.reason_code == Msg.SLOW.reason
        assert [r["endpoint"] for r in second.event.what_we_saw["endpoints"] if r["mbps"] < 10] == ["pypi"]
        # this node's own readings went into the window under its hotkey
        mine = [json.loads(m) for _, m in redis.fleet["pypi"] if json.loads(m)[0] == ctx.miner_hotkey]
        assert len(mine) == 2 and all(m[3] == 1 for m in mine)


async def test_the_fleet_control_leaves_out_this_providers_own_nodes():
    with flags():
        # every slow PyPI reading in the window is this provider's: the rest of the fleet reads it fast
        fleet = FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(0, 30))
        ctx, _, _ = make_ctx(PYPI_ONLY_SLOW, redis=fleet)
        for i in range(30):
            fleet.fleet["pypi"].append((Clock.now - 30, f"{ctx.miner_hotkey}|sibling-{i}|0|1"))
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason


async def test_too_few_fleet_readings_keep_the_excuse():
    with flags():
        ctx, _, _ = make_ctx(PYPI_ONLY_SLOW, redis=FleetRedis(pypi=(0, 5), hf=(0, 5), cf_up=(0, 5)))
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.passed and res.event.reason_code == Msg.OK.reason


@pytest.mark.parametrize("enforced", [True, False])
async def test_the_instructions_name_slow_downloads_only_when_the_upload_was_fast(enforced):
    with flags(enforced=enforced) as clock:
        ctx, _, _ = make_ctx(SLOW, SLOW)
        check = ContainerDownloadSpeedCheck()
        first = await check.run(ctx)
        clock.now += 31 * 60
        second = await check.run(ctx)
    for res in (first, second):
        assert "downloads from PyPI or Hugging Face too slowly" in res.event.remediation
        assert "upload" not in res.event.remediation


async def test_the_instructions_name_the_upload_when_only_the_upload_was_slow():
    slow_up = FAST.replace(UP, line("cf_up", 0, 30.0, 143))
    with flags():
        ctx, _, _ = make_ctx(slow_up, redis=FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(1, 30)))
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.event.reason_code == Msg.SLOW_ONCE.reason
    assert "uploads too slowly" in res.event.remediation and "PyPI" not in res.event.remediation


async def test_a_slow_upload_fails_unless_the_fleet_uploads_slow_too():
    slow_up = FAST.replace(UP, line("cf_up", 0, 30.0, 143))  # the upload did not finish in time: an upper-bound slow reading
    with flags():
        ctx, _, _ = make_ctx(slow_up, redis=FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(1, 30)))
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason
    with flags():
        ctx, _, _ = make_ctx(slow_up, redis=FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(20, 30)))
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.OK.reason


def test_parse_an_upload_that_did_not_finish_is_an_upper_bound_reading():
    up = parse_download_speed(line("cf_up", 0, 30.0, 143))[2]
    assert up.error is None and up.mbps == pytest.approx(CAP_BYTES * 8 / 30 / 1e6) and "upper bound" in up.detail
    assert parse_download_speed(line("cf_up", CAP_BYTES, 2.0, 0))[2].mbps == pytest.approx(CAP_BYTES * 8 / 2 / 1e6)
    assert parse_download_speed(line("cf_up", 0, 0.1, 1, "wget: bad address 'speed.cloudflare.com'"))[2].error == (
        "dns_failed"
    )


def test_the_upload_posts_the_capped_body_to_the_speed_test_sink():
    cmd = download_speed_command(True)
    assert "--post-file=/tmp/u" in cmd and "https://speed.cloudflare.com/__up" in cmd
    assert f"head -c {CAP_BYTES} /dev/zero" in cmd


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


@pytest.mark.parametrize(
    "upload",
    [
        line("cf_up", 0, 0.1, 1, "wget: server returned error: HTTP/1.1 403 Forbidden"),
        line("cf_up", 0, 0.1, 1, "wget: server returned error: HTTP/1.1 429 Too Many Requests"),
    ],
)
@pytest.mark.parametrize("enforced", [True, False])
async def test_an_upload_the_receiver_refuses_keeps_the_streak_only_when_enforced(upload, enforced):
    fast_downloads = RENTALS + line("pypi", CAP_BYTES, 1.0, 0) + line("hf", CAP_BYTES, 1.0, 0) + upload
    with flags(enforced=enforced) as clock:
        ctx, runner, redis = make_ctx(SLOW, fast_downloads)
        check = ContainerDownloadSpeedCheck()
        await check.run(ctx)
        assert json.loads(next(iter(redis.store.values())))["failures_in_a_row"] == 1
        clock.now += 31 * 60
        res = await check.run(ctx)
    saved = json.loads(next(iter(redis.store.values())))
    if enforced:
        assert res.passed and res.event.reason_code == Msg.SLOW_ONCE.reason
        assert saved["failures_in_a_row"] == 1 and saved["measured"] is False
    else:
        # record-only the upload decides nothing: fast downloads clear the failure, as on main
        assert res.passed and res.event.reason_code == Msg.OK.reason
        assert saved["failures_in_a_row"] == 0 and saved["measured"] is True


async def test_fast_downloads_and_upload_clear_a_failure():
    with flags() as clock:
        ctx, runner, redis = make_ctx(SLOW, FAST)
        check = ContainerDownloadSpeedCheck()
        await check.run(ctx)
        clock.now += 31 * 60
        res = await check.run(ctx)
    assert res.passed and res.event.reason_code == Msg.OK.reason
    assert json.loads(next(iter(redis.store.values())))["failures_in_a_row"] == 0


@pytest.mark.parametrize(
    "stderr",
    ["wget: error getting response: Connection reset by peer", "wget: can't connect to remote host: Connection refused"],
)
async def test_an_upload_error_is_judged_by_the_fleet_and_recorded_in_its_window(stderr):
    cut_up = FAST.replace(UP, line("cf_up", 0, 0.2, 1, stderr))
    with flags():
        # the other providers' uploads go through: this node's failed upload is its own
        fleet = FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(0, 30))
        ctx, _, _ = make_ctx(cut_up, redis=fleet)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason
        assert json.loads(fleet.fleet["cf_up"][-1][1])[3] == 1
    with flags():
        # a quarter of the fleet fails the upload too: the receiver's doing, no verdict
        ctx, _, redis = make_ctx(cut_up, redis=FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(10, 30)))
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.passed and res.event.reason_code == Msg.UNMEASURED.reason
        assert json.loads(next(iter(redis.store.values())))["failures_in_a_row"] == 0


async def test_a_window_entry_in_another_format_is_skipped_not_fatal():
    with flags():
        fleet = FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(0, 30))
        for junk in (
            "5Old|node|1790000000.000|1",
            "not json",
            json.dumps({"a": 1}),
            json.dumps([1, 2]),
            # well-formed JSON with a list or a dict where a node id is expected
            json.dumps(["5Other", ["node"], Clock.now - 30, 1]),
            json.dumps([{"hk": 1}, "node", Clock.now - 30, 1]),
        ):
            fleet.fleet["pypi"].append((Clock.now - 30, junk))
        ctx, _, _ = make_ctx(PYPI_ONLY_SLOW, redis=fleet)
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.event.reason_code == Msg.SLOW_ONCE.reason


async def test_a_uuid_with_any_character_round_trips_through_the_window():
    with flags():
        fleet = FleetRedis()
        ctx, _, _ = make_ctx(PYPI_ONLY_SLOW, redis=fleet)
        ctx.executor.uuid = 'odd|uuid"]'
        await ContainerDownloadSpeedCheck().run(ctx)
    assert json.loads(fleet.fleet["pypi"][-1][1])[1] == 'odd|uuid"]'


async def test_the_fleet_comparison_result_is_logged(caplog):
    cut_up = FAST.replace(UP, line("cf_up", 0, 0.2, 1, "wget: error getting response: Connection reset by peer"))
    for stdout in (PYPI_ONLY_SLOW, cut_up):
        with flags(enforced=False):
            fleet = FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(0, 30))
            ctx, _, redis = make_ctx(stdout, redis=fleet)
            caplog.clear()
            with caplog.at_level("INFO"):
                res = await ContainerDownloadSpeedCheck().run(ctx)
            expected = Msg.OK  # a reset upload is a node-side reading, judged like a slow one
            assert res.passed and res.event.reason_code == expected.reason
            assert json.loads(next(iter(redis.store.values())))["failures_in_a_row"] == 0
            logged = next(r for r in caplog.records if "CONTAINER_DOWNLOAD_SPEED" in r.getMessage())
            assert logged.msg.extra["would_fail"] == ["pypi" if stdout is PYPI_ONLY_SLOW else "cf_up"]
            assert json.loads(fleet.fleet["pypi"][-1][1])[0] == ctx.miner_hotkey
    with flags(enforced=False):
        ctx, _, _ = make_ctx(SLOW)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason


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


@pytest.mark.parametrize(
    "stderr",
    [
        "wget: error getting response: Connection reset by peer",
        "ssl_client: SSL_connect: unexpected eof while reading",
        "net/http: TLS handshake timeout",
        "wget: connection closed prematurely",
    ],
)
async def test_connection_reset_counts_only_when_the_validator_reaches_it(stderr):
    cut = RENTALS + line("pypi", 0, 0.2, 1, stderr) + line("hf", CAP_BYTES, 1.0, 0)
    assert parse_download_speed(cut)[0].error == "connection_reset"
    with flags(reachable=True):
        ctx, _, _ = make_ctx(cut)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason
    with flags(reachable=False):
        ctx, _, redis = make_ctx(cut)
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


def test_the_upload_body_has_no_nul_bytes(tmp_path):
    # BusyBox wget sends a --post-file body through strlen(), so a NUL body goes out empty and the upload reads fast
    import os
    import subprocess

    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "timeout").write_text('#!/bin/sh\nshift\nexec "$@"\n')
    (stub / "wget").write_text(
        '#!/bin/sh\nfor a; do case "$a" in --post-file=*) cp "${a#--post-file=}" "$OUT";; esac; done\nexit 0\n'
    )
    for f in stub.iterdir():
        f.chmod(0o755)
    out = tmp_path / "body"
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "OUT": str(out)}
    subprocess.run(["sh", "-c", module._FETCH_SCRIPT], env=env, check=True, capture_output=True, timeout=60)
    body = out.read_bytes()
    assert len(body) == module.CAP_BYTES and b"\0" not in body


async def test_the_fleet_control_counts_each_other_node_once_by_its_latest_reading():
    # each other node counts once, by its latest reading in the window
    with flags():
        fleet = FleetRedis(pypi=(0, 30), hf=(0, 30), cf_up=(0, 30))
        for n in range(3):
            for k in range(6):
                at = Clock.now - 3000 + k * 300
                fleet.fleet["pypi"].append((at, member("5Slow", f"slow-{n}", at, 1)))
        ctx, _, _ = make_ctx(PYPI_ONLY_SLOW, redis=fleet)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason
    with flags():
        # a node that read slow and then fast counts as fast
        fleet = FleetRedis(pypi=(10, 30), hf=(0, 30), cf_up=(0, 30))
        for i in range(10):
            fleet.fleet["pypi"].append((Clock.now - 30, member(f"5Other{i % 7}", f"node-{i}", Clock.now - 30, 0)))
        ctx, _, _ = make_ctx(PYPI_ONLY_SLOW, redis=fleet)
        res = await ContainerDownloadSpeedCheck().run(ctx)
        assert res.event.reason_code == Msg.SLOW_ONCE.reason


async def test_the_median_best_endpoint_of_the_last_readings_goes_to_the_backend():
    # DAH-6239: the listing shows what a pod downloads at next to the host-side test. Each reading counts its best
    # endpoint (the DAH-5979 rule), the median of the last MEDIAN_OF readings rides in the specs on every cycle.
    fast_mbps = CAP_BYTES * 8 / 1e6
    with flags() as clock:
        ctx, runner, _ = make_ctx(FAST, PYPI_ONLY_SLOW, FAST, SLOW, SLOW, SLOW)
        check = ContainerDownloadSpeedCheck()
        first = await check.run(ctx)
        specs = first.updates["state"].specs
        assert specs[module.SPECS_KEY] == pytest.approx(fast_mbps, abs=0.1)
        assert specs[module.SPECS_MEASURED_AT_KEY] == clock.now
        # a cycle that does not measure still sends the last figure
        clock.now += 60
        not_due = await check.run(ctx)
        assert len(runner.commands) == 1 and not_due.updates["state"].specs[module.SPECS_KEY] == specs[module.SPECS_KEY]
        for _ in range(5):
            clock.now += 6 * 3600
            res = await check.run(ctx)
        # the last 5: F(hf) F S S S; the first FAST fell out of the window
        assert res.updates["state"].specs[module.SPECS_KEY] == pytest.approx(0.5, abs=0.1)
        assert res.updates["state"].specs[module.SPECS_MEASURED_AT_KEY] == clock.now


async def test_the_upload_reading_is_never_the_listed_download_figure():
    # lio#2411: the upload is measured with the downloads, but the listing shows what a pod downloads at
    slow_down_fast_up = (
        RENTALS + line("pypi", CAP_BYTES, 80.0, 0) + line("hf", CAP_BYTES, 80.0, 0) + line("cf_up", CAP_BYTES, 1.0, 0)
    )
    with flags():
        ctx, _, _ = make_ctx(slow_down_fast_up)
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.updates["state"].specs[module.SPECS_KEY] == pytest.approx(CAP_BYTES * 8 / 80.0 / 1e6, abs=0.1)


async def test_no_figure_before_the_first_measured_reading():
    with flags(reachable=False):
        ctx, _, _ = make_ctx("docker: Error response from daemon: unknown runtime")
        res = await ContainerDownloadSpeedCheck().run(ctx)
    assert res.event.reason_code == Msg.UNMEASURED.reason
    assert module.SPECS_KEY not in res.updates.get("state", ctx.state).specs


async def test_a_reading_off_the_rentals_network_is_not_the_listed_figure():
    # review: a node with no rental yet measures on the default bridge, a path the verdict does not enforce on; that
    # reading is not what a pod gets, so it never becomes the in-pod figure
    fast_on_bridge = FAST.replace(RENTALS, BRIDGE)
    with flags() as clock:
        ctx, _, _ = make_ctx(fast_on_bridge, FAST)
        check = ContainerDownloadSpeedCheck()
        first = await check.run(ctx)
        assert module.SPECS_KEY not in first.updates.get("state", ctx.state).specs
        clock.now += 6 * 3600
        second = await check.run(ctx)
    assert second.updates["state"].specs[module.SPECS_KEY] == pytest.approx(CAP_BYTES * 8 / 1e6, abs=0.1)
