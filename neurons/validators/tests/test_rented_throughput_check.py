"""A renter on a node listed at ~3 Gbps downloaded at 0.7-2 MB/s and the disk crawled, while the idle-node
download check never looks at a rented node."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.task.checks import container_download_speed as download_module
from services.task.checks import rented_throughput as module
from services.task.checks.rented_throughput import (
    RentedThroughputCheck,
    parse_busy,
    parse_disk,
    throughput_command,
)
from services.task.messages import RentedThroughputMessages as Msg
from services.task.score_calculator import calculate_scores

from tests.helpers import build_services, build_state, default_executor, make_context
from tests.test_container_download_speed_check import FAST, SLOW, Clock, FakeRedis, FakeRunner
from tests.test_partial_rental_split_scoring import _build_incentive, _make_job

EXECUTOR = default_executor()
pytestmark = pytest.mark.asyncio


def snaps(rx_bytes: int = 0, sectors: int = 0, tx_bytes: int = 0, counters: bool = True) -> str:
    """The docker root on nvme0n1, then two host snapshots 2 s apart: bytes received and sent on the NICs and sectors
    moved on nvme0n1 in between. counters=False: the host's NIC counters were not readable."""
    net0 = "rx=1000 tx=1000 " if counters else ""
    net1 = f"rx={1000 + rx_bytes} tx={1000 + tx_bytes} " if counters else ""
    return (
        "lium_tp_device name=nvme0n1p2\nlium_tp_device name=nvme0n1\n"
        f"lium_tp t=100.0 {net0}io=nvme0n1:500,sda:7,\n"
        f"lium_tp t=102.0 {net1}io=nvme0n1:{500 + sectors},sda:7,\n"
    )


def disk(mb: int = 256, secs: float = 0.5, rc: int = 0) -> str:
    return f"lium_disk bytes={mb * 1024 * 1024} start=10.0 end={10 + secs} rc={rc}\n"


GOOD = snaps() + disk() + FAST
SLOW_DISK = snaps() + disk(secs=30.0) + FAST  # 9 MB/s
SLOW_NET = snaps() + disk() + SLOW
# the renter is pulling 400 Mbps: a slow probe download is the shared link, not the host
BUSY_NET = snaps(rx_bytes=100_000_000) + disk() + SLOW
BUSY_UPLOAD = snaps(tx_bytes=100_000_000) + disk() + SLOW
NO_COUNTERS = snaps(counters=False) + disk() + SLOW
# the renter's loader keeps the disk busy; the network is healthy
BUSY_DISK = snaps(sectors=2_000_000) + disk() + FAST
NET_UNMEASURED = snaps(counters=False) + disk() + FAST


def rented(hotkey: str = "miner-hotkey", status: str | None = None) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            EXECUTOR.uuid: RentedExecutor(
                miner_hotkey=hotkey,
                executor_ip_address=EXECUTOR.address,
                executor_ip_port=str(EXECUTOR.port),
                pods=[RentedPod(pod_id="p1", container_name="pod_p1", status=status)],
            )
        }
    )


@contextmanager
def flags():
    fake = SimpleNamespace(
        RENTED_THROUGHPUT_CHECK_ENABLED=True,
        RENTED_THROUGHPUT_INTERVAL_HOURS=3.0,
        RENTED_THROUGHPUT_DISK_MIN_MB_PER_S=50.0,
        RENTED_THROUGHPUT_BUSY_MBPS=200.0,
        RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S=200.0,
        CONTAINER_DOWNLOAD_SPEED_MIN_MBPS=10.0,
    )
    clock = Clock()

    async def reaches(url):
        return True

    with (
        patch.object(module, "settings", fake),
        patch.object(download_module, "settings", fake),
        patch.object(module, "time", clock),
        patch.object(download_module, "validator_reaches", reaches),
        patch.object(module, "phase_offset_seconds", lambda uuid: 3 * 3600),
    ):
        yield clock


def make_ctx(*answers: str, rented_data=None):
    runner = FakeRunner(*answers)
    ctx = make_context(
        state=build_state(
            rented_data=rented() if rented_data is None else rented_data, sysbox_runtime=True
        ),
        runner=runner,
        services=build_services(redis=FakeRedis()),
    )
    return ctx, runner


def test_defaults_enforce_with_a_kill_switch():
    fields = type(module.settings).model_fields
    assert fields["RENTED_THROUGHPUT_CHECK_ENABLED"].default is True
    assert fields["RENTED_THROUGHPUT_DISK_MIN_MB_PER_S"].default == 50.0
    assert RentedThroughputCheck.fatal is False


def test_command_samples_the_host_then_probes_disk_and_network_in_the_renter_runtime():
    cmd = throughput_command(True)
    assert cmd.index("/proc/1/net/dev") < cmd.index("/usr/bin/docker run")
    assert cmd.index("DockerRootDir") < cmd.index("/proc/1/mountinfo") < cmd.index("lium_tp_device")
    assert "--runtime=sysbox-runc" in cmd and '--network "$net"' in cmd
    assert (
        "oflag=direct" in cmd
        and "conv=fsync" in cmd
        and "https://files.pythonhosted.org/packages/" in cmd
    )


def test_parse_disk_and_busy():
    assert parse_disk(disk(secs=1.0)).mb_per_s == pytest.approx(268.4, rel=1e-3)
    assert parse_disk(disk(mb=20, secs=60.0, rc=143)).mb_per_s == pytest.approx(
        0.35, rel=1e-2
    )  # timed out: slow
    assert parse_disk(disk(rc=1)).error == "write_failed"
    # O_DIRECT refused, the buffered retry is the reading
    assert parse_disk(disk(mb=0, rc=1) + disk(secs=2.0)).error is None
    assert parse_disk("").error == "no_output"
    rx, io = parse_busy(snaps(rx_bytes=50_000_000, sectors=2_000_000))
    assert rx == pytest.approx(200.0) and io == pytest.approx(512.0)
    assert parse_busy(snaps(rx_bytes=1, tx_bytes=50_000_000))[0] == pytest.approx(200.0)
    assert parse_busy(snaps(counters=False, sectors=2_000_000)) == (None, pytest.approx(512.0))
    assert parse_busy(snaps().replace("lium_tp_device name=nvme0n1\n", ""))[1] is None
    assert parse_busy("") == (None, None)


async def test_good_reading_passes():
    with flags():
        ctx, runner = make_ctx(GOOD)
        res = await RentedThroughputCheck().run(ctx)
    assert res.passed and res.event.reason_code == Msg.OK.reason and len(runner.commands) == 1


@pytest.mark.parametrize("resolved", [True, False])
@pytest.mark.parametrize("seconds", [0.5, 30])
@pytest.mark.parametrize("relevant", [0, 2_000_000])
@pytest.mark.parametrize("unrelated", [0, 2_000_000])
async def test_only_the_docker_root_disk_excuses_the_disk(resolved, seconds, relevant, unrelated):
    output = (
        ("lium_tp_device name=sda\n" if resolved else "")
        + "lium_tp t=100 rx=1000 tx=1000 io=sda:7,nvme0n1:500,\n"
        f"lium_tp t=102 rx=1000 tx=1000 io=sda:{7 + relevant},nvme0n1:{500 + unrelated},\n"
        + disk(secs=seconds)
        + FAST
    )
    with flags():
        ctx, _ = make_ctx(output)
        _, net, disk_side = await RentedThroughputCheck()._measure_and_judge(ctx)
    assert net == "pass"
    # an unresolved docker root is never judged on another disk's traffic
    assert disk_side == (
        "none" if relevant or not resolved else "fail" if seconds == 30 else "pass"
    )


@pytest.mark.parametrize(
    "answer, recovery",
    [
        (SLOW_DISK, GOOD),
        (SLOW_NET, GOOD),
        # the failed side passes while the other side is no verdict: the streak still ends
        (SLOW_NET, BUSY_DISK),
        (SLOW_DISK, NET_UNMEASURED),
    ],
)
async def test_two_slow_readings_zero_the_score_until_that_side_passes(answer, recovery):
    with flags() as clock:
        ctx, runner = make_ctx(answer, answer, recovery)
        check = RentedThroughputCheck()
        first = await check.run(ctx)
        assert first.passed and first.event.reason_code == Msg.SLOW_ONCE.reason
        clock.now += 31 * 60
        second = await check.run(ctx)
        assert not second.passed and second.event.reason_code == Msg.SLOW.reason
        assert second.updates == {"rented_throughput_passed": False}
        assert "on that resource stayed under its busy threshold" in second.event.remediation
        clock.now += 60  # the finding stands between readings
        assert (await check.run(ctx)).passed is False and len(runner.commands) == 2
        clock.now += 31 * 60
        assert (await check.run(ctx)).passed is True


# the host's NICs busy either way, or their counters unreadable: a slow download is no verdict
@pytest.mark.parametrize("busy", [BUSY_NET, BUSY_UPLOAD, NO_COUNTERS])
async def test_a_busy_renter_is_no_verdict_and_keeps_the_streak(busy):
    with flags() as clock:
        ctx, runner = make_ctx(SLOW_NET, busy, SLOW_NET)
        check = RentedThroughputCheck()
        await check.run(ctx)
        clock.now += 31 * 60
        busy = await check.run(ctx)
        assert busy.passed and busy.event.reason_code == Msg.SLOW_ONCE.reason
        clock.now += 31 * 60
        assert (await check.run(ctx)).passed is False


async def test_idle_foreign_or_stopped_rentals_are_left_alone():
    for rented_data in (
        RentedExecutorsResponse(executors={}),
        rented(hotkey="other"),
        rented(status="STOPPED"),
    ):
        with flags():
            ctx, runner = make_ctx(rented_data=rented_data)
            res = await RentedThroughputCheck().run(ctx)
        assert res.passed and not runner.commands and res.event.reason_code == Msg.SKIPPED.reason


# A download over the default bridge (lium-rentals not inspectable) did not measure the renter's path: no network verdict,
# so it neither strikes nor clears a failure established on lium-rentals; only a fast lium-rentals reading recovers.
@pytest.mark.parametrize("network", ["lium-rentals", "bridge"])
@pytest.mark.parametrize("fast", [True, False])
async def test_only_a_reading_on_the_rentals_network_judges_the_network(network, fast):
    answer = (GOOD if fast else SLOW_NET).replace(
        "lium_dl_net name=lium-rentals", f"lium_dl_net name={network}"
    )
    on_rentals = network == "lium-rentals"
    with flags():
        ctx, _ = make_ctx(answer)
        _, net, _ = await RentedThroughputCheck()._measure_and_judge(ctx)
    assert net == (("pass" if fast else "fail") if on_rentals else "none")
    with flags() as clock:
        ctx, _ = make_ctx(SLOW_NET, SLOW_NET, answer)
        check = RentedThroughputCheck()
        await check.run(ctx)
        clock.now += 31 * 60
        assert (await check.run(ctx)).passed is False
        clock.now += 31 * 60
        third = await check.run(ctx)
    assert third.passed is (on_rentals and fast)
    assert third.updates == ({} if on_rentals and fast else {"rented_throughput_passed": False})
    with flags() as clock:  # two slow bridge readings never zero the score
        ctx, _ = make_ctx(answer, answer)
        check = RentedThroughputCheck()
        await check.run(ctx)
        clock.now += 31 * 60
        assert (await check.run(ctx)).passed is (fast or not on_rentals)


# bond0 carries eth0+eth1's traffic, br-ex (OVS, uplinks enslaved to ovs-system) the same, and eth0.100 is a VLAN on
# eth0. Each case sums to eth0+eth1+eth2 = rx 1005 tx 101: a counted bond0/br-ex/VLAN or a dropped uplink shows.
# With the host's sysfs only physical NICs (a device link) count; without it, names and the host's bonding status.
_BOND = {"bond0": None, "br-ex": None, "eth0": "bond0", "eth1": "bond0", "eth2": ""}
_OVS = {"ovs-system": None, "br-ex": None, "eth0": "ovs-system", "eth1": "ovs-system", "eth2": ""}


@pytest.mark.parametrize(
    "sysfs, bonded",
    [(_BOND, False), (_OVS, False), ({}, True)],
    ids=["bond", "ovs", "no_host_sysfs"],
)
def test_bond_members_and_vlans_are_not_counted_twice(tmp_path, sysfs, bonded):
    net = tmp_path / "net"
    net.mkdir()
    for name, master in sysfs.items():  # None: virtual; else physical, enslaved to master when set
        (net / name).mkdir()
        if master is not None:
            (net / name / "device").mkdir()
        if master:
            (net / name / "master").symlink_to(net / master)
    bonding = tmp_path / "bonding"
    bonding.mkdir()
    if bonded:
        (bonding / "bond0").write_text(
            "Bonding Mode: IEEE 802.3ad\nSlave Interface: eth0\nMII Status: up\nSlave Interface: eth1\n"
        )
    rows = [
        ("lo", 9, 9),
        ("bond0", 1000, 100),
        ("br-ex", 1000, 100),
        ("ovs-system", 0, 0),
        ("eth0", 600, 60),
        ("eth1", 400, 40),
        ("eth0.100", 300, 30),
        ("eth2", 5, 1),
    ]
    dev = tmp_path / "dev"
    dev.write_text(
        "Inter-|   Receive\n face |bytes\n"
        + "".join(
            f"{n}: {rx} 0 0 0 0 0 0 0 {tx} 0 0 0 0 0 0 0\n"
            # no bond on the OVS host, else the old rule's bond0 would stand in for the dropped uplinks
            for n, rx, tx in rows
            if not (sysfs is _OVS and n == "bond0")
        )
    )
    script = (
        module._SNAPSHOT.replace("/proc/1/root/sys/class/net/*", f"{net}/*")
        .replace("/proc/1/net/bonding/*", f"{bonding}/*")
        .replace("/proc/1/net/dev", str(dev))
        .replace("[ /proc/1/root/ -ef / ] ||", "")
        .replace(f"sleep {module.BUSY_WINDOW_SECONDS}; ", "")
    )
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30).stdout
    assert out.count("rx=1005 tx=101") == 2, out


# Redis reads but refuses writes: the reading taken is kept in memory, so the next cycle in the interval runs no probe
async def test_unrecorded_reading_still_holds_the_cadence():
    class ReadOnlyRedis(FakeRedis):
        async def set(self, key, value, ex=None):
            raise ConnectionError("READONLY")

    redis = ReadOnlyRedis()
    redis.store[f"rented_throughput:{EXECUTOR.uuid}"] = '{"at": 0}'
    with flags() as clock, patch.dict(module._UNSAVED, clear=True):
        runner = FakeRunner(GOOD, GOOD)
        ctx = make_context(
            state=build_state(rented_data=rented(), sysbox_runtime=True),
            runner=runner,
            services=build_services(redis=redis),
        )
        check = RentedThroughputCheck()
        assert (await check.run(ctx)).passed and len(runner.commands) == 1
        clock.now += 60
        assert (await check.run(ctx)).passed and len(runner.commands) == 1
        clock.now += 3 * 3600
        await check.run(ctx)
        assert len(runner.commands) == 2


# No record yet, Redis reads but refuses writes, a real phase below the interval: the initial state is kept in memory
# (else each cycle recreates it in the future and the node is never probed); once Redis takes writes it is saved
async def test_a_new_node_is_probed_while_redis_refuses_writes():
    class FlakyRedis(FakeRedis):
        down = True

        async def set(self, key, value, ex=None):
            if self.down:
                raise ConnectionError("READONLY")
            await super().set(key, value, ex=ex)

    redis = FlakyRedis()
    runner = FakeRunner(GOOD, GOOD)
    ctx = make_context(
        state=build_state(rented_data=rented(), sysbox_runtime=True),
        runner=runner,
        services=build_services(redis=redis),
    )
    check = RentedThroughputCheck()
    with (
        flags() as clock,
        patch.object(module, "phase_offset_seconds", lambda uuid: 3600),
        patch.dict(module._UNSAVED, clear=True),
    ):
        for _ in range(2):  # due 2 h after the first cycle
            await check.run(ctx)
            assert not runner.commands
            clock.now += 3600
        await check.run(ctx)
        assert len(runner.commands) == 1
        redis.down = False
        clock.now += 3 * 3600
        await check.run(ctx)
        assert len(runner.commands) == 2 and module._UNSAVED == {}
        assert f"rented_throughput:{EXECUTOR.uuid}" in redis.store


# Redis refuses writes for both nodes: each keeps its own unsaved streak (one node's failed save never drops the
# other's), and once Redis takes writes again the saved state is used and nothing stays in memory
async def test_unrecorded_streaks_grow_for_every_node_until_redis_recovers():
    class FlakyRedis(FakeRedis):
        down = True

        async def set(self, key, value, ex=None):
            if self.down:
                raise ConnectionError("READONLY")
            await super().set(key, value, ex=ex)

    redis = FlakyRedis()
    other = EXECUTOR.model_copy(update={"uuid": "other-node"})
    nodes = []
    for executor in (EXECUTOR, other):
        redis.store[f"rented_throughput:{executor.uuid}"] = '{"at": 0}'
        rented_data = rented()
        rented_data.executors[executor.uuid] = rented_data.executors.pop(EXECUTOR.uuid)
        ctx = make_context(
            executor=executor,
            state=build_state(rented_data=rented_data, sysbox_runtime=True),
            runner=FakeRunner(*[SLOW_DISK] * 5),
            services=build_services(redis=redis),
        )
        nodes.append(ctx)
    check = RentedThroughputCheck()
    with flags() as clock, patch.dict(module._UNSAVED, clear=True):
        for streak in (1, 2, 3, 4):
            clock.now += 31 * 60
            for ctx in nodes:
                await check.run(ctx)
            assert [module._UNSAVED[c.executor.uuid].failures_in_a_row for c in nodes] == [
                streak,
                streak,
            ]
        redis.down = False
        clock.now += 31 * 60
        for ctx in nodes:
            assert (await check.run(ctx)).passed is False
        assert module._UNSAVED == {}
        assert all(
            '"failures_in_a_row": 5' in redis.store[f"rented_throughput:{c.executor.uuid}"]
            for c in nodes
        )


# One incentive cycle from the check's verdict to the pools: a throughput-failed node earns from neither pool, also
# when only part of a split node is rented (its free GPUs would otherwise draw idle pay)
@pytest.mark.parametrize("rented_gpus", [8, 1])
@pytest.mark.parametrize(
    "answers, enabled, earns",
    [
        ([GOOD], True, True),
        ([SLOW_DISK, SLOW_DISK], True, False),
        ([SLOW_DISK, SLOW_DISK, GOOD], True, True),
        ([SLOW_DISK, SLOW_DISK], False, True),
    ],
    ids=["pass", "fail", "recovery", "disabled"],
)
async def test_a_throughput_failed_node_earns_from_neither_pool(
    rented_gpus, answers, enabled, earns
):
    with flags() as clock, patch.dict(module._UNSAVED, clear=True):
        module.settings.RENTED_THROUGHPUT_CHECK_ENABLED = enabled
        ctx, _ = make_ctx(*answers)
        for _ in answers:
            result = await RentedThroughputCheck().run(ctx)
            clock.now += 31 * 60
    unpriced = ctx.executor.model_copy(
        update={"price_per_gpu": None}
    )  # no market price table in tests
    score_ctx = ctx.model_copy(update={**(result.updates or {}), "executor": unpriced})
    score, job_score, _ = calculate_scores(score_ctx, rented=True)
    job = _make_job(gpu_count=8, rented_gpu_count=rented_gpus)
    job.score, job.job_score = score, job_score
    incentive = _build_incentive(job)
    incentive._calculate_rental_share = AsyncMock(return_value=0.1)
    await incentive.calculate_mining_scores()

    idle = job.incentive_idle or 0.0
    assert ((job.incentive or 0.0) - idle > 0) is earns  # mining (rented) pool
    assert idle == (
        pytest.approx(0.1) if earns and rented_gpus == 1 else 0.0
    )  # rental-share (idle) pool
