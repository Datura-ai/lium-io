"""DAH-5244: the host disk reading the validator takes before it starts a container. On the stand a writer whose
dirty pages sat on the Docker root made `docker run` wait 50-370 s while the same device under a direct-I/O
writer gave 1 s: `sync -f` on the Docker root is the signal, the disk counters around it are telemetry."""

from __future__ import annotations

import subprocess

import pytest
from services.task import host_disk as module
from services.task.host_disk import (
    DISK_PRESSURE_SYNCFS_SECONDS,
    DiskPressureReading,
    disk_pressure_command,
    parse_disk_pressure,
)


def output(
    *,
    start: float = 100.0,
    end: float = 100.3,
    rc: str = "0",
    devices: tuple[str, ...] = ("nvme0n1p2", "nvme0n1"),
    ticks: int = 100,
    ms: int = 500,
    ios: int = 50,
    sectors: int = 20_000,
    dirty_kb: int = 1_048_576,
    psi: str = "3.50",
) -> str:
    device_lines = "".join(f"lium_tp_device name={d}\n" for d in devices)
    first = "lium_dp t=100.0 dirty=524288:1024 psi=1.00 io=nvme0n1:1000:100:200:10,sda:5:0:0:0,\n"
    second = (
        f"lium_dp t=102.0 dirty={dirty_kb}:2048 psi={psi} "
        f"io=nvme0n1:{1000 + sectors}:{100 + ticks}:{200 + ms}:{10 + ios},sda:5:0:0:0,\n"
    )
    return device_lines + first + second + f"lium_dp_sync start={start} end={end} rc={rc}\n"


def test_command_times_a_detached_syncfs_on_the_docker_root_between_two_snapshots():
    cmd = disk_pressure_command()
    assert "timeout 5 /usr/bin/docker info" in cmd
    assert 'sync -f "$sr"' in cmd and "</dev/null >/dev/null 2>&1 &" in cmd
    assert cmd.index("DockerRootDir") < cmd.index("lium_tp_device") < cmd.index("sync -f")
    assert cmd.count("dps;") == 2 and "/proc/diskstats" in cmd and "$h/proc/meminfo" in cmd
    assert cmd.index("lium_dp_sync") < cmd.index("dp=0")


def test_a_quick_sync_is_no_pressure_and_the_disk_fields_are_read():
    reading = parse_disk_pressure(output())
    assert reading == DiskPressureReading(
        syncfs_seconds=0.3,
        syncfs_timed_out=False,
        device="nvme0n1",
        mb_per_s=5.1,
        util_percent=5.0,
        await_ms=10.0,
        dirty_mb=1024,
        writeback_mb=2,
        psi_io_full_avg10=3.5,
        pressure=False,
        reason=None,
        error=None,
    )


@pytest.mark.parametrize(
    "end, rc, seconds, timed_out",
    [(105.0, "0", 5.0, False), (120.0, "", 20.0, True)],
    ids=["slow", "timed_out"],
)
def test_a_slow_or_unfinished_sync_is_pressure(end, rc, seconds, timed_out):
    reading = parse_disk_pressure(output(end=end, rc=rc))
    assert reading.pressure and reading.reason == "syncfs"
    assert reading.syncfs_seconds == seconds and reading.syncfs_timed_out is timed_out
    assert seconds >= DISK_PRESSURE_SYNCFS_SECONDS or timed_out


def test_a_saturated_disk_without_a_sync_backlog_is_telemetry_only():
    # the direct-I/O stage of the stand: device pinned, host Dirty high, sync fast
    reading = parse_disk_pressure(output(ticks=2000, ms=20_000, ios=100, sectors=400_000, dirty_kb=1_500_000))
    assert reading.util_percent == 100.0 and reading.await_ms == 200.0 and reading.mb_per_s == 102.4
    assert reading.pressure is False


def test_await_needs_enough_completed_ios():
    assert parse_disk_pressure(output(ios=5, ms=1000)).await_ms is None
    assert parse_disk_pressure(output(ios=20, ms=1000)).await_ms == 50.0


def test_the_busiest_docker_root_device_is_reported_whatever_its_name():
    # an nbd or dm device under the root counts like a disk; a device that is not under the root does not
    out = output(devices=("dm-0", "nbd0")).replace("io=nvme0n1:", "io=nbd0:")
    assert parse_disk_pressure(out).device == "nbd0"
    out = output(devices=("sdb",))
    assert parse_disk_pressure(out).device is None


@pytest.mark.parametrize(
    "transform, error",
    [
        (lambda o: o.replace("rc=", "rc=127"), "syncfs_unavailable"),
        (lambda o: o.replace("rc=", "rc=1"), "syncfs_failed"),
        (lambda o: "\n".join(ln for ln in o.splitlines() if not ln.startswith("lium_dp_sync")), "no_output"),
        (lambda o: "", "no_output"),
    ],
    ids=["no_sync_binary", "sync_failed", "no_sync_line", "empty"],
)
def test_a_reading_that_could_not_be_taken_is_never_pressure(transform, error):
    reading = parse_disk_pressure(transform(output(end=120.0, rc="")))
    assert reading.error == error and reading.pressure is False


def test_an_unresolved_docker_root_device_still_times_the_sync():
    # btrfs/zfs roots give a 0:NN device with no sysfs entry; the sync needs the path, not the device
    out = output(end=120.0, rc="").replace("lium_tp_device name=nvme0n1p2\n", "").replace("lium_tp_device name=nvme0n1\n", "")
    reading = parse_disk_pressure(out)
    assert reading.pressure is True and reading.error is None and reading.device is None and reading.dirty_mb == 1024


def test_a_counter_reset_between_snapshots_is_skipped():
    out = output().replace("io=nvme0n1:21000:200:700:60", "io=nvme0n1:10:200:700:60")
    reading = parse_disk_pressure(out)
    assert reading.device is None and reading.dirty_mb == 1024


# The probe itself under sh, with the host files and the sync replaced: the sync's duration is what comes out,
# a sync that outlives the poll cap is reported without a return code, and the shell's own `dp` agrees.
@pytest.mark.parametrize("sync_takes, expect_dp", [("0.2", "0"), ("60", "1")], ids=["fast", "stuck"])
def test_probe_shell_reports_the_sync_duration_and_sets_dp(tmp_path, sync_takes, expect_dp, monkeypatch):
    monkeypatch.setattr(module, "SYNCFS_POLL_SECONDS", 1)
    (tmp_path / "uptime").write_text("100.00 400.00\n")
    (tmp_path / "uptime_later").write_text("100.30 400.00\n")
    (tmp_path / "meminfo").write_text("Dirty:            4096 kB\nWriteback:        512 kB\n")
    (tmp_path / "pressure").mkdir()
    (tmp_path / "pressure" / "io").write_text("some avg10=1.00 avg60=0.00 avg300=0.00 total=1\nfull avg10=0.50 avg60=0.00 avg300=0.00 total=1\n")
    (tmp_path / "diskstats").write_text(" 259 0 nvme0n1 10 0 1000 200 10 0 1000 300 0 100 500 0 0 0 0 0 0\n")
    probe = module._PRESSURE_PROBE.replace(
        f"while [ $i -lt {module.SYNCFS_POLL_SECONDS * 4} ]", "while [ $i -lt 4 ]"
    )
    script = (
        'r=/var/lib/docker; dv="nvme0n1p2 nvme0n1"; '
        + probe.replace("h=/proc/1/root; [ $h/ -ef / ] && h=; ", f"h={tmp_path}; ")
        .replace('sync -f "$sr"', f"sleep {sync_takes}")
        .replace("echo \"$rc $(cut -d' ' -f1 /proc/uptime)\"", f"echo \"$rc $(cut -d' ' -f1 {tmp_path}/uptime_later)\"")
        .replace("/proc/uptime", f"{tmp_path}/uptime")
        .replace("/proc/diskstats", f"{tmp_path}/diskstats")
        .replace("$h/proc/meminfo", f"{tmp_path}/meminfo")
        .replace("$h/proc/pressure/io", f"{tmp_path}/pressure/io")
        + 'echo "dp=$dp"'
    )
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30).stdout
    reading = parse_disk_pressure("lium_tp_device name=nvme0n1\n" + out)
    assert out.strip().endswith(f"dp={expect_dp}"), out
    assert reading.syncfs_timed_out is (expect_dp == "1")
    if expect_dp == "0":
        assert reading.syncfs_seconds == 0.3  # the sync's own end, not the 2 s snapshot window
    assert reading.device == "nvme0n1" and reading.dirty_mb == 4 and reading.psi_io_full_avg10 == 0.5
