"""DAH-2928 — get_disk_health() in machine_scrape.py and DiskHealthCheck in the pipeline.

A renter's file on a pod changed on disk after it was written, with no error reaching the container;
the executor's specs said nothing about its disks. The scrape now reports four independent readings
(read-only docker-root mount, a write probe, kernel disk errors, sysfs/SMART state) and the check fails
the executor only on the one that is unambiguous: the docker root refuses writes.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers are
extracted by ast and executed in their own namespace (same pattern as test_scrape_infiniband.py).
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, build_state

from services.task.checks.disk_health import DiskHealthCheck, disk_error_summary
from services.task.messages import DiskHealthMessages as Msg

SRC = Path(__file__).resolve().parents[1] / "src"

DISK_HEALTH_HELPERS = {
    "HOST_MOUNTS_PATH",
    "HOST_ROOT_PREFIX",
    "BLOCK_SYSFS_PATH",
    "NVME_SYSFS_PATH",
    "KERNEL_ERRORS_CMD",
    "KERNEL_ERROR_LINES_KEPT",
    "KERNEL_ERROR_LINE_CHARS",
    "KERNEL_DISK_ERROR_PATTERN",
    "ERRNO_EIO",
    "ERRNO_EROFS",
    "read_sysfs_value",
    "DiskHealthObservation",
    "mounts_holding",
    "probe_write",
    "kernel_disk_errors",
    "block_device_io_errors",
    "nvme_controller_states",
    "smart_health",
    "get_disk_health",
}

# /proc/1/mounts of a host whose docker root sits on its own NVMe filesystem; the root stays rw.
HOST_MOUNTS = """
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
/dev/nvme0n1p2 / ext4 rw,relatime,errors=remount-ro 0 0
/dev/nvme1n1 /var/lib/docker ext4 rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid,nodev,size=13158620k,mode=755 0 0
"""

# The same host after ext4 hit an error and honoured errors=remount-ro on the docker disk.
HOST_MOUNTS_DOCKER_RO = HOST_MOUNTS.replace("/var/lib/docker ext4 rw,relatime", "/var/lib/docker ext4 ro,relatime")

# dmesg --level=err on a host with a failing NVMe and one unrelated error, as printed by 6.x kernels.
DMESG_WITH_DISK_ERRORS = """
[  120.442112] nvme nvme1: I/O Cmd(0x2) @ LBA 2411724800, 256 blocks, I/O Error (sct 0x2 / sc 0x81) MORE
[  120.442131] critical medium error, dev nvme1n1, sector 2411724800 op 0x0:(READ) flags 0x80700 phys_seg 32 prio class 2
[  120.442140] EXT4-fs error (device nvme1n1): ext4_find_entry:1663: inode #131073: comm dockerd: reading directory lblock 0
[  121.001003] EXT4-fs (nvme1n1): Remounting filesystem read-only
[  200.100000] usb 1-1: device descriptor read/64, error -71
"""


@pytest.fixture
def scrape() -> dict[str, Any]:
    """The disk-health helpers, executed in a namespace of their own."""
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        DISK_HEALTH_HELPERS,
        {"os": os, "re": re, "glob": glob, "json": json, "shutil": shutil, "tempfile": tempfile},
    )


# --------------------------------------------------------------------------------------------------
# mounts
# --------------------------------------------------------------------------------------------------
def test_read_only_docker_root_mount_is_reported(scrape: dict[str, Any]) -> None:
    # Act
    read_only = scrape["mounts_holding"](HOST_MOUNTS_DOCKER_RO, "/var/lib/docker")

    # Assert
    assert read_only == ["/var/lib/docker"]


def test_read_write_mounts_report_nothing(scrape: dict[str, Any]) -> None:
    assert scrape["mounts_holding"](HOST_MOUNTS, "/var/lib/docker") == []


def test_a_read_only_root_counts_when_the_docker_root_lives_on_it(scrape: dict[str, Any]) -> None:
    # Arrange — docker root on the root filesystem, root remounted ro
    mounts = "/dev/sda2 / ext4 ro,relatime,errors=remount-ro 0 0\ntmpfs /run tmpfs rw 0 0\n"

    # Act / Assert
    assert scrape["mounts_holding"](mounts, "/var/lib/docker") == ["/"]


def test_a_read_only_mount_elsewhere_does_not_count(scrape: dict[str, Any]) -> None:
    # Arrange — an immutable /usr, as image-based hosts have, is not the disk containers write to
    mounts = "/dev/sda2 / ext4 rw,relatime 0 0\n/dev/sda3 /usr ext4 ro,relatime 0 0\n"

    # Act / Assert
    assert scrape["mounts_holding"](mounts, "/var/lib/docker") == []


# --------------------------------------------------------------------------------------------------
# write probe
# --------------------------------------------------------------------------------------------------
def test_write_probe_passes_on_a_writable_directory_and_leaves_nothing_behind(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Act
    verdict, error = scrape["probe_write"](str(tmp_path))

    # Assert
    assert (verdict, error) == ("ok", "")
    assert list(tmp_path.iterdir()) == []


def test_write_probe_fails_on_a_read_only_filesystem(scrape: dict[str, Any], monkeypatch) -> None:
    # Arrange — the kernel's answer on an errors=remount-ro disk
    def refuse(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    # Act
    verdict, error = scrape["probe_write"]("/var/lib/docker")

    # Assert
    assert verdict == "failed"
    assert "Read-only file system" in error


def test_write_probe_fails_on_an_io_error(scrape: dict[str, Any], monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    verdict, _ = scrape["probe_write"]("/var/lib/docker")

    assert verdict == "failed"


def test_write_probe_is_skipped_not_failed_when_the_scrape_may_not_write_there(
    scrape: dict[str, Any], monkeypatch
) -> None:
    # Arrange — EACCES says nothing about the disk, only about where the scrape runs from
    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    # Act
    verdict, error = scrape["probe_write"]("/var/lib/docker")

    # Assert
    assert verdict == "skipped"
    assert "Permission denied" in error


def test_write_probe_is_skipped_on_a_missing_directory(scrape: dict[str, Any], tmp_path: Path) -> None:
    verdict, _ = scrape["probe_write"](str(tmp_path / "nope"))

    assert verdict == "skipped"


# --------------------------------------------------------------------------------------------------
# kernel log
# --------------------------------------------------------------------------------------------------
def test_kernel_disk_errors_counts_block_and_filesystem_faults_only(scrape: dict[str, Any]) -> None:
    # Act
    count, lines = scrape["kernel_disk_errors"](DMESG_WITH_DISK_ERRORS)

    # Assert — the USB descriptor error is not a disk fault
    assert count == 4
    assert [line.split("] ", 1)[1][:22] for line in lines] == [
        "nvme nvme1: I/O Cmd(0x",
        "critical medium error,",
        "EXT4-fs error (device ",
        "EXT4-fs (nvme1n1): Rem",
    ]


def test_kernel_disk_errors_keeps_only_the_tail_and_truncates_lines(scrape: dict[str, Any]) -> None:
    # Arrange — 20 SATA errors of 300 characters
    long_line = "[1.0] ata3.00: failed command: READ FPDMA QUEUED " + "x" * 300
    log = "\n".join([long_line] * 20)

    # Act
    count, lines = scrape["kernel_disk_errors"](log)

    # Assert
    assert count == 20
    assert len(lines) == scrape["KERNEL_ERROR_LINES_KEPT"]
    assert all(len(line) == scrape["KERNEL_ERROR_LINE_CHARS"] for line in lines)


def test_a_clean_kernel_log_counts_nothing(scrape: dict[str, Any]) -> None:
    log = "[1.0] usb 1-1: device descriptor read/64, error -71\n[2.0] nvidia: loading out-of-tree module taints kernel.\n"

    assert scrape["kernel_disk_errors"](log) == (0, [])


# --------------------------------------------------------------------------------------------------
# sysfs and SMART
# --------------------------------------------------------------------------------------------------
def test_block_io_error_counters_report_only_nonzero_devices(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — sysfs prints the SCSI counters in hex
    for device, count in (("sda", "0x0"), ("sdb", "0x1a"), ("nvme0n1", None)):
        device_dir = tmp_path / device / "device"
        device_dir.mkdir(parents=True)
        if count is not None:
            (device_dir / "ioerr_cnt").write_text(f"{count}\n")
    scrape["BLOCK_SYSFS_PATH"] = str(tmp_path)

    # Act / Assert
    assert scrape["block_device_io_errors"]() == {"sdb": 26}


def test_nvme_controller_states_report_only_controllers_that_are_not_live(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    for controller, state in (("nvme0", "live"), ("nvme1", "resetting")):
        (tmp_path / controller).mkdir()
        (tmp_path / controller / "state").write_text(f"{state}\n")
    scrape["NVME_SYSFS_PATH"] = str(tmp_path)

    assert scrape["nvme_controller_states"]() == {"nvme1": "resetting"}


def test_smart_is_unavailable_without_smartctl(scrape: dict[str, Any], monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert scrape["smart_health"]() == "unavailable"


def test_smart_reads_the_verdict_per_device_and_keeps_going_past_a_failing_call(
    scrape: dict[str, Any], monkeypatch
) -> None:
    # Arrange — smartctl -j output for a healthy and a failing disk; the third device makes smartctl exit non-zero
    reports = {
        "/dev/nvme0n1": json.dumps({"smart_status": {"passed": True}}),
        "/dev/sda": json.dumps({"smart_status": {"passed": False}}),
    }

    def fake_run_cmd(cmd):
        device = cmd.split()[-1]
        if device not in reports:
            raise RuntimeError(f"run_cmd error {cmd!r} returncode=4")
        return reports[device]

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/sbin/smartctl")
    monkeypatch.setattr(glob, "glob", lambda pattern: {"/dev/sd?": ["/dev/sda", "/dev/sdb"], "/dev/nvme?n1": ["/dev/nvme0n1"]}[pattern])
    scrape["run_cmd"] = fake_run_cmd

    # Act
    verdicts = scrape["smart_health"]()

    # Assert
    assert verdicts["/dev/nvme0n1"] == "PASSED"
    assert verdicts["/dev/sda"] == "FAILED"
    assert verdicts["/dev/sdb"].startswith("error: ")


# --------------------------------------------------------------------------------------------------
# the whole observation
# --------------------------------------------------------------------------------------------------
def test_get_disk_health_reports_every_reading_side_by_side(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    # Arrange — a host whose docker disk just went read-only; the scrape sees the host through /proc/1/root
    mounts_path = tmp_path / "mounts"
    mounts_path.write_text(HOST_MOUNTS_DOCKER_RO)
    scrape["HOST_MOUNTS_PATH"] = str(mounts_path)
    scrape["HOST_ROOT_PREFIX"] = str(tmp_path / "host-root")
    scrape["BLOCK_SYSFS_PATH"] = str(tmp_path / "block")
    scrape["NVME_SYSFS_PATH"] = str(tmp_path / "nvme")
    scrape["docker_api_get"] = lambda path: {"DockerRootDir": "/var/lib/docker"}
    scrape["run_cmd"] = lambda cmd: DMESG_WITH_DISK_ERRORS
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def refuse(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    # Act
    payload = scrape["get_disk_health"]().as_payload()

    # Assert
    assert payload["dh_docker_root_dir"] == "/var/lib/docker"
    assert payload["dh_read_only_mounts"] == ["/var/lib/docker"]
    assert payload["dh_write_probe"] == "failed"
    assert "Read-only file system" in payload["dh_write_probe_error"]
    assert payload["dh_kernel_io_errors"] == 4
    assert len(payload["dh_kernel_io_error_lines"]) == 4
    assert payload["dh_kernel_log_error"] == ""
    assert payload["dh_block_io_errors"] == {}
    assert payload["dh_nvme_states"] == {}
    assert payload["dh_smart"] == "unavailable"


def test_get_disk_health_records_an_unreadable_kernel_log_instead_of_a_zero(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    # Arrange — dmesg without CAP_SYSLOG
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "missing-mounts")
    scrape["HOST_ROOT_PREFIX"] = str(tmp_path / "host-root")
    scrape["BLOCK_SYSFS_PATH"] = str(tmp_path / "block")
    scrape["NVME_SYSFS_PATH"] = str(tmp_path / "nvme")
    scrape["docker_api_get"] = lambda path: {}

    def denied(cmd):
        raise RuntimeError("run_cmd error 'dmesg' returncode=1 stderr='dmesg: read kernel buffer failed: Operation not permitted'")

    scrape["run_cmd"] = denied
    monkeypatch.setattr(shutil, "which", lambda name: None)
    scrape["probe_write"] = lambda directory: ("ok", "")

    # Act
    payload = scrape["get_disk_health"]().as_payload()

    # Assert
    assert payload["dh_docker_root_dir"] == "/var/lib/docker"
    assert payload["dh_kernel_io_errors"] == 0
    assert "Operation not permitted" in payload["dh_kernel_log_error"]
    assert payload["dh_write_probe"] == "ok"


def test_every_disk_health_key_is_registered_in_both_obfuscation_tables() -> None:
    # Arrange
    scrape_source = ast.parse((SRC / "miner_jobs" / "machine_scrape.py").read_text())
    emitted = {
        node.value
        for node in ast.walk(scrape_source)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value.startswith("dh_") or node.value.startswith("data_disk_health"))
    }
    service = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    original_keys: list[str] = []
    mapped_keys: list[str] = []
    for node in ast.walk(service):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ORIGINAL_KEYS":
            original_keys = list(ast.literal_eval(node.value))
        if isinstance(node, ast.FunctionDef) and node.name == "generate_key_mappings":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    mapped_keys = [key.value for key in child.keys if isinstance(key, ast.Constant)]
                    break

    # Act / Assert — present in both, and in the rename table no key comes before a longer key it is
    # a substring of (ecrypt_miner_job_files renames by sequential str.replace over the source)
    assert emitted, "the scrape emits no disk-health keys — the parse is looking in the wrong place"
    assert emitted <= set(original_keys), f"missing from ORIGINAL_KEYS: {sorted(emitted - set(original_keys))}"
    assert emitted <= set(mapped_keys), f"missing from generate_key_mappings: {sorted(emitted - set(mapped_keys))}"
    for index, key in enumerate(mapped_keys):
        longer_later = [other for other in mapped_keys[index + 1 :] if key != other and key in other]
        assert not longer_later, f"{key!r} is renamed before {longer_later} and would corrupt them"


# --------------------------------------------------------------------------------------------------
# DiskHealthCheck
# --------------------------------------------------------------------------------------------------
def _health(**overrides) -> dict[str, Any]:
    base = {
        "docker_root_dir": "/var/lib/docker",
        "read_only_mounts": [],
        "write_probe": "ok",
        "write_probe_error": "",
        "kernel_io_errors": 0,
        "kernel_io_error_lines": [],
        "kernel_log_error": "",
        "block_io_errors": {},
        "nvme_states": {},
        "smart": "unavailable",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_a_healthy_disk_passes(context_factory):
    ctx = context_factory(state=build_state(specs={"disk_health": _health()}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_a_read_only_docker_root_fails_the_executor(context_factory):
    health = _health(read_only_mounts=["/var/lib/docker"], write_probe="failed", write_probe_error="EROFS")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_WRITABLE.reason
    assert result.event.what_we_saw["read_only_mounts"] == ["/var/lib/docker"]


@pytest.mark.asyncio
async def test_a_write_probe_refused_with_eio_fails_even_with_a_read_write_mount(context_factory):
    health = _health(write_probe="failed", write_probe_error="OSError: [Errno 5] Input/output error")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_WRITABLE.reason


@pytest.mark.asyncio
async def test_a_skipped_write_probe_does_not_fail(context_factory):
    # the scrape could not write there for a reason unrelated to the disk (EACCES, missing directory)
    health = _health(write_probe="skipped", write_probe_error="PermissionError: [Errno 13] Permission denied")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_kernel_io_errors_are_reported_but_do_not_fail(context_factory):
    health = _health(kernel_io_errors=3, kernel_io_error_lines=["critical medium error, dev nvme1n1"])
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.ERRORS_REPORTED.reason
    assert result.event.what_we_saw["kernel_io_errors"] == 3


@pytest.mark.asyncio
async def test_a_failed_smart_verdict_is_reported(context_factory):
    health = _health(smart={"/dev/sda": "PASSED", "/dev/sdb": "FAILED"})
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.ERRORS_REPORTED.reason
    assert result.event.what_we_saw["smart"] == {"/dev/sdb": "FAILED"}


@pytest.mark.asyncio
async def test_a_scrape_without_the_probe_is_unknown_not_bad(context_factory):
    ctx = context_factory(state=build_state(specs={"disk_health_scrape_error": "RuntimeError('x')"}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert result.event.what_we_saw["scrape_error"] == "RuntimeError('x')"


def test_disk_error_summary_keeps_only_the_readings_that_say_something_is_wrong():
    health = _health(block_io_errors={"sdb": 26}, nvme_states={"nvme1": "resetting"}, smart={"/dev/sda": "PASSED"})

    assert disk_error_summary(health) == {"block_io_errors": {"sdb": 26}, "nvme_states": {"nvme1": "resetting"}}
