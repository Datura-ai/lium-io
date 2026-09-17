"""DAH-2928 — get_disk_health() in machine_scrape.py and DiskHealthCheck in the pipeline.

A renter's file on a pod changed on disk after it was written, with no error reaching the container;
the executor's specs said nothing about its disks. The scrape now reports two readings (is the
docker root's filesystem mounted read-only, does a write to it go through); the check is non-fatal
and warns when the docker root refuses writes, without changing the score until the reading is
proven on live executors.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers are
extracted by ast and executed in their own namespace (same pattern as test_scrape_infiniband.py).
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, build_state
from services.task.checks.disk_health import DiskHealthCheck
from services.task.checks.gpu_vram_precheck import GpuVramPrecheck
from services.task.messages import DiskHealthMessages as Msg
from services.task.pipeline_factory import PipelineFactory

SRC = Path(__file__).resolve().parents[1] / "src"

DISK_HEALTH_HELPERS = {
    "HOST_MOUNTS_PATH",
    "HOST_ROOT_PREFIX",
    "ERRNO_EIO",
    "ERRNO_ENOSPC",
    "ERRNO_EROFS",
    "ERRNO_EDQUOT",
    "READ_ONLY_MOUNT_OPTIONS",
    "DiskHealthObservation",
    "mounts_holding",
    "write_probe_failure_reason",
    "probe_write",
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

# The /proc/mounts line a 7.0 kernel wrote for a docker root whose ext4 hit an error under errors=remount-ro
# (EC2 20260916T041828Z-z2w6, reading loop_root_ro_kernel): kernels 6.6+ keep `rw` and add `emergency_ro`;
# the write probe on that root failed with EROFS while the mount half of the check saw nothing.
HOST_MOUNTS_DOCKER_EMERGENCY_RO = (
    "/dev/nvme0n1p2 / ext4 rw,relatime,errors=remount-ro 0 0\n"
    "/dev/loop3 /mnt/liumdisk ext4 rw,relatime,errors=remount-ro,emergency_ro 0 0\n"
)

@pytest.fixture
def scrape() -> dict[str, Any]:
    """The disk-health helpers, executed in a namespace of their own."""
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        DISK_HEALTH_HELPERS,
        {"os": os, "tempfile": tempfile},
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


def test_a_kernel_6_6_emergency_remount_counts_as_read_only(scrape: dict[str, Any]) -> None:
    # Arrange — the real line: `rw` kept, `emergency_ro` added; a check that only looks for `ro` returns []
    # Act
    read_only = scrape["mounts_holding"](HOST_MOUNTS_DOCKER_EMERGENCY_RO, "/mnt/liumdisk/docker")

    # Assert
    assert read_only == ["/mnt/liumdisk"]


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


@pytest.mark.parametrize(
    "mounts",
    [
        "/dev/sda2 / ext4 ro,relatime,errors=remount-ro 0 0\n"
        "/dev/nvme0n1 /var/lib/docker ext4 rw,relatime 0 0\n",
        "/dev/nvme0n1 /var/lib/docker ext4 rw,relatime 0 0\n"
        "/dev/sda2 / ext4 ro,relatime,errors=remount-ro 0 0\n",
    ],
    ids=["root-first", "docker-first"],
)
def test_a_read_only_root_above_a_writable_docker_root_does_not_count(
    scrape: dict[str, Any], mounts: str
) -> None:
    # Arrange — root remounted ro after an error, the docker root on its own writable disk:
    # containers still start, so this is no read-only docker root, whatever the line order
    # Act / Assert
    assert scrape["mounts_holding"](mounts, "/var/lib/docker") == []


def test_only_the_covering_mount_is_reported_when_it_and_a_parent_are_read_only(
    scrape: dict[str, Any],
) -> None:
    # Arrange — both / and the docker disk are ro: the disk a write lands on is the one named
    mounts = "/dev/sda2 / ext4 ro,relatime 0 0\n/dev/nvme0n1 /var/lib/docker ext4 ro,relatime 0 0\n"

    # Act / Assert
    assert scrape["mounts_holding"](mounts, "/var/lib/docker") == ["/var/lib/docker"]


def test_a_mount_over_the_same_point_takes_the_later_line(scrape: dict[str, Any]) -> None:
    # Arrange — /var/lib/docker mounted ro, then a rw filesystem mounted over it: writes land on
    # the later one
    mounts = (
        "/dev/sda2 / ext4 rw,relatime 0 0\n"
        "/dev/sdb1 /var/lib/docker ext4 ro,relatime 0 0\n"
        "/dev/nvme0n1 /var/lib/docker ext4 rw,relatime 0 0\n"
    )

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
    assert error.startswith("read_only: ")
    assert "Read-only file system" in error


def test_write_probe_fails_on_an_io_error(scrape: dict[str, Any], monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    verdict, error = scrape["probe_write"]("/var/lib/docker")

    assert (verdict, error.split(":")[0]) == ("failed", "io_error")


@pytest.mark.parametrize(
    ("errno_value", "message", "reason"),
    [(28, "No space left on device", "no_space"), (122, "Disk quota exceeded", "quota")],
    ids=["ENOSPC", "EDQUOT"],
)
def test_write_probe_fails_on_a_full_docker_root(
    scrape: dict[str, Any], monkeypatch, errno_value: int, message: str, reason: str
) -> None:
    # Arrange — a full disk or an exhausted quota cannot start a container either; before, both
    # fell through to `skipped`, which the check reads as DISK_HEALTH_OK
    def refuse(*args, **kwargs):
        raise OSError(errno_value, message)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    # Act
    verdict, error = scrape["probe_write"]("/var/lib/docker")

    # Assert — failed, with its own reason rather than the read-only one
    assert verdict == "failed"
    assert error.startswith(f"{reason}: ")
    assert message in error


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
# --------------------------------------------------------------------------------------------------
# the whole observation
# --------------------------------------------------------------------------------------------------
def test_get_disk_health_reports_both_readings_side_by_side(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    # Arrange — a host whose docker disk just went read-only; the scrape sees the host through /proc/1/root
    mounts_path = tmp_path / "mounts"
    mounts_path.write_text(HOST_MOUNTS_DOCKER_RO)
    scrape["HOST_MOUNTS_PATH"] = str(mounts_path)
    scrape["HOST_ROOT_PREFIX"] = str(tmp_path / "host-root")
    scrape["docker_api_get"] = lambda path: {"DockerRootDir": "/var/lib/docker"}

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
    assert set(payload) == {"dh_docker_root_dir", "dh_read_only_mounts", "dh_write_probe", "dh_write_probe_error"}


def test_get_disk_health_falls_back_to_the_default_docker_root_and_an_unreadable_mount_table(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — no docker /info answer, no readable mount table: unknown mounts, not a false ro
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "missing-mounts")
    scrape["HOST_ROOT_PREFIX"] = str(tmp_path / "host-root")
    scrape["docker_api_get"] = lambda path: {}
    scrape["probe_write"] = lambda directory: ("ok", "")

    # Act
    payload = scrape["get_disk_health"]().as_payload()

    # Assert
    assert payload["dh_docker_root_dir"] == "/var/lib/docker"
    assert payload["dh_read_only_mounts"] == []
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
async def test_a_read_only_docker_root_is_a_warning_and_does_not_fail_the_executor(context_factory):
    health = _health(read_only_mounts=["/var/lib/docker"], write_probe="failed", write_probe_error="EROFS")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_WRITABLE.reason
    assert result.event.severity == "warning"
    assert result.event.what_we_saw["read_only_mounts"] == ["/var/lib/docker"]


@pytest.mark.asyncio
async def test_a_read_only_covering_mount_alone_is_the_same_warning(context_factory):
    # the mount table read ro, then a remount to rw landed before the probe wrote: still reported
    health = _health(read_only_mounts=["/var/lib/docker"], write_probe="ok")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_WRITABLE.reason
    assert result.event.what_we_saw["write_probe"] == "ok"


@pytest.mark.asyncio
async def test_a_write_probe_refused_with_eio_is_a_warning_on_a_read_write_mount(context_factory):
    health = _health(write_probe="failed", write_probe_error="OSError: [Errno 5] Input/output error")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_WRITABLE.reason
    assert result.event.severity == "warning"


@pytest.mark.asyncio
async def test_a_skipped_write_probe_does_not_fail(context_factory):
    # the scrape could not write there for a reason unrelated to the disk (EACCES, missing directory)
    health = _health(write_probe="skipped", write_probe_error="PermissionError: [Errno 13] Permission denied")
    ctx = context_factory(state=build_state(specs={"disk_health": health}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_a_scrape_without_the_probe_is_unknown_not_bad(context_factory):
    ctx = context_factory(state=build_state(specs={"disk_health_scrape_error": "RuntimeError('x')"}))

    result = await DiskHealthCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNKNOWN.reason
    assert result.event.what_we_saw["scrape_error"] == "RuntimeError('x')"


# --------------------------------------------------------------------------------------------------
# pipelines
# --------------------------------------------------------------------------------------------------
def test_the_check_follows_the_vram_precheck_in_both_pipelines():
    # the dry run runs the same scrape and the other pure-data checks; this one was missing from it
    for build in (PipelineFactory.build_checks, PipelineFactory.build_dry_run_checks):
        ids = [check.check_id for check in build()]

        disk_health_at, vram_precheck_at = ids.index(DiskHealthCheck.check_id), ids.index(GpuVramPrecheck.check_id)
        assert disk_health_at == vram_precheck_at + 1, build.__name__
