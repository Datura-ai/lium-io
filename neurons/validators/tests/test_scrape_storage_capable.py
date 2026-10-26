"""DAH-1997 (Fish 21 Oct): the storage limit is required only of hosts that can build it.

storage_limit_capable() in machine_scrape.py: docker's data root already on xfs (only `pquota` missing), 100 GiB+
free on docker's filesystem (setup's xfs image file), or a blank whole disk of 100 GiB+ that passes `lium gpu-splitting setup`'s target checks (not mounted, not swap, no holders, not
partitioned, writable, not removable or non-local, no signature). Partitions and free regions are never counted.
A host with less free space and no spare disk is not capable and keeps the STORAGE_LIMIT_MISSING warning only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

HELPERS = {
    "HOST_MOUNTS_PATH",
    "HOST_ROOT_PREFIX",
    "SYS_CLASS_BLOCK_PATH",
    "SYS_DEV_BLOCK_PATH",
    "ST_MODE_TYPE_MASK",
    "ST_MODE_BLOCK_DEVICE",
    "STORAGE_CAPABLE_MIN_BYTES",
    "SYSFS_SECTOR_BYTES",
    "PROC_SWAPS_PATH",
    "NOT_A_SPARE_DISK_PREFIXES",
    "NON_LOCAL_SYSFS_PARTS",
    "SIGNATURE_SPAN_BYTES",
    "carries_signature",
    "MountLine",
    "covering_mount",
    "kernel_name_of_device_node",
    "read_sysfs_text",
    "spare_block_device",
    "storage_limit_capable",
}
SECTORS_200_GB = 200 * 1024**3 // 512
ROOT_ON_SDA1 = "/dev/sda1 / ext4 rw,relatime 0 0\n"


@pytest.fixture
def scrape(tmp_path: Path) -> dict[str, Any]:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        HELPERS,
        {"os": os, "shutil": shutil, "docker_api_get": lambda path: {"DockerRootDir": "/var/lib/docker"}},
    )
    namespace["HOST_ROOT_PREFIX"] = str(tmp_path / "pid1-root")
    namespace["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")
    namespace["PROC_SWAPS_PATH"] = str(tmp_path / "swaps")
    namespace["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    (tmp_path / "class_block").mkdir()
    (tmp_path / "swaps").write_text("Filename Type Size Used Priority\n")
    return namespace


def device(scrape: dict[str, Any], name: str, sectors: int = SECTORS_200_GB, **files: str) -> Path:
    path = Path(scrape["SYS_CLASS_BLOCK_PATH"]) / name
    path.mkdir(parents=True)
    (path / "size").write_text(f"{sectors}\n")
    (path / "holders").mkdir()
    for file_name, text in files.items():
        (path / file_name).write_text(text)
    # the device node: a sparse all-zero file (a blank disk) of the same size
    node = Path(scrape["HOST_ROOT_PREFIX"]) / "dev" / name
    node.parent.mkdir(parents=True, exist_ok=True)
    with open(node, "wb") as node_file:
        node_file.truncate(sectors * 512)
    return path


def capable(scrape: dict[str, Any], mounts: str) -> bool:
    Path(scrape["HOST_MOUNTS_PATH"]).write_text(mounts)
    return scrape["storage_limit_capable"]()


def test_a_single_disk_host_is_not_capable(scrape: dict[str, Any]) -> None:
    sda = device(scrape, "sda")
    (sda / "sda1").mkdir()
    (sda / "sda1" / "partition").write_text("1\n")
    device(scrape, "sda1")

    assert capable(scrape, ROOT_ON_SDA1) is False


def test_docker_root_already_on_xfs_is_capable(scrape: dict[str, Any]) -> None:
    assert capable(scrape, ROOT_ON_SDA1 + "/dev/sdb /var/lib/docker xfs rw,relatime 0 0\n") is True


def test_an_unused_second_disk_is_capable(scrape: dict[str, Any]) -> None:
    device(scrape, "nvme1n1")

    assert capable(scrape, ROOT_ON_SDA1) is True
    assert scrape["spare_block_device"](ROOT_ON_SDA1) == "nvme1n1"


@pytest.mark.parametrize(
    "case",
    ["mounted", "swap", "lvm_member", "small", "read_only", "loop", "removable", "ext4", "end_signature", "no_node"],
)
def test_a_disk_in_use_or_unfit_is_not_spare(scrape: dict[str, Any], case: str) -> None:
    name = "loop0" if case == "loop" else "sdb"
    path = device(
        scrape,
        name,
        sectors=50 * 1000**3 // 512 if case == "small" else SECTORS_200_GB,
        **({"ro": "1\n"} if case == "read_only" else {"removable": "1\n"} if case == "removable" else {}),
    )
    node = Path(scrape["HOST_ROOT_PREFIX"]) / "dev" / name
    if case == "ext4":  # an unmounted ext4 partition: the superblock magic at byte 1080
        with open(node, "r+b") as node_file:
            node_file.seek(1080)
            node_file.write(b"\x53\xef")
    if case == "end_signature":  # a backup GPT header at the end of the disk
        with open(node, "r+b") as node_file:
            node_file.seek(SECTORS_200_GB * 512 - 512)
            node_file.write(b"EFI PART")
    if case == "no_node":
        node.unlink()
    if case == "lvm_member":
        (path / "holders" / "dm-0").mkdir()
    if case == "swap":
        Path(scrape["PROC_SWAPS_PATH"]).write_text(
            "Filename Type Size Used Priority\n/dev/sdb partition 1 0 -2\n"
        )
    mounts = ROOT_ON_SDA1 + ("/dev/sdb /data ext4 rw 0 0\n" if case == "mounted" else "")

    assert scrape["spare_block_device"](mounts) is None


def test_a_usb_disk_is_not_spare(scrape: dict[str, Any], tmp_path: Path) -> None:
    usb = tmp_path / "devices" / "pci0000:00" / "usb2" / "2-1" / "block" / "sdc"
    usb.parent.mkdir(parents=True)
    device(scrape, "sdc").rename(usb)
    (Path(scrape["SYS_CLASS_BLOCK_PATH"]) / "sdc").symlink_to(usb)

    assert scrape["spare_block_device"](ROOT_ON_SDA1) is None


@pytest.mark.parametrize("parent", ["sda", "sdb"])
def test_an_unused_partition_is_never_spare(scrape: dict[str, Any], parent: str) -> None:
    # setup refuses a partition on the root disk (protected) and decides others with a read-only mount,
    # so a blank partition, on the root disk or another one, does not make the host capable
    disk = device(scrape, parent)
    (disk / f"{parent}2").mkdir()
    (disk / f"{parent}2" / "partition").write_text("2\n")
    device(scrape, f"{parent}2", partition="2\n")

    assert scrape["spare_block_device"](ROOT_ON_SDA1) is None
    assert capable(scrape, ROOT_ON_SDA1) is False


def test_free_space_on_dockers_filesystem_is_capable(scrape: dict[str, Any]) -> None:
    # a single-disk host with room for the xfs image file `lium gpu-splitting setup` builds
    docker_root = Path(scrape["HOST_ROOT_PREFIX"]) / "var" / "lib" / "docker"
    docker_root.mkdir(parents=True)
    free = {"bytes": scrape["STORAGE_CAPABLE_MIN_BYTES"]}
    scrape["shutil"] = SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=free["bytes"]))

    assert capable(scrape, ROOT_ON_SDA1) is True
    free["bytes"] -= 1
    assert capable(scrape, ROOT_ON_SDA1) is False


PROBE_HELPERS = {
    "HOST_MOUNTS_PATH",
    "MountLine",
    "covering_mount",
    "XFS_PROJECT_QUOTA_OPTIONS",
    "docker_storage_limits_containers",
    "check_storage_limit_ability",
}
XFS_PQUOTA_DOCKER = "/dev/sdb /var/lib/docker xfs rw,relatime,attr2,inode64,prjquota 0 0\n"


def probe(tmp_path: Path, driver: str, mounts: str, run: Any) -> tuple[bool, str]:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {
            "subprocess": SimpleNamespace(run=run, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired),
            "docker_api_get": lambda path: {"Driver": driver, "DockerRootDir": "/var/lib/docker"},
            "COMMANDS": {"CHECK_STORAGE_LIMIT_ABILITY": ["docker", "run"]},
        },
    )
    namespace["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    (tmp_path / "mounts").write_text(ROOT_ON_SDA1 + mounts)
    return namespace["check_storage_limit_ability"]()


def passing_run(*args: Any, **kwargs: Any) -> Any:
    return SimpleNamespace(returncode=0)


def timed_out_run(*args: Any, **kwargs: Any) -> Any:
    raise subprocess.TimeoutExpired("docker", 30)


def test_overlay2_on_xfs_with_pquota_and_a_passing_probe_is_supported(tmp_path: Path) -> None:
    assert probe(tmp_path, "overlay2", XFS_PQUOTA_DOCKER, passing_run)[0] is True


@pytest.mark.parametrize(
    "driver, mounts",
    [
        ("overlayfs", XFS_PQUOTA_DOCKER),  # Docker 29's containerd image store passes the probe with no limit
        ("overlay2", "/dev/sdb /var/lib/docker xfs rw,relatime 0 0\n"),  # no project quota
        ("overlay2", ""),  # docker's root on the ext4 root disk
    ],
)
def test_a_passing_probe_without_overlay2_xfs_pquota_is_not_supported(
    tmp_path: Path, driver: str, mounts: str
) -> None:
    supported, detail = probe(tmp_path, driver, mounts, passing_run)

    assert supported is False
    assert "not overlay2 on xfs with pquota" in detail


def test_a_timed_out_probe_on_a_capped_setup_is_not_a_missing_limit(tmp_path: Path) -> None:
    supported, detail = probe(tmp_path, "overlay2", XFS_PQUOTA_DOCKER, timed_out_run)

    assert supported is True
    assert detail.startswith("Test command timed out")
