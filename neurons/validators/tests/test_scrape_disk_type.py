"""DAH-3674 — get_disk_type() in machine_scrape.py: nvme | ssd | hdd | unknown for the disk under docker's data root.

Two aggregators that list Lium (GPU Finder, rentgpu.org) filter on disk type; the specs carried the disk's size and
health and nothing about its kind. The scrape now takes the docker root's mount source off the host mount table, names
the block device by its major:minor (stat through /proc/1/root, then /sys/dev/block), walks a partition up to its whole
disk in sysfs, and reads the kernel's `rotational` flag. A reading, not a
verdict: nothing scores or gates on it, and anything the scrape cannot read is `unknown`, never a guess.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers are extracted by ast
and executed in their own namespace (same pattern as test_disk_health.py).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

DISK_TYPE_HELPERS = {
    "HOST_MOUNTS_PATH",
    "HOST_ROOT_PREFIX",
    "SYS_CLASS_BLOCK_PATH",
    "SYS_DEV_BLOCK_PATH",
    "ST_MODE_TYPE_MASK",
    "ST_MODE_BLOCK_DEVICE",
    "DISK_TYPE_NVME",
    "DISK_TYPE_SSD",
    "DISK_TYPE_HDD",
    "DISK_TYPE_UNKNOWN",
    "UNTYPED_DEVICE_PREFIXES",
    "covering_mount",
    "block_device_holding",
    "kernel_name_of_device_node",
    "whole_disk_of",
    "disk_type_of",
    "get_disk_type",
}

# /proc/1/mounts of a host whose docker root sits on its own NVMe disk; the root is a partition of a SATA disk.
HOST_MOUNTS = """
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
/dev/sda2 / ext4 rw,relatime,errors=remount-ro 0 0
/dev/nvme1n1 /var/lib/docker ext4 rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid,nodev,size=13158620k,mode=755 0 0
"""


@pytest.fixture
def scrape(tmp_path: Path) -> dict[str, Any]:
    """The disk-type helpers, executed in a namespace of their own, with PID 1's root pointed at an
    empty directory so no test resolves a device link through the real /proc/1/root."""
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", DISK_TYPE_HELPERS, {"os": os}
    )
    namespace["HOST_ROOT_PREFIX"] = str(tmp_path / "pid1-root")
    return namespace


def fake_disk(
    sysfs_root: Path, disk: str, rotational: str | None, partitions: tuple[str, ...] = ()
) -> None:
    """A whole disk and its partitions the way the kernel lays them out: /sys/class/block/<name> is a
    symlink into /sys/devices/..., a partition's target sits inside its disk's and carries `partition`."""
    disk_path = sysfs_root / "devices" / disk
    disk_path.mkdir(parents=True)
    if rotational is not None:
        (disk_path / "queue").mkdir()
        (disk_path / "queue" / "rotational").write_text(f"{rotational}\n")
    (sysfs_root / "class_block").mkdir(exist_ok=True)
    (sysfs_root / "class_block" / disk).symlink_to(disk_path)
    for partition in partitions:
        (disk_path / partition).mkdir()
        (disk_path / partition / "partition").write_text("1\n")
        (sysfs_root / "class_block" / partition).symlink_to(disk_path / partition)


# --------------------------------------------------------------------------------------------------
# the block device behind the docker root
# --------------------------------------------------------------------------------------------------
def test_the_covering_mounts_device_is_named(scrape: dict[str, Any]) -> None:
    assert scrape["block_device_holding"](HOST_MOUNTS, "/var/lib/docker") == "nvme1n1"


def test_a_docker_root_on_the_root_filesystem_reads_the_root_partition(
    scrape: dict[str, Any],
) -> None:
    # /mnt/docker is not a mount point of its own: the covering mount is `/`, on sda2
    assert scrape["block_device_holding"](HOST_MOUNTS, "/mnt/docker") == "sda2"


@pytest.mark.parametrize(
    "mounts",
    [
        "overlay / overlay rw,relatime,lowerdir=/a,upperdir=/b,workdir=/c 0 0\n",
        "tmpfs /var/lib/docker tmpfs rw 0 0\n",
        "10.0.0.5:/export /var/lib/docker nfs4 rw,relatime 0 0\n",
        "",
    ],
    ids=["overlay", "tmpfs", "nfs", "empty-table"],
)
def test_a_mount_that_is_not_a_dev_node_names_no_device(
    scrape: dict[str, Any], mounts: str
) -> None:
    assert scrape["block_device_holding"](mounts, "/var/lib/docker") is None


def fake_device_nodes(
    scrape: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    sysfs_root: Path,
    nodes: dict[str, tuple[int, int, str]],
) -> list[str]:
    """Device nodes under PID 1's root the way stat sees them: `nodes` maps a /dev path to its
    (major, minor, sysfs device directory). os.stat answers a block-device st_mode and that
    major:minor for the host-root-prefixed path and stats anything else for real (the sysfs walk
    goes through os.path.exists); /sys/dev/block gets the real `<major>:<minor>` link the kernel
    keeps there. Returns the paths under PID 1's root stat was asked for."""
    stat_calls: list[str] = []
    by_host_path = {f"{scrape['HOST_ROOT_PREFIX']}{path}": spec for path, spec in nodes.items()}
    real_stat = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(path, str) or not path.startswith(scrape["HOST_ROOT_PREFIX"]):
            return real_stat(path, *args, **kwargs)
        stat_calls.append(path)
        if path not in by_host_path:
            raise FileNotFoundError(2, "No such file or directory", path)
        major, minor, _ = by_host_path[path]
        return SimpleNamespace(st_mode=0o060660, st_rdev=os.makedev(major, minor))

    monkeypatch.setattr(os, "stat", stat)
    (sysfs_root / "dev_block").mkdir(exist_ok=True)
    for major, minor, device_dir in nodes.values():
        (sysfs_root / "dev_block" / f"{major}:{minor}").symlink_to(f"../../devices/{device_dir}")
    scrape["SYS_DEV_BLOCK_PATH"] = str(sysfs_root / "dev_block")
    return stat_calls


def test_a_by_uuid_link_resolves_to_the_kernel_name_by_its_device_number(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — the mount table names the fstab link, /dev/disk/by-uuid/…, the way systemd mounted
    # it. The link lives in the host's /dev (udev's), which this container's /dev has not; only a
    # stat through /proc/1/root reaches it
    stat_calls = fake_device_nodes(
        scrape,
        monkeypatch,
        tmp_path,
        {
            "/dev/disk/by-uuid/7d2c-uuid": (
                259,
                2,
                "pci0000:00/0000:00:1d.0/nvme/nvme0/nvme0n1/nvme0n1p2",
            )
        },
    )
    mounts = "/dev/disk/by-uuid/7d2c-uuid /var/lib/docker ext4 rw,relatime 0 0\n"

    # Act / Assert — named by 259:2, not by the link; and stat'ed through PID 1's root
    assert scrape["block_device_holding"](mounts, "/var/lib/docker") == "nvme0n1p2"
    assert stat_calls == [f"{scrape['HOST_ROOT_PREFIX']}/dev/disk/by-uuid/7d2c-uuid"]


def test_a_mapper_link_resolves_to_its_dm_device_and_types_as_unknown(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — an LVM docker root: the mount table names /dev/mapper/vg-docker, device-mapper 253:0
    fake_device_nodes(
        scrape, monkeypatch, tmp_path, {"/dev/mapper/vg-docker": (253, 0, "virtual/block/dm-0")}
    )
    fake_disk(tmp_path, "dm-0", "0")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")
    mounts = "/dev/mapper/vg-docker /var/lib/docker ext4 rw,relatime 0 0\n"

    # Act
    device = scrape["block_device_holding"](mounts, "/var/lib/docker")

    # Assert — the kernel name comes through, and a stacked device still reads as unknown
    assert device == "dm-0"
    assert scrape["disk_type_of"](device) == "unknown"


def test_a_partition_named_by_device_number_walks_up_to_its_disk_type(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — by-uuid partition of a SATA SSD: the kernel name from /sys/dev/block is the one
    # /sys/class/block indexes by, so the partition walk and the rotational read follow from it
    fake_device_nodes(
        scrape, monkeypatch, tmp_path, {"/dev/disk/by-uuid/9f3e-uuid": (8, 1, "sda/sda1")}
    )
    fake_disk(tmp_path, "sda", "0", ("sda1",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")
    mounts = "/dev/disk/by-uuid/9f3e-uuid /var/lib/docker ext4 rw,relatime 0 0\n"

    assert (
        scrape["disk_type_of"](scrape["block_device_holding"](mounts, "/var/lib/docker")) == "ssd"
    )


def test_a_stat_that_fails_or_finds_no_block_device_keeps_the_name_as_mounted(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — outside the executor container /proc/1/root is another user's (EPERM); a node sysfs
    # does not list (no /sys/dev/block entry); a source that is no block device at all
    def denied(path: str, *args: Any, **kwargs: Any) -> SimpleNamespace:
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(os, "stat", denied)
    assert scrape["block_device_holding"](HOST_MOUNTS, "/var/lib/docker") == "nvme1n1"

    fake_device_nodes(scrape, monkeypatch, tmp_path, {})
    monkeypatch.setattr(
        os,
        "stat",
        lambda path, *a, **k: SimpleNamespace(st_mode=0o060660, st_rdev=os.makedev(259, 9)),
    )
    assert scrape["block_device_holding"](HOST_MOUNTS, "/var/lib/docker") == "nvme1n1"

    monkeypatch.setattr(
        os, "stat", lambda path, *a, **k: SimpleNamespace(st_mode=0o100644, st_rdev=0)
    )
    assert scrape["block_device_holding"](HOST_MOUNTS, "/var/lib/docker") == "nvme1n1"

    # a by-uuid link on the fallback path is a name sysfs does not know: unknown downstream, never a guess
    mounts = "/dev/disk/by-uuid/7d2c-uuid /var/lib/docker ext4 rw,relatime 0 0\n"
    assert scrape["block_device_holding"](mounts, "/var/lib/docker") == "7d2c-uuid"
    assert scrape["disk_type_of"]("7d2c-uuid") == "unknown"


# --------------------------------------------------------------------------------------------------
# partition -> whole disk
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("disk", "partition"),
    [("nvme0n1", "nvme0n1p1"), ("sda", "sda1"), ("mmcblk0", "mmcblk0p1")],
)
def test_a_partition_walks_up_to_its_whole_disk(
    scrape: dict[str, Any], tmp_path: Path, disk: str, partition: str
) -> None:
    fake_disk(tmp_path, disk, "0", (partition,))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["whole_disk_of"](partition) == disk
    assert scrape["whole_disk_of"](disk) == disk


def test_a_device_sysfs_does_not_list_is_returned_as_it_is(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # no `partition` file to read, so nothing says it is one — and the caller's rotational read then fails
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["whole_disk_of"]("nvme9n9") == "nvme9n9"


# --------------------------------------------------------------------------------------------------
# the type
# --------------------------------------------------------------------------------------------------
def test_an_nvme_device_is_nvme_by_name(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — rotational says 0 too, but the name decides first: NVMe is more than "solid state"
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p1",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("nvme0n1p1") == "nvme"
    assert scrape["disk_type_of"]("nvme0n1") == "nvme"


def test_rotational_0_is_ssd_and_1_is_hdd(scrape: dict[str, Any], tmp_path: Path) -> None:
    fake_disk(tmp_path, "sda", "0", ("sda1",))
    fake_disk(tmp_path, "sdb", "1", ("sdb1",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("sda1") == "ssd"
    assert scrape["disk_type_of"]("sdb1") == "hdd"


@pytest.mark.parametrize(
    ("device", "rotational"),
    [
        ("dm-0", "0"),
        ("md127", "0"),
        ("loop3", "1"),
        ("nbd0", "0"),
        ("rbd0", "0"),
        ("drbd0", "0"),
        ("vda1", "1"),
        ("xvda1", "1"),
    ],
)
def test_a_stacked_or_virtual_device_is_unknown_not_its_own_rotational_flag(
    scrape: dict[str, Any], tmp_path: Path, device: str, rotational: str
) -> None:
    # Arrange — the flag describes the virtual device, not the disks behind it: a virtio `vda`
    # reports 1 whatever backs it (this VM, 19 Sep 2026), so a CVM on NVMe would read as hdd
    disk = device.rstrip("0123456789") if device.startswith(("vd", "xvd")) else device
    fake_disk(tmp_path, disk, rotational, (device,) if disk != device else ())
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"](device) == "unknown"


@pytest.mark.parametrize(
    ("rotational", "case"),
    [
        (None, "no queue/rotational file"),
        ("", "empty file"),
        ("2", "a value the kernel does not define"),
    ],
)
def test_an_unreadable_or_odd_rotational_flag_is_unknown(
    scrape: dict[str, Any], tmp_path: Path, rotational: str | None, case: str
) -> None:
    fake_disk(tmp_path, "sdc", rotational)
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("sdc") == "unknown", case


def test_no_device_is_unknown(scrape: dict[str, Any]) -> None:
    assert scrape["disk_type_of"](None) == "unknown"
    assert scrape["disk_type_of"]("") == "unknown"


# --------------------------------------------------------------------------------------------------
# get_disk_type: the docker root, the mount table, sysfs, end to end
# --------------------------------------------------------------------------------------------------
def test_get_disk_type_reads_the_disk_under_the_docker_root(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — docker's data root on a spinning SATA disk while the root filesystem is NVMe
    (tmp_path / "mounts").write_text(
        "/dev/nvme0n1p2 / ext4 rw,relatime 0 0\n/dev/sda1 /data/docker ext4 rw,relatime 0 0\n"
    )
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p2",))
    fake_disk(tmp_path, "sda", "1", ("sda1",))
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")
    scrape["docker_api_get"] = lambda path: {"DockerRootDir": "/data/docker"}

    # Act / Assert
    assert scrape["get_disk_type"]() == "hdd"


def test_get_disk_type_falls_back_to_var_lib_docker_when_the_docker_socket_is_down(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    (tmp_path / "mounts").write_text("/dev/nvme0n1p2 / ext4 rw,relatime 0 0\n")
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p2",))
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    def socket_down(path: str) -> dict:
        raise ConnectionError("docker socket unreachable")

    scrape["docker_api_get"] = socket_down

    assert scrape["get_disk_type"]() == "nvme"


# --------------------------------------------------------------------------------------------------
# the wire: the key survives the obfuscation rename and lands as hard_disk.disk_type
# --------------------------------------------------------------------------------------------------
def test_the_disk_type_key_is_registered_in_both_obfuscation_tables() -> None:
    # Arrange — ecrypt_miner_job_files renames keys by sequential str.replace over the source, and
    # MachineSpecScrapeCheck maps them back through ORIGINAL_KEYS; a key missing from either table
    # is dropped on the wire, silently
    scrape_source = ast.parse((SRC / "miner_jobs" / "machine_scrape.py").read_text())
    emitted = {
        node.value
        for node in ast.walk(scrape_source)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("hard_disk_")
    }
    service = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    original_table = next(
        ast.literal_eval(node.value)
        for node in ast.walk(service)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ORIGINAL_KEYS"
    )
    mapped_keys: list[str] = []
    for node in ast.walk(service):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_key_mappings":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    mapped_keys = [key.value for key in child.keys if isinstance(key, ast.Constant)]
                    break

    # Assert — emitted, mapped to the name lium-platform's HardDiskSpec declares, and renamed before
    # any key it contains
    assert "hard_disk_disk_type" in emitted
    assert original_table["hard_disk_disk_type"] == "disk_type"
    assert "hard_disk_disk_type" in mapped_keys
    for index, key in enumerate(mapped_keys):
        longer_later = [
            other for other in mapped_keys[index + 1 :] if key != other and key in other
        ]
        assert not longer_later, f"{key!r} is renamed before {longer_later} and would corrupt them"
