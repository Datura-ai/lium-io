"""DAH-3674 / DAH-3746 — get_docker_root_disk_type() in machine_scrape.py: nvme | ssd | hdd | unknown for the disk under
docker's data root.

Two aggregators that list Lium (GPU Finder, rentgpu.org) filter on disk type; the specs carried the disk's size and
health and nothing about its kind. The scrape now takes the docker root's mount source off the host mount table, names
the block device by its major:minor (stat through /proc/1/root, then /sys/dev/block), walks a partition up to its whole
disk in sysfs, follows a stacked device (LVM, md RAID) through `slaves/` to the physical disks, and reads the kernel's
`rotational` flag. A virtual machine's disk is unknown: its flag describes the emulated controller (DAH-3746: 34 of 35
QEMU/DigitalOcean `sda` VMs read `hdd`). A reading, not a verdict: nothing scores or gates on it, and anything the scrape
cannot read is `unknown`, never a guess.

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
    "SYS_DMI_ID_PATH",
    "SYS_HYPERVISOR_TYPE_PATH",
    "ST_MODE_TYPE_MASK",
    "ST_MODE_BLOCK_DEVICE",
    "DISK_TYPE_NVME",
    "DISK_TYPE_SSD",
    "DISK_TYPE_HDD",
    "DISK_TYPE_UNKNOWN",
    "UNTYPED_DEVICE_PREFIXES",
    "STACKED_DEVICE_MAX_DEPTH",
    "VIRTUAL_DISK_PREFIXES",
    "VIRTUAL_DISK_IDENTITY_MARKERS",
    "VIRTUAL_MACHINE_DMI_MARKERS",
    "MountLine",
    "covering_mount",
    "block_device_holding",
    "kernel_name_of_device_node",
    "whole_disk_of",
    "read_sysfs_text",
    "host_is_a_virtual_machine",
    "disk_is_virtual",
    "slaves_of",
    "slowest_disk_type",
    "disk_type_following_slaves",
    "disk_type_of",
    "get_docker_root_disk_type",
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
    empty directory so no test resolves a device link through the real /proc/1/root, and the DMI /
    hypervisor paths at files that do not exist - a bare-metal host until a test says otherwise
    (the CI runner itself is a VM)."""
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", DISK_TYPE_HELPERS, {"os": os}
    )
    namespace["HOST_ROOT_PREFIX"] = str(tmp_path / "pid1-root")
    namespace["SYS_DMI_ID_PATH"] = str(tmp_path / "dmi_id")
    namespace["SYS_HYPERVISOR_TYPE_PATH"] = str(tmp_path / "hypervisor_type")
    return namespace


def fake_disk(
    sysfs_root: Path,
    disk: str,
    rotational: str | None,
    partitions: tuple[str, ...] = (),
    slaves: tuple[str, ...] = (),
    vendor: str | None = None,
    model: str | None = None,
) -> None:
    """A whole disk and its partitions the way the kernel lays them out: /sys/class/block/<name> is a
    symlink into /sys/devices/..., a partition's target sits inside its disk's and carries `partition`,
    a stacked device lists what it is built on under `slaves/`, and a SCSI/ATA disk names its maker
    under `device/vendor` + `device/model` (padded the way the kernel prints them)."""
    disk_path = sysfs_root / "devices" / disk
    disk_path.mkdir(parents=True)
    if rotational is not None:
        (disk_path / "queue").mkdir()
        (disk_path / "queue" / "rotational").write_text(f"{rotational}\n")
    (disk_path / "slaves").mkdir()
    for slave in slaves:
        (disk_path / "slaves" / slave).symlink_to(sysfs_root / "devices" / slave)
    if vendor is not None or model is not None:
        (disk_path / "device").mkdir()
        (disk_path / "device" / "vendor").write_text(f"{vendor or '':<8}\n")
        (disk_path / "device" / "model").write_text(f"{model or '':<16}\n")
    (sysfs_root / "class_block").mkdir(exist_ok=True)
    (sysfs_root / "class_block" / disk).symlink_to(disk_path)
    for partition in partitions:
        (disk_path / partition).mkdir()
        (disk_path / partition / "partition").write_text("1\n")
        (sysfs_root / "class_block" / partition).symlink_to(disk_path / partition)


def fake_dmi(sysfs_root: Path, sys_vendor: str, product_name: str) -> None:
    """/sys/class/dmi/id/{sys_vendor,product_name} as the firmware reports them."""
    (sysfs_root / "dmi_id").mkdir(exist_ok=True)
    (sysfs_root / "dmi_id" / "sys_vendor").write_text(f"{sys_vendor}\n")
    (sysfs_root / "dmi_id" / "product_name").write_text(f"{product_name}\n")


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


def test_a_mapper_link_resolves_to_its_dm_device_and_types_by_the_disk_under_it(
    scrape: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — an LVM docker root: the mount table names /dev/mapper/vg-docker, device-mapper 253:0,
    # built on a partition of an NVMe disk
    fake_device_nodes(
        scrape, monkeypatch, tmp_path, {"/dev/mapper/vg-docker": (253, 0, "virtual/block/dm-0")}
    )
    fake_disk(tmp_path, "dm-0", "0", slaves=("nvme0n1p3",))
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p3",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")
    mounts = "/dev/mapper/vg-docker /var/lib/docker ext4 rw,relatime 0 0\n"

    # Act
    device = scrape["block_device_holding"](mounts, "/var/lib/docker")

    # Assert — the kernel name comes through, and the stacked device reads as the disk beneath it
    assert device == "dm-0"
    assert scrape["disk_type_of"](device) == "nvme"


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
def test_a_device_with_no_physical_disk_to_follow_is_unknown_not_its_own_rotational_flag(
    scrape: dict[str, Any], tmp_path: Path, device: str, rotational: str
) -> None:
    # Arrange — the flag describes the virtual device, not the disks behind it: a virtio `vda`
    # reports 1 whatever backs it (this VM, 19 Sep 2026), so a CVM on NVMe would read as hdd; a
    # dm/md device that lists no slaves has nothing to follow
    disk = device.rstrip("0123456789") if device.startswith(("vd", "xvd")) else device
    fake_disk(tmp_path, disk, rotational, (device,) if disk != device else ())
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"](device) == "unknown"


# --------------------------------------------------------------------------------------------------
# DAH-3746: a virtual machine's disk is unknown, whatever `rotational` says
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("vendor", "model"),
    [
        ("QEMU", "QEMU HARDDISK"),  # QEMU virtio-scsi / SCSI emulation
        ("ATA", "QEMU HARDDISK"),  # QEMU SATA/IDE emulation: udev's ID_BUS=ata, vendor reads ATA
        ("DO", "Volume"),  # a DigitalOcean block-storage volume
        ("VBOX", "HARDDISK"),
        ("VMware", "Virtual disk"),
        ("Msft", "Virtual Disk"),  # Hyper-V / Azure
        ("Google", "PersistentDisk"),
    ],
    ids=["qemu-scsi", "qemu-ata", "digitalocean-volume", "virtualbox", "vmware", "hyper-v", "gce"],
)
def test_an_sda_whose_vendor_or_model_names_a_hypervisor_is_unknown_not_hdd(
    scrape: dict[str, Any], tmp_path: Path, vendor: str, model: str
) -> None:
    # Arrange — the emulated SATA disk reports rotational=1 (34 of 35 checked VMs read hdd);
    # no DMI at all, so the disk's own identity is the only tell
    fake_disk(tmp_path, "sda", "1", ("sda1",), vendor=vendor, model=model)
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("sda1") == "unknown"
    assert scrape["disk_is_virtual"]("sda") is True


@pytest.mark.parametrize(
    ("sys_vendor", "product_name"),
    [
        ("QEMU", "Standard PC (Q35 + ICH9, 2009)"),
        ("Red Hat", "KVM"),
        ("DigitalOcean", "Droplet"),
        ("Amazon EC2", "g5.xlarge"),
        ("Google", "Google Compute Engine"),
        ("Microsoft Corporation", "Virtual Machine"),
        ("VMware, Inc.", "VMware Virtual Platform"),
        ("Xen", "HVM domU"),
    ],
    ids=["qemu", "kvm", "digitalocean", "ec2", "gce", "hyper-v", "vmware", "xen"],
)
def test_a_host_whose_dmi_names_a_hypervisor_types_every_disk_as_unknown(
    scrape: dict[str, Any], tmp_path: Path, sys_vendor: str, product_name: str
) -> None:
    # Arrange — the disk itself carries no tell (a bare `sda`, no device/vendor), rotational=1;
    # only the firmware says the machine is a VM
    fake_dmi(tmp_path, sys_vendor, product_name)
    fake_disk(tmp_path, "sda", "1", ("sda1",))
    fake_disk(tmp_path, "nvme0n1", "0")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["host_is_a_virtual_machine"]() is True
    assert scrape["disk_type_of"]("sda1") == "unknown"
    # an NVMe name under a hypervisor is an emulated controller too (EBS, GCE pd) - not a reading
    assert scrape["disk_type_of"]("nvme0n1") == "unknown"


def test_a_sys_hypervisor_type_file_marks_the_host_as_a_vm(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — Xen PV/HVM guests carry /sys/hypervisor/type and may leave DMI blank
    (tmp_path / "hypervisor_type").write_text("xen\n")
    fake_disk(tmp_path, "sda", "1")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["host_is_a_virtual_machine"]() is True
    assert scrape["disk_type_of"]("sda") == "unknown"


@pytest.mark.parametrize(
    ("sys_vendor", "product_name", "vendor", "model", "rotational", "expected"),
    [
        ("Supermicro", "SYS-420GP-TNR", "ATA", "Samsung SSD 870", "0", "ssd"),
        ("Dell Inc.", "PowerEdge R750xa", "SEAGATE", "ST16000NM000J", "1", "hdd"),
        ("ASUSTeK COMPUTER INC.", "ESC8000A-E12", "ATA", "WDC WD40EFRX", "1", "hdd"),
    ],
    ids=["supermicro-ssd", "dell-hdd", "asus-hdd"],
)
def test_a_bare_metal_host_keeps_reading_its_rotational_flag(
    scrape: dict[str, Any],
    tmp_path: Path,
    sys_vendor: str,
    product_name: str,
    vendor: str,
    model: str,
    rotational: str,
    expected: str,
) -> None:
    # Negative control: a real board maker in DMI and a real disk maker on the device leave the
    # reading exactly where DAH-3674 put it
    fake_dmi(tmp_path, sys_vendor, product_name)
    fake_disk(tmp_path, "sda", rotational, ("sda1",), vendor=vendor, model=model)
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["host_is_a_virtual_machine"]() is False
    assert scrape["disk_is_virtual"]("sda") is False
    assert scrape["disk_type_of"]("sda1") == expected


def test_an_unreadable_dmi_directory_reads_as_bare_metal(scrape: dict[str, Any]) -> None:
    # the fixture points the DMI and hypervisor paths at nothing: no tell, no VM verdict
    assert scrape["host_is_a_virtual_machine"]() is False


# --------------------------------------------------------------------------------------------------
# DAH-3746: a stacked device (LVM, dm-crypt, md RAID) is typed by the physical disks under it
# --------------------------------------------------------------------------------------------------
def test_lvm_over_a_partition_of_an_nvme_disk_reads_nvme(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — /dev/mapper/vg-root is dm-0, whose slaves/ lists nvme0n1p3 (the PV is a partition)
    fake_disk(tmp_path, "dm-0", "0", slaves=("nvme0n1p3",))
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p3",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["slaves_of"]("dm-0") == ["nvme0n1p3"]
    assert scrape["disk_type_of"]("dm-0") == "nvme"


def test_lvm_over_a_spinning_disk_reads_hdd(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — the whole disk is the PV, so slaves/ names sdb itself; dm-0's own flag says 0
    fake_disk(tmp_path, "dm-0", "0", slaves=("sdb",))
    fake_disk(tmp_path, "sdb", "1")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("dm-0") == "hdd"


def test_md_raid_over_two_ssds_reads_ssd(scrape: dict[str, Any], tmp_path: Path) -> None:
    fake_disk(tmp_path, "md127", "0", ("md127p1",), slaves=("sda2", "sdb2"))
    fake_disk(tmp_path, "sda", "0", ("sda2",))
    fake_disk(tmp_path, "sdb", "0", ("sdb2",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    # the docker root sits on a partition of the array: partition -> md127 -> its members
    assert scrape["disk_type_of"]("md127p1") == "ssd"


def test_a_mixed_stack_reads_as_its_slowest_member(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — a volume group spanning an NVMe disk and a spinning one: the LV delivers what the
    # spinning disk does
    fake_disk(tmp_path, "dm-0", "0", slaves=("nvme0n1", "sdb1"))
    fake_disk(tmp_path, "nvme0n1", "0")
    fake_disk(tmp_path, "sdb", "1", ("sdb1",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("dm-0") == "hdd"
    assert scrape["slowest_disk_type"](["nvme", "ssd"]) == "ssd"
    assert scrape["slowest_disk_type"](["nvme", "unknown"]) == "nvme"
    assert scrape["slowest_disk_type"](["unknown", "unknown"]) == "unknown"
    assert scrape["slowest_disk_type"]([]) == "unknown"


def test_a_stack_is_followed_through_several_layers(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — LVM (dm-1) over LUKS (dm-0) over an md mirror over two NVMe partitions
    fake_disk(tmp_path, "dm-1", "0", slaves=("dm-0",))
    fake_disk(tmp_path, "dm-0", "0", slaves=("md0",))
    fake_disk(tmp_path, "md0", "0", slaves=("nvme0n1p2", "nvme1n1p2"))
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p2",))
    fake_disk(tmp_path, "nvme1n1", "0", ("nvme1n1p2",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("dm-1") == "nvme"


def test_a_stack_whose_members_none_resolve_is_unknown(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # dm-crypt over a loop file: the slave has no physical disk, so nothing resolves
    fake_disk(tmp_path, "dm-0", "0", slaves=("loop0",))
    fake_disk(tmp_path, "loop0", "1")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("dm-0") == "unknown"


def test_a_slave_cycle_ends_at_the_depth_cap_as_unknown(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # a sysfs that lists dm-0 under dm-1's slaves and dm-1 under dm-0's cannot happen on a sound
    # kernel; the cap turns it into unknown rather than a RecursionError that loses the hard_disk block
    fake_disk(tmp_path, "dm-0", "0", slaves=("dm-1",))
    fake_disk(tmp_path, "dm-1", "0", slaves=("dm-0",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("dm-0") == "unknown"


def test_a_vm_disk_under_a_stack_is_still_unknown(scrape: dict[str, Any], tmp_path: Path) -> None:
    # LVM on a DigitalOcean droplet: dm-0 over vda1 - the member is virtual, so nothing resolves
    fake_disk(tmp_path, "dm-0", "0", slaves=("vda1",))
    fake_disk(tmp_path, "vda", "1", ("vda1",))
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    assert scrape["disk_type_of"]("dm-0") == "unknown"


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
# get_docker_root_disk_type: the docker root, the mount table, sysfs, end to end
# --------------------------------------------------------------------------------------------------
def test_get_docker_root_disk_type_reads_the_disk_under_the_docker_root(
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
    assert scrape["get_docker_root_disk_type"]() == "hdd"


def test_get_docker_root_disk_type_on_a_qemu_vm_is_unknown_end_to_end(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — the prod shape DAH-3746 found: a QEMU guest, docker root on /, / on sda1, sda's
    # rotational flag 1 (read as hdd on 34 of 35 such nodes)
    (tmp_path / "mounts").write_text("/dev/sda1 / ext4 rw,relatime 0 0\n")
    fake_dmi(tmp_path, "QEMU", "Standard PC (i440FX + PIIX, 1996)")
    fake_disk(tmp_path, "sda", "1", ("sda1",), vendor="QEMU", model="QEMU HARDDISK")
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")
    scrape["docker_api_get"] = lambda path: {"DockerRootDir": "/var/lib/docker"}

    assert scrape["get_docker_root_disk_type"]() == "unknown"


def test_get_docker_root_disk_type_falls_back_to_var_lib_docker_when_the_docker_socket_is_down(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    (tmp_path / "mounts").write_text("/dev/nvme0n1p2 / ext4 rw,relatime 0 0\n")
    fake_disk(tmp_path, "nvme0n1", "0", ("nvme0n1p2",))
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    scrape["SYS_CLASS_BLOCK_PATH"] = str(tmp_path / "class_block")

    def socket_down(path: str) -> dict:
        raise ConnectionError("docker socket unreachable")

    scrape["docker_api_get"] = socket_down

    assert scrape["get_docker_root_disk_type"]() == "nvme"


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
