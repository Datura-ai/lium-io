"""DAH-1997 (26 Oct plan): the storage limit and who must build it, as machine_scrape.py decides.

The limit counts only for overlay2 on xfs mounted with pquota (and a `--storage-opt size=` container that runs);
a probe that timed out gives no verdict (None), not "no limit". A host is capable of building the limit when
`lium gpu-splitting setup` could place its xfs image file: on the local filesystem with the most free space, other
than docker's own mount, 100 GiB left after a reserve of max(10%, 20 GiB); otherwise it keeps the warning only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

HELPERS = {
    "HOST_MOUNTS_PATH",
    "COMMANDS",
    "PROJECT_QUOTA_MOUNT_OPTIONS",
    "MountLine",
    "covering_mount",
    "storage_limit_setup_gap",
    "check_storage_limit_ability",
    "STORAGE_CAPABLE_MIN_BYTES",
    "STORAGE_IMAGE_FILE_RESERVE_FRACTION",
    "STORAGE_IMAGE_FILE_MIN_RESERVE_BYTES",
    "STORAGE_IMAGE_FILE_FSTYPES",
    "HOST_ROOT_PREFIX",
    "image_file_room",
    "storage_limit_capable",
}
OVERLAY2 = {"Driver": "overlay2", "DockerRootDir": "/var/lib/docker"}
ROOT = "/dev/sda1 / ext4 rw,relatime 0 0\n"
DOCKER_XFS_PQUOTA = ROOT + "/dev/sdb /var/lib/docker xfs rw,relatime,prjquota 0 0\n"


def namespace(tmp_path: Path, mounts: str, info: dict, run=None) -> dict[str, Any]:
    scrape = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        HELPERS,
        {
            "subprocess": SimpleNamespace(
                run=run or (lambda *a, **k: SimpleNamespace(returncode=0)),
                PIPE=subprocess.PIPE,
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
            "docker_api_get": lambda path: info,
        },
    )
    scrape["HOST_MOUNTS_PATH"] = str(tmp_path / "mounts")
    (tmp_path / "mounts").write_text(mounts)
    return scrape


@pytest.mark.parametrize(
    "mounts",
    [
        DOCKER_XFS_PQUOTA,
        ROOT.replace("ext4 rw", "xfs rw,pquota"),
    ],  # own mount, or docker under an xfs /
)
def test_overlay2_on_xfs_with_pquota_whose_container_runs_is_supported(tmp_path, mounts):
    assert namespace(tmp_path, mounts, OVERLAY2)["check_storage_limit_ability"]() == (
        True,
        "Storage limit is supported.",
    )


@pytest.mark.parametrize(
    "info, mounts, missing",
    [
        ({**OVERLAY2, "Driver": "zfs"}, DOCKER_XFS_PQUOTA, "not overlay2"),
        (OVERLAY2, ROOT + "/dev/sdb /var/lib/docker ext4 rw,prjquota 0 0\n", "not xfs"),
        (OVERLAY2, ROOT + "/dev/sdb /var/lib/docker xfs rw,relatime 0 0\n", "pquota"),
        (OVERLAY2, ROOT + "/dev/sdb /var/lib/docker xfs rw,pqnoenforce 0 0\n", "pquota"),
    ],
)
def test_a_missing_driver_filesystem_or_pquota_is_not_supported_even_if_the_container_runs(
    tmp_path, info, mounts, missing
):
    supported, detail = namespace(tmp_path, mounts, info)["check_storage_limit_ability"]()

    assert supported is False
    assert missing in detail


def test_a_container_that_does_not_run_is_not_supported(tmp_path):
    scrape = namespace(
        tmp_path, DOCKER_XFS_PQUOTA, OVERLAY2, run=lambda *a, **k: SimpleNamespace(returncode=125)
    )

    assert scrape["check_storage_limit_ability"]() == (False, "Storage limit is not supported.")


def test_a_timed_out_probe_gives_no_verdict(tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 30)

    scrape = namespace(tmp_path, DOCKER_XFS_PQUOTA, OVERLAY2, run=timeout)

    assert scrape["check_storage_limit_ability"]() == (None, "Test command timed out.")


GIB = 1024**3
ROOT_ONLY = "/dev/sda1 / ext4 rw 0 0\n"
DOCKER_OWN_DISK = ROOT_ONLY + "/dev/sdb /var/lib/docker ext4 rw 0 0\n"


@pytest.mark.parametrize(
    "mounts, filesystems, capable",
    [
        (ROOT_ONLY, {"/": (500, 150)}, True),  # 150 free - 50 reserve (10%) = 100 GiB
        (ROOT_ONLY, {"/": (500, 149)}, False),
        (ROOT_ONLY, {"/": (150, 120)}, True),  # 120 free - 20 GiB minimum reserve = 100 GiB
        (ROOT_ONLY, {"/": (150, 119)}, False),
        # docker's own mount is where the image file gets mounted, never where it is put
        (DOCKER_OWN_DISK, {"/": (150, 10), "/var/lib/docker": (2000, 1500)}, False),
        (
            "/dev/sda1 / btrfs rw 0 0\n",
            {"/": (2000, 1500)},
            False,
        ),  # copy-on-write: fallocate reserves nothing
        (
            "tmpfs /tmp tmpfs rw 0 0\n" + ROOT_ONLY,
            {"/": (150, 120)},
            True,
        ),  # not a block device: skipped
        # setup takes the filesystem with the most free space: 200 free of 2000 leaves 0 after the 200 reserve
        (
            ROOT_ONLY + "/dev/sdc /data ext4 rw 0 0\n",
            {"/": (2000, 200), "/data": (150, 130)},
            False,
        ),
    ],
)
def test_capable_is_setups_image_file_rule(tmp_path, mounts, filesystems, capable):
    scrape = namespace(tmp_path, mounts, OVERLAY2)
    scrape["HOST_ROOT_PREFIX"] = "/host"
    statvfs = {
        f"/host{point}": SimpleNamespace(f_blocks=size * GIB, f_bavail=free * GIB, f_frsize=1)
        for point, (size, free) in filesystems.items()
    }
    scrape["os"] = SimpleNamespace(statvfs=statvfs.__getitem__)

    assert scrape["storage_limit_capable"]() is capable
