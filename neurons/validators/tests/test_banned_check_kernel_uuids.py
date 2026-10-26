"""DAH-2662 — bans match the kernel's GPU UUIDs, not only the ones the host reports.

2026-08-10 (ticket-0211): an `ld.so.preload` shim overrode nvmlDeviceGetUUID and reported the real
UUID with its last hex digit incremented. Everything read through NVML on that host is the
operator's choice; /proc/driver/nvidia/gpus/*/information is the kernel's and the shim cannot
author it (the DAH-2614 truth path). A ban keyed only on the reported UUID is void the moment the
banned operator re-registers with new UUIDs and a fresh hotkey.

Enforcement sits behind KERNEL_GPU_BAN_ENFORCEMENT_ENABLED (shadow by default, like every other
fatal-path widening in this repo): the tests that expect a ban through the kernel view flip it on.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from core.config import settings
from neurons.validators.src.services.task.checks.banned_provider import BannedProviderCheck
from neurons.validators.src.services.task.messages import BannedProviderMessages
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from services import nvidia_devices  # the module banned_provider.py reads the timeout from
from tests.helpers import build_state

REAL = "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17"  # what the kernel sees
SPOOFED = "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d18"  # what the shimmed NVML reports

# /proc/self/mountinfo as seen from inside a GPU container (1×A100 Lium pod, 15 Sep 2026):
# libnvidia-container's own tmpfs at /proc/driver/nvidia, then the real procfs bound per card.
CONTAINER_MOUNTS = [
    "1575 1532 0:126 / /proc/driver/nvidia rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,mode=555,inode64",
    "1619 1575 0:23 /driver/nvidia/gpus/0001:00:00.0 /proc/driver/nvidia/gpus/0001:00:00.0 "
    "ro,nosuid,nodev,noexec,relatime master:5 - proc proc rw",
]
# The 2026-08-19 kit (providerban ddae45d9: "tmpfs overlay on /proc/driver/nvidia/gpus"), as the same
# container shows it after `mount -t tmpfs none /proc/driver/nvidia/gpus/0001:00:00.0` (same pod).
OVERLAY_MOUNT = (
    "1424 1619 0:133 / /proc/driver/nvidia/gpus/0001:00:00.0 rw,relatime - tmpfs none "
    "rw,uid=231072,gid=231072,inode64"
)


INFO_FILE = "/proc/driver/nvidia/gpus/0001:00:00.0/information"


def _ssh_with_procfs(
    lines: list[str], mounts: list[str] = CONTAINER_MOUNTS, file_fs: list[str] | None = None
) -> AsyncMock:
    """The one round-trip read_kernel_gpu_view makes: mountinfo, `stat -f` per file, the UUID rows."""
    if file_fs is None:
        file_fs = [f"{INFO_FILE} proc" for _ in lines] + [
            f"{INFO_FILE}|regular file" for _ in lines
        ]
    stdout = "\n".join(
        [
            *mounts,
            nvidia_devices.KERNEL_GPU_FS_SEPARATOR,
            *file_fs,
            nvidia_devices.KERNEL_GPU_VIEW_SEPARATOR,
            *lines,
        ]
    )
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout=stdout, stderr=""))
    return ssh


@pytest.fixture
def enforcing(monkeypatch):
    monkeypatch.setattr(settings, "KERNEL_GPU_BAN_ENFORCEMENT_ENABLED", True)


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.setattr(settings, "KERNEL_GPU_BAN_ENFORCEMENT_ENABLED", False)


@pytest.mark.asyncio
async def test_provider_ban_catches_spoofed_uuid_through_the_kernel_view(
    context_factory, enforcing
):
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == BannedProviderMessages.PROVIDER_BANNED.reason
    assert result.event.what_we_saw["kernel_gpu_uuids"] == [REAL]
    assert result.event.what_we_saw["reported_uuids_differ_from_kernel"] is True
    assert result.updates["is_provider_banned"] is True


@pytest.mark.asyncio
async def test_kernel_list_read_through_an_overlay_is_not_trusted_and_fails_when_enforcing(
    context_factory, enforcing
):
    """The shim test: the overlay serves a clean UUID, the kernel list is withheld, the check fails."""
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented),
        ssh=_ssh_with_procfs([f"{SPOOFED}, 0"], mounts=[*CONTAINER_MOUNTS, OVERLAY_MOUNT]),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == BannedProviderMessages.KERNEL_GPU_VIEW_OVERLAID.reason
    assert result.event.what_we_saw["kernel_gpu_uuids"] is None
    assert result.event.what_we_saw["kernel_gpu_foreign_mounts"] == [
        "/proc/driver/nvidia/gpus/0001:00:00.0 tmpfs"
    ]
    assert "kernel_gpu_uuids" not in result.updates["state"].specs
    assert "is_provider_banned" not in result.updates


