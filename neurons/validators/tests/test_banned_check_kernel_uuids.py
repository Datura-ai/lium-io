"""DAH-2662 — bans match the kernel's GPU UUIDs, not only the ones the host reports.

2026-08-10 (ticket-0211): an `ld.so.preload` shim overrode nvmlDeviceGetUUID and reported the real
UUID with its last hex digit incremented. Everything read through NVML on that host is the
operator's choice; /proc/driver/nvidia/gpus/*/information is the kernel's and the shim cannot
author it (the DAH-2614 truth path). A ban keyed only on the reported UUID is void the moment the
banned operator re-registers with new UUIDs and a fresh hotkey.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from neurons.validators.src.services.task.checks.banned_gpu import BannedGpuCheck
from neurons.validators.src.services.task.checks.banned_provider import BannedProviderCheck
from neurons.validators.src.services.task.messages import BannedGpuMessages, BannedProviderMessages
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from tests.helpers import build_state

REAL = "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17"  # what the kernel sees
SPOOFED = "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d18"  # what the shimmed NVML reports


def _ssh_with_procfs(lines: list[str]):
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout="\n".join(lines), stderr=""))
    return ssh


@pytest.mark.asyncio
async def test_provider_ban_catches_spoofed_uuid_through_the_kernel_view(context_factory):
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
async def test_gpu_ban_catches_spoofed_uuid_through_the_kernel_view(context_factory):
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await BannedGpuCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == BannedGpuMessages.GPU_BANNED.reason
    assert result.updates["clear_verified_job_info"] is True


@pytest.mark.asyncio
async def test_kernel_identity_is_recorded_on_state_and_specs_for_the_backend(context_factory):
    rented = RentedExecutorsResponse(executors={})
    ctx = context_factory(
        state=build_state(gpu_uuids=REAL, rented_data=rented, specs={"gpu": {"count": 1}}),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].kernel_gpu_uuids == [REAL]
    assert result.updates["state"].specs == {"gpu": {"count": 1}, "kernel_gpu_uuids": [REAL]}
    assert result.event.what_we_saw["reported_uuids_differ_from_kernel"] is False


@pytest.mark.asyncio
async def test_unreadable_procfs_falls_back_to_reported_uuids_and_is_not_a_spoof(context_factory):
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids="GPU-honest", rented_data=rented), ssh=_ssh_with_procfs([])
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_gpu_uuids"] is None
    assert result.event.what_we_saw["reported_uuids_differ_from_kernel"] is False
    assert "state" not in result.updates


@pytest.mark.asyncio
async def test_gpu_ban_reuses_the_kernel_view_read_by_the_provider_check(context_factory):
    """One procfs read per cycle: BannedGpuCheck takes the list from state instead of ssh."""
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ssh = _ssh_with_procfs(["GPU-should-not-be-read, 0"])
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented, kernel_gpu_uuids=[REAL]), ssh=ssh
    )

    result = await BannedGpuCheck().run(ctx)

    assert result.passed is False
    ssh.run.assert_not_awaited()
