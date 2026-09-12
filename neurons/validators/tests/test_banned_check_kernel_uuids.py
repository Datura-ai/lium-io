"""DAH-2662 — bans match the kernel's GPU UUIDs, not only the ones the host reports.

2026-08-10 (ticket-0211): an `ld.so.preload` shim overrode nvmlDeviceGetUUID and reported the real
UUID with its last hex digit incremented. Everything read through NVML on that host is the
operator's choice; /proc/driver/nvidia/gpus/*/information is the kernel's and the shim cannot
author it (the DAH-2614 truth path). A ban keyed only on the reported UUID is void the moment the
banned operator re-registers with new UUIDs and a fresh hotkey.

Enforcement sits behind KERNEL_GPU_BAN_ENFORCEMENT_ENABLED (shadow by default, like every other
fatal-path widening in this repo): the tests that expect a ban through the kernel view flip it on.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from core.config import settings
from neurons.validators.src.services.task.checks.banned_gpu import BannedGpuCheck
from neurons.validators.src.services.task.checks.banned_provider import BannedProviderCheck
from neurons.validators.src.services.task.messages import BannedGpuMessages, BannedProviderMessages
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from services import nvidia_devices  # the module banned_provider.py reads the timeout from
from tests.helpers import build_state

REAL = "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d17"  # what the kernel sees
SPOOFED = "GPU-f2bfa67f-5281-aabc-aa90-c91764f90d18"  # what the shimmed NVML reports


def _ssh_with_procfs(lines: list[str]):
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=MagicMock(exit_status=0, stdout="\n".join(lines), stderr=""))
    return ssh


@pytest.fixture
def enforcing(monkeypatch):
    monkeypatch.setattr(settings, "KERNEL_GPU_BAN_ENFORCEMENT_ENABLED", True)


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.setattr(settings, "KERNEL_GPU_BAN_ENFORCEMENT_ENABLED", False)


def _after(ctx, result):
    """The pipeline's handoff: the next check runs on the context with this result's updates applied."""
    return ctx.model_copy(update=result.updates)


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
@pytest.mark.parametrize(
    ("check", "rented"),
    [
        (
            BannedProviderCheck(),
            RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL]),
        ),
        (BannedGpuCheck(), RentedExecutorsResponse(executors={}, banned_guids=[REAL])),
    ],
    ids=["provider", "gpu"],
)
async def test_shadow_mode_bans_on_the_reported_list_only_and_says_what_the_flip_changes(
    context_factory, shadow, check, rented
):
    """Default (enforcement off): a kernel-only match passes, exactly as on main, and the event says so."""
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await check.run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_gpu_uuids"] == [REAL]
    assert result.event.what_we_saw["kernel_view_would_ban"] is True


@pytest.mark.asyncio
async def test_shadow_mode_still_records_the_kernel_view_on_specs(context_factory, shadow):
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.event.what_we_saw["kernel_ban_enforced"] is False
    assert result.updates["state"].specs["kernel_gpu_uuids"] == [REAL]


@pytest.mark.asyncio
async def test_kernel_view_would_ban_counts_only_gpu_bans_not_hotkey_bans(context_factory, shadow):
    """The shadow-week signal measures what the flip changes; a hotkey ban applies either way."""
    rented = RentedExecutorsResponse(executors={}, banned_hotkeys=["miner-hotkey"])
    ctx = context_factory(
        state=build_state(gpu_uuids=REAL, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
        miner_hotkey="miner-hotkey",
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["kernel_view_would_ban"] is False


@pytest.mark.asyncio
async def test_gpu_ban_catches_spoofed_uuid_through_the_kernel_view(context_factory, enforcing):
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
async def test_unreadable_procfs_falls_back_to_reported_uuids_and_is_not_a_spoof(
    context_factory, enforcing
):
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids="GPU-honest", rented_data=rented, specs={"gpu": {"count": 1}}),
        ssh=_ssh_with_procfs([]),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_gpu_uuids"] is None
    assert result.event.what_we_saw["reported_uuids_differ_from_kernel"] is False
    # the failed attempt is remembered; specs never carry a kernel list the host did not give
    assert result.updates["state"].kernel_gpu_uuids_read is True
    assert result.updates["state"].kernel_gpu_uuids is None
    assert "kernel_gpu_uuids" not in result.updates["state"].specs


@pytest.mark.asyncio
async def test_gpu_ban_reuses_the_kernel_view_read_by_the_provider_check(
    context_factory, enforcing
):
    """One procfs read per cycle: the two checks run back to back and the second never touches ssh."""
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ssh = _ssh_with_procfs([f"{REAL}, 0"])
    ctx = context_factory(state=build_state(gpu_uuids=SPOOFED, rented_data=rented), ssh=ssh)

    provider = await BannedProviderCheck().run(ctx)
    assert provider.passed is True  # banned_guids is the GPU list, not the provider list
    gpu = await BannedGpuCheck().run(_after(ctx, provider))

    assert gpu.passed is False
    assert gpu.event.what_we_saw["kernel_gpu_uuids"] == [REAL]
    ssh.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_unreadable_procfs_is_read_once_per_cycle_not_once_per_check(context_factory, shadow):
    """A wedged host costs one bounded read: BannedGpuCheck takes the failed attempt from state."""
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ssh = _ssh_with_procfs([])
    ctx = context_factory(state=build_state(gpu_uuids="GPU-honest", rented_data=rented), ssh=ssh)

    provider = await BannedProviderCheck().run(ctx)
    gpu = await BannedGpuCheck().run(_after(ctx, provider))

    assert gpu.passed is True
    assert gpu.event.what_we_saw["kernel_gpu_uuids"] is None
    ssh.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_hung_procfs_read_is_bounded_and_falls_back_to_reported_uuids(
    context_factory, monkeypatch
):
    """A wedged host cannot hold the fatal check open: the read times out and counts as unreadable."""
    monkeypatch.setattr(nvidia_devices, "KERNEL_GPU_UUID_READ_TIMEOUT_SECONDS", 0.01)

    async def _never_returns(*_args, **_kwargs):
        await asyncio.sleep(60)

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=_never_returns)
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(state=build_state(gpu_uuids="GPU-honest", rented_data=rented), ssh=ssh)

    result = await asyncio.wait_for(BannedProviderCheck().run(ctx), timeout=5)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_gpu_uuids"] is None
    ssh.run.assert_awaited_once()
