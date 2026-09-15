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
):
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
async def test_gpu_ban_with_no_reported_uuids_is_still_matched_on_the_kernel_view(
    context_factory, enforcing
):
    """Regression: the check used to return GPU_UUID_EMPTY before the kernel read, so a host that
    reported no UUIDs at all slipped past a ban keyed on the kernel's list."""
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=None, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await BannedGpuCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == BannedGpuMessages.GPU_BANNED.reason
    assert result.event.what_we_saw["kernel_gpu_uuids"] == [REAL]


@pytest.mark.asyncio
async def test_no_reported_uuids_in_shadow_mode_stays_empty_but_records_the_kernel_view(
    context_factory, shadow
):
    """Control: with enforcement off the empty list still passes as GPU_UUID_EMPTY, and the event
    carries what the flip would change."""
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=None, rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"]),
    )

    result = await BannedGpuCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == BannedGpuMessages.UUID_EMPTY.reason
    assert result.event.what_we_saw["kernel_gpu_uuids"] == [REAL]
    assert result.event.what_we_saw["kernel_view_would_ban"] is True


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


def test_foreign_mount_parser_passes_the_container_runtime_layout_and_flags_the_overlay():
    """Regression: a "tmpfs under /proc/driver/nvidia" rule would flag every GPU container (the
    runtime's own tmpfs sits there); the rule is a non-procfs mount on the gpus subtree."""
    assert nvidia_devices.foreign_mounts_over_proc_nvidia_gpus("\n".join(CONTAINER_MOUNTS)) == []
    assert nvidia_devices.foreign_mounts_over_proc_nvidia_gpus(
        "\n".join([*CONTAINER_MOUNTS, OVERLAY_MOUNT])
    ) == ["/proc/driver/nvidia/gpus/0001:00:00.0 tmpfs"]


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


@pytest.mark.asyncio
async def test_overlay_in_shadow_passes_but_names_the_mount_and_hands_it_to_the_gpu_check(
    context_factory, shadow
):
    rented = RentedExecutorsResponse(executors={}, banned_guids=[REAL])
    ssh = _ssh_with_procfs([f"{SPOOFED}, 0"], mounts=[*CONTAINER_MOUNTS, OVERLAY_MOUNT])
    ctx = context_factory(state=build_state(gpu_uuids=SPOOFED, rented_data=rented), ssh=ssh)

    provider = await BannedProviderCheck().run(ctx)
    gpu = await BannedGpuCheck().run(_after(ctx, provider))

    assert provider.passed is True
    assert provider.event.reason_code == BannedProviderMessages.PROVIDER_ALLOWED.reason
    assert provider.event.what_we_saw["kernel_gpu_foreign_mounts"] == [
        "/proc/driver/nvidia/gpus/0001:00:00.0 tmpfs"
    ]
    assert gpu.passed is True
    assert gpu.event.what_we_saw["kernel_gpu_uuids"] is None
    ssh.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_information_file_not_on_procfs_is_withheld_even_with_no_mount_of_its_own(
    context_factory, enforcing
):
    """Regression (self-review, 15 Sep): the executor container is privileged, so an operator can
    drop the card's proc bind and write a plain `information` file into the runtime's tmpfs; the
    mount table then shows nothing on the gpus subtree. The file's own filesystem still says tmpfs."""
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids=SPOOFED, rented_data=rented),
        ssh=_ssh_with_procfs(
            [f"{SPOOFED}, 0"], mounts=CONTAINER_MOUNTS[:1], file_fs=[f"{INFO_FILE} tmpfs"]
        ),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == BannedProviderMessages.KERNEL_GPU_VIEW_OVERLAID.reason
    assert result.event.what_we_saw["kernel_gpu_foreign_mounts"] == [f"{INFO_FILE} tmpfs"]
    assert result.updates["state"].kernel_gpu_uuids is None


@pytest.mark.asyncio
async def test_uuid_rows_without_a_procfs_stat_row_are_unreadable_not_a_finding(
    context_factory, enforcing
):
    """No `stat` output (an image without GNU stat) is fail-open like every other unreadable procfs."""
    rented = RentedExecutorsResponse(executors={}, banned_provider_guids=[REAL])
    ctx = context_factory(
        state=build_state(gpu_uuids="GPU-honest", rented_data=rented),
        ssh=_ssh_with_procfs([f"{REAL}, 0"], file_fs=[]),
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["kernel_gpu_uuids"] is None
    assert result.event.what_we_saw["kernel_gpu_foreign_mounts"] == []


@pytest.mark.parametrize(
    ("mounts", "file_fs", "expected"),
    [
        pytest.param(
            CONTAINER_MOUNTS,
            [f"{INFO_FILE} proc", f"{INFO_FILE}|symbolic link"],
            [f"{INFO_FILE} symbolic link"],
            id="symlink-to-another-procfs-file",
        ),
        pytest.param(
            [
                CONTAINER_MOUNTS[0],
                "1620 1575 0:23 /sys/kernel/core_pattern /proc/driver/nvidia/gpus/0001:00:00.0/information "
                "rw,relatime - proc proc rw",
            ],
            [f"{INFO_FILE} proc", f"{INFO_FILE}|regular file"],
            [f"{INFO_FILE} proc:/sys/kernel/core_pattern"],
            id="proc-bind-of-another-procfs-file",
        ),
    ],
)
def test_a_procfs_file_that_is_not_the_cards_own_node_is_foreign(mounts, file_fs, expected):
    """Regression (self-review round 2): "on procfs" is not "the driver's node". A symlink or a
    proc bind pointing at a root-writable procfs file passes the fstype rules alone."""
    assert (
        nvidia_devices.foreign_mounts_over_proc_nvidia_gpus("\n".join(mounts), "\n".join(file_fs))
        == expected
    )


def test_a_stat_row_whose_path_is_not_a_cards_information_file_is_foreign():
    """Regression (self-review round 3): a directory name with a newline under gpus/ splits its own
    stat rows, so neither half starts with the gpus path; every row must be <gpus>/<bus id>/information."""
    rows = "/proc/driver/nvidia/gpus/0001:00:00.0/information proc\n/proc/driver/nvidia/gpus/s\n"
    rows += "ym/information proc\n/proc/driver/nvidia/gpus/s\nym/information|symbolic link"
    found = nvidia_devices.foreign_mounts_over_proc_nvidia_gpus("\n".join(CONTAINER_MOUNTS), rows)
    assert found[0] == "/proc/driver/nvidia/gpus/s unexpected path"
    assert "ym/information proc" not in found  # a split half is named, never counted as evidence
    assert len(found) == 4
