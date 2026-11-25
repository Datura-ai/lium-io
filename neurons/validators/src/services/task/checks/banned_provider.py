from __future__ import annotations

import asyncio
from dataclasses import replace

from services.nvidia_devices import KernelGpuView, read_kernel_gpu_view

from ..messages import BannedProviderMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node


def reported_gpu_uuids(ctx: Context) -> list[str]:
    return [gpu_uuid for gpu_uuid in (ctx.state.gpu_uuids or "").split(",") if gpu_uuid]


KERNEL_VIEW_SOURCE_CONTAINER = "container"
KERNEL_VIEW_SOURCE_HOST = "host_pid1_root"


async def kernel_gpu_view(ctx: Context) -> KernelGpuView:
    """Kernel-side GPU identity, read once per cycle: this check stores it in ctx.state for BannedGpuCheck.

    `kernel_gpu_uuids_read_attempted` marks the attempt, so an unreadable procfs is read once per cycle too —
    the second check must not open another 15 s SSH read on the same wedged host.
    """
    return (await kernel_gpu_view_with_source(ctx))[0]


async def kernel_gpu_view_with_source(ctx: Context) -> tuple[KernelGpuView, str, list[str]]:
    """The kernel list, the view it came from, and the container's foreign mounts when the host list decided.

    The list is the host's, read through PID 1's root, when that read is the host's own procfs
    (read_kernel_gpu_view's host rules); otherwise the executor container's, as before. Both are read at once; when
    the host list decides, what the container view showed (a stale bind after a reset) is still reported as evidence.

    DAH-7076: the container's binds under /proc/driver/nvidia/gpus are made when the executor starts. A GPU reset
    re-creates the card's host entry, the bind keeps pointing at the deleted one, and the container's list reads
    as overlaid until the executor restarts. Where the PID namespace is not shared, PID 1's mount table is the
    container's, the host rules refuse it, and the container read runs exactly as before.
    """
    state = ctx.state
    if state.kernel_gpu_uuids_read_attempted:
        view = KernelGpuView(state.kernel_gpu_uuids, state.kernel_gpu_foreign_mounts)
        return view, state.kernel_gpu_view_source, state.container_kernel_gpu_foreign_mounts
    if ctx.ssh is None:
        return KernelGpuView(uuids=None, foreign_mounts=[]), KERNEL_VIEW_SOURCE_CONTAINER, []
    # neither read raises: a failed read is an unusable (None) list
    host, container = await asyncio.gather(
        read_kernel_gpu_view(ctx.ssh, host_view=True), read_kernel_gpu_view(ctx.ssh)
    )
    if host.uuids:
        return host, KERNEL_VIEW_SOURCE_HOST, container.foreign_mounts
    return container, KERNEL_VIEW_SOURCE_CONTAINER, []


async def kernel_gpu_uuids(ctx: Context) -> list[str] | None:
    return (await kernel_gpu_view(ctx)).uuids


def uuids_to_match_bans_against(reported: list[str], kernel: list[str] | None) -> list[str]:
    """The UUID list a ban is matched against: reported + kernel."""
    return list(dict.fromkeys(reported + (kernel or [])))


def has_live_rental(ctx: Context) -> bool:
    """Whether a renter's pod is running on this executor right now.

    A ban stops new rentals; it is not a reason to stop verifying a node whose pod is already
    in use. A node that stops being verified goes stale, is flipped inactive by the hourly
    sweep and loses renter billing and its payouts, which is not what a ban should do.
    """
    rental = rental_of_this_node(ctx)
    return bool(rental and rental.pods)


class BannedProviderCheck:
    check_id = "gpu.validate.banned_provider"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        gpu_uuids = reported_gpu_uuids(ctx)
        # DAH-2662: match bans against the kernel's GPU list too (shapes: kernel_gpu_view)
        (kernel_uuids, foreign_mounts), source, container_mounts = await kernel_gpu_view_with_source(ctx)
        is_banned = bool(
            rented_data
            and rented_data.is_provider_banned(
                miner_hotkey=ctx.miner_hotkey,
                miner_coldkey=ctx.miner_coldkey,
                gpu_uuids=uuids_to_match_bans_against(gpu_uuids, kernel_uuids),
                executor_uuid=ctx.executor.uuid,
            )
        )
        # a kernel UUID on the GPU ban list (hotkey/coldkey bans match without the kernel view)
        kernel_view_would_ban = bool(
            rented_data
            and any(
                gpu_uuid in rented_data.banned_provider_guids for gpu_uuid in kernel_uuids or []
            )
        )
        what = {
            "miner_hotkey": ctx.miner_hotkey,
            "miner_coldkey": ctx.miner_coldkey,
            "gpu_uuids": gpu_uuids,
            "kernel_gpu_uuids": kernel_uuids,
            "reported_uuids_differ_from_kernel": bool(kernel_uuids)
            and set(gpu_uuids) != set(kernel_uuids),
            "kernel_view_would_ban": kernel_view_would_ban,
            "kernel_ban_enforced": True,
            "kernel_gpu_foreign_mounts": foreign_mounts,
            "kernel_overlay_enforced": True,
            "kernel_view_source": source,
            "container_kernel_gpu_foreign_mounts": container_mounts,
        }
        specs = ctx.state.specs
        if kernel_uuids is not None:
            specs = {**specs, "kernel_gpu_uuids": kernel_uuids}
        updates: dict[str, object] = {
            "state": replace(
                ctx.state,
                kernel_gpu_uuids=kernel_uuids,
                kernel_gpu_uuids_read_attempted=True,
                kernel_gpu_foreign_mounts=foreign_mounts,
                kernel_gpu_view_source=source,
                container_kernel_gpu_foreign_mounts=container_mounts,
                specs=specs,
            )
        }

        if is_banned:
            event = render_message(Msg.PROVIDER_BANNED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(
                passed=False,
                event=event,
                updates={**updates, "is_provider_banned": True},
                fatal=not has_live_rental(ctx),
            )

        if foreign_mounts:
            event = render_message(
                Msg.KERNEL_GPU_VIEW_OVERLAID, ctx=ctx, check_id=self.check_id, what=what
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={**updates, "kernel_gpu_view_passed": False},
                fatal=not has_live_rental(ctx),
            )

        event = render_message(Msg.PROVIDER_ALLOWED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event, updates=updates)
