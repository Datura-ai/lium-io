from __future__ import annotations

from dataclasses import replace

from core.config import settings
from services.nvidia_devices import KernelGpuView, read_kernel_gpu_view

from ..messages import BannedProviderMessages as Msg, render_message
from ..pipeline import CheckResult, Context


def reported_gpu_uuids(ctx: Context) -> list[str]:
    return [gpu_uuid for gpu_uuid in (ctx.state.gpu_uuids or "").split(",") if gpu_uuid]


async def kernel_gpu_view(ctx: Context) -> KernelGpuView:
    """Kernel-side GPU identity, read once per cycle: this check stores it in ctx.state for BannedGpuCheck.

    `kernel_gpu_uuids_read_attempted` marks the attempt, so an unreadable procfs is read once per cycle too —
    the second check must not open another 15 s SSH read on the same wedged host.
    """
    if ctx.state.kernel_gpu_uuids_read_attempted:
        return KernelGpuView(ctx.state.kernel_gpu_uuids, ctx.state.kernel_gpu_foreign_mounts)
    if ctx.ssh is None:
        return KernelGpuView(uuids=None, foreign_mounts=[])
    return await read_kernel_gpu_view(ctx.ssh)


async def kernel_gpu_uuids(ctx: Context) -> list[str] | None:
    return (await kernel_gpu_view(ctx)).uuids


def uuids_to_match_bans_against(reported: list[str], kernel: list[str] | None) -> list[str]:
    """The UUID list a ban is matched against: reported + kernel once enforcement is on, reported only in shadow."""
    if not settings.KERNEL_GPU_BAN_ENFORCEMENT_ENABLED:
        return reported
    return list(dict.fromkeys(reported + (kernel or [])))


class BannedProviderCheck:
    check_id = "gpu.validate.banned_provider"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        gpu_uuids = reported_gpu_uuids(ctx)
        # DAH-2662: match bans against the kernel's GPU list too (flag and shapes: config.py, kernel_gpu_view)
        kernel_uuids, foreign_mounts = await kernel_gpu_view(ctx)
        is_banned = bool(
            rented_data
            and rented_data.is_provider_banned(
                miner_hotkey=ctx.miner_hotkey,
                miner_coldkey=ctx.miner_coldkey,
                gpu_uuids=uuids_to_match_bans_against(gpu_uuids, kernel_uuids),
            )
        )
        # what flipping enforcement changes: a kernel UUID on the ban list (hotkey/coldkey bans
        # already apply on both sides of the flag)
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
            "kernel_ban_enforced": settings.KERNEL_GPU_BAN_ENFORCEMENT_ENABLED,
            "kernel_gpu_foreign_mounts": foreign_mounts,
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
                specs=specs,
            )
        }

        if is_banned:
            event = render_message(Msg.PROVIDER_BANNED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(
                passed=False, event=event, updates={**updates, "is_provider_banned": True}
            )

        if foreign_mounts and settings.KERNEL_GPU_BAN_ENFORCEMENT_ENABLED:
            event = render_message(
                Msg.KERNEL_GPU_VIEW_OVERLAID, ctx=ctx, check_id=self.check_id, what=what
            )
            return CheckResult(passed=False, event=event, updates=updates)

        event = render_message(Msg.PROVIDER_ALLOWED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event, updates=updates)
