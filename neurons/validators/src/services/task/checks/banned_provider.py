from __future__ import annotations

from dataclasses import replace

from core.config import settings
from services.nvidia_devices import read_kernel_gpu_uuids

from ..messages import BannedProviderMessages as Msg, render_message
from ..pipeline import CheckResult, Context


def reported_gpu_uuids(ctx: Context) -> list[str]:
    return [gpu_uuid for gpu_uuid in (ctx.state.gpu_uuids or "").split(",") if gpu_uuid]


async def kernel_gpu_uuids(ctx: Context) -> list[str] | None:
    """Kernel-side GPU identity, read once per cycle: this check stores it in ctx.state for BannedGpuCheck.

    `kernel_gpu_uuids_read` marks the attempt, so an unreadable procfs is read once per cycle too —
    the second check must not open another 15 s SSH read on the same wedged host.
    """
    if ctx.state.kernel_gpu_uuids_read:
        return ctx.state.kernel_gpu_uuids
    if ctx.ssh is None:
        return None
    return await read_kernel_gpu_uuids(ctx.ssh)


def ban_match_uuids(reported: list[str], kernel: list[str] | None) -> list[str]:
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
        # DAH-2662: a banned operator re-registers the same cards under a fresh hotkey with the
        # reported UUIDs rewritten by an NVML shim. Match the ban against the kernel's view as well —
        # the shim does not author /proc/driver/nvidia — and record it on the executor's specs so the
        # backend keys future bans on an identity the host did not choose. Shadow first
        # (KERNEL_GPU_BAN_ENFORCEMENT_ENABLED=False): the kernel list is read and recorded, the ban
        # is still matched on the reported list, and `kernel_view_would_ban` says what the flip changes.
        kernel_uuids = await kernel_gpu_uuids(ctx)
        is_banned = bool(
            rented_data
            and rented_data.is_provider_banned(
                miner_hotkey=ctx.miner_hotkey,
                miner_coldkey=ctx.miner_coldkey,
                gpu_uuids=ban_match_uuids(gpu_uuids, kernel_uuids),
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
        }
        specs = ctx.state.specs
        if kernel_uuids is not None:
            specs = {**specs, "kernel_gpu_uuids": kernel_uuids}
        updates: dict[str, object] = {
            "state": replace(
                ctx.state, kernel_gpu_uuids=kernel_uuids, kernel_gpu_uuids_read=True, specs=specs
            )
        }

        if is_banned:
            event = render_message(Msg.PROVIDER_BANNED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(
                passed=False, event=event, updates={**updates, "is_provider_banned": True}
            )

        event = render_message(Msg.PROVIDER_ALLOWED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event, updates=updates)
