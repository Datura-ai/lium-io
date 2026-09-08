from __future__ import annotations

from dataclasses import replace

from services.nvidia_devices import read_kernel_gpu_uuids

from ..messages import BannedProviderMessages as Msg, render_message
from ..pipeline import CheckResult, Context


def reported_gpu_uuids(ctx: Context) -> list[str]:
    return [gpu_uuid for gpu_uuid in (ctx.state.gpu_uuids or "").split(",") if gpu_uuid]


async def kernel_gpu_uuids(ctx: Context) -> list[str] | None:
    """Kernel-side GPU identity, read once per cycle: this check stores it in ctx.state for BannedGpuCheck."""
    if ctx.state.kernel_gpu_uuids is not None:
        return ctx.state.kernel_gpu_uuids
    if ctx.ssh is None:
        return None
    return await read_kernel_gpu_uuids(ctx.ssh)


class BannedProviderCheck:
    check_id = "gpu.validate.banned_provider"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        gpu_uuids = reported_gpu_uuids(ctx)
        # DAH-2662: a banned operator re-registers the same cards under a fresh hotkey with the
        # reported UUIDs rewritten by an NVML shim. Match the ban against the kernel's view as well —
        # the shim does not author /proc/driver/nvidia — and record it on the executor's specs so the
        # backend keys future bans on an identity the host did not choose.
        kernel_uuids = await kernel_gpu_uuids(ctx)
        ban_uuids = list(dict.fromkeys(gpu_uuids + (kernel_uuids or [])))
        is_banned = bool(
            rented_data
            and rented_data.is_provider_banned(
                miner_hotkey=ctx.miner_hotkey,
                miner_coldkey=ctx.miner_coldkey,
                gpu_uuids=ban_uuids,
            )
        )
        what = {
            "miner_hotkey": ctx.miner_hotkey,
            "miner_coldkey": ctx.miner_coldkey,
            "gpu_uuids": gpu_uuids,
            "kernel_gpu_uuids": kernel_uuids,
            "reported_uuids_differ_from_kernel": bool(kernel_uuids)
            and set(gpu_uuids) != set(kernel_uuids),
        }
        updates: dict = {}
        if kernel_uuids is not None:
            updates["state"] = replace(
                ctx.state,
                kernel_gpu_uuids=kernel_uuids,
                specs={**ctx.state.specs, "kernel_gpu_uuids": kernel_uuids},
            )

        if is_banned:
            event = render_message(Msg.PROVIDER_BANNED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(
                passed=False, event=event, updates={**updates, "is_provider_banned": True}
            )

        event = render_message(Msg.PROVIDER_ALLOWED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event, updates=updates)
