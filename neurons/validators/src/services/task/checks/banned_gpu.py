from __future__ import annotations

from ..messages import BannedGpuMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .banned_provider import kernel_gpu_uuids, reported_gpu_uuids, uuids_to_match_bans_against


class BannedGpuCheck:
    """Block miners whose GPU UUIDs appear on the banlist maintained in Redis.

    Legacy validation refused to score temporarily ineligible GPUs (e.g., due to fraud or
    hardware defects). Keeping this explicit check lets policy updates propagate without
    editing the core pipeline.
    """

    check_id = "gpu.validate.banned"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        current_uuids = ctx.state.gpu_uuids or ""
        uuids = reported_gpu_uuids(ctx)
        # DAH-2662: also the kernel's view of the cards (BannedProviderCheck read it this cycle);
        # a shim that rewrites the reported UUIDs does not rewrite /proc/driver/nvidia. Shadow
        # until KERNEL_GPU_BAN_ENFORCEMENT_ENABLED: matched on the reported list, kernel view kept.
        # The kernel read comes before the empty-list return: a host that reports no UUIDs at all
        # is still matched on the kernel's list, so a kernel-only ban is not skipped.
        kernel_uuids = await kernel_gpu_uuids(ctx)

        # Get banned GUIDs from backend API response
        rented_data = ctx.state.rented_data
        banned_guids = rented_data.banned_guids if rented_data else []
        is_banned = any(
            guid in banned_guids for guid in uuids_to_match_bans_against(uuids, kernel_uuids)
        )
        kernel_view_would_ban = any(guid in banned_guids for guid in kernel_uuids or [])

        if not current_uuids and not is_banned:
            event = render_message(
                Msg.UUID_EMPTY,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "kernel_gpu_uuids": kernel_uuids,
                    "kernel_view_would_ban": kernel_view_would_ban,
                },
            )
            return CheckResult(passed=True, event=event)

        if is_banned:
            event = render_message(
                Msg.GPU_BANNED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "gpu_uuids": current_uuids,
                    "kernel_gpu_uuids": kernel_uuids,
                    "kernel_view_would_ban": kernel_view_would_ban,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = render_message(
            Msg.GPU_ALLOWED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "gpu_uuids": current_uuids,
                "kernel_gpu_uuids": kernel_uuids,
                "kernel_view_would_ban": kernel_view_would_ban,
            },
        )
        return CheckResult(passed=True, event=event)
