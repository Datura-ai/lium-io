from __future__ import annotations

import logging

from services.executor_connectivity.dind_probe import DIND_SSHD_NOT_READY, HOST_SYSBOX_FS_BROKEN
from services.executor_connectivity.models import DIND_INNER_DOCKERD_CODES, DindLogCause

from ..availability import ReachSource, ReachTarget
from ..messages import MessageTemplate, render_message
from ..messages import SysboxRequiredMessages as Msg
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node
from .rented_gpu_drop import GPU_DROP_REDIS_ERRORS

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# DAH-6637: a node whose sysbox-fs read broken stays hidden until a probe reads it clean; a cycle without a read
# (probe skipped or deferred, read timed out) keeps the hold. Keyed like the GPU-loss hold (hotkey, executor UUID).
SYSBOX_FS_HOLD_KEY_PREFIX = "sysbox_fs_hold"
# shortcut: not a hold window, only how long Redis keeps the mark of a node no cycle sees any more; every held cycle
# renews it.
SYSBOX_FS_HOLD_TTL_SECONDS = 3 * 24 * 3600

# DAH-3634: probe causes that get their own reason code instead of SYSBOX_REQUIRED_MISSING, keyed
# by `ContextState.dind_probe_error.code`, written by dind_probe.diagnose_docker_run_error.
_NVIDIA_HOOK_TEMPLATE_BY_REASON_CODE: dict[str, MessageTemplate] = {
    Msg.NVIDIA_RUNTIME_MISMATCH.reason: Msg.NVIDIA_RUNTIME_MISMATCH,
    Msg.NVIDIA_CONTAINER_HOOK_FAILED.reason: Msg.NVIDIA_CONTAINER_HOOK_FAILED,
}


class SysboxRequiredCheck:
    """Ban unrented executors that lack the sysbox runtime from appearing on the network.

    DAH-2313: sysbox is required to be allowed on the network. A machine that has no sysbox
    AND is not currently rented is rejected here (fatal) so the pipeline halts before scoring
    and the executor is reported with a zero score. Already-rented no-sysbox machines are left
    untouched so live rentals are never disrupted.
    """

    check_id = "executor.validate.sysbox_required"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        # Determine rental status the same way as port_count.py / rented_machine.py.
        rented_executor = rental_of_this_node(ctx)
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        if not ctx.state.sysbox_runtime and not is_rented:
            # DAH-2742: the verification is deliberately kept. ConnectivityOrchestrator.verify
            # forces sysbox_runtime=False whenever the DinD probe fails at all, not only on a
            # genuinely missing runtime, so a transient miss must not flip the executor instantly.
            return CheckResult(passed=False, event=self._render_missing_sysbox_event(ctx, is_rented))

        cause = ctx.state.dind_probe_error
        broken_text = await self._sysbox_fs_hold(ctx, cause)
        if broken_text is not None:
            # DAH-6637: hidden from new rentals only, rented or not; never fatal, score and verification kept.
            what = {
                "reach_source": str(ReachSource.VALIDATOR),
                "reach_target": str(ReachTarget.NODE),
                "is_rented": is_rented,
                "dind_probe_error": broken_text,
            }
            event = render_message(Msg.HOST_SYSBOX_FS_BROKEN, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event, fatal=False)

        event = render_message(
            Msg.SYSBOX_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"sysbox_runtime": ctx.state.sysbox_runtime, "is_rented": is_rented},
        )
        return CheckResult(passed=True, event=event)

    @staticmethod
    async def _sysbox_fs_hold(ctx: Context, cause: DindLogCause | None) -> str | None:
        """The broken read's text while the node is on the sysbox-fs hold, else None. A broken read starts or renews
        the hold, a clean read ends it, a cycle with neither keeps what Redis has. A Redis that cannot be read is no
        hold; a dry run reads and writes nothing."""
        key = f"{SYSBOX_FS_HOLD_KEY_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"
        redis = ctx.services.redis
        try:
            if cause is not None and cause.code == HOST_SYSBOX_FS_BROKEN:
                if not settings.DRY_RUN:
                    await redis.set(key, cause.text, ex=SYSBOX_FS_HOLD_TTL_SECONDS)
                return cause.text
            if ctx.state.sysbox_fs_read_clean:
                if not settings.DRY_RUN:
                    await redis.delete(key)
                return None
            held = await redis.get(key)
        except GPU_DROP_REDIS_ERRORS:
            logger.warning(
                _m("SYSBOX_FS_HOLD_REDIS_UNAVAILABLE", extra=get_extra_info(ctx.default_extra)), exc_info=True
            )
            return cause.text if cause is not None and cause.code == HOST_SYSBOX_FS_BROKEN else None
        if held is None:
            return None
        if not settings.DRY_RUN:
            await redis.set(key, held, ex=SYSBOX_FS_HOLD_TTL_SECONDS)
        text = held.decode() if isinstance(held, bytes) else str(held)
        return f"held since an earlier cycle, no clean read yet: {text}"

    def _render_missing_sysbox_event(self, ctx: Context, is_rented: bool) -> ValidationEvent:
        """The failure event for an unrented node with no sysbox verdict: the plain SYSBOX_MISSING
        template, or the probe cause's own template and remediation when the probe never measured
        sysbox at all."""
        what: dict[str, bool | str] = {
            "sysbox_runtime": ctx.state.sysbox_runtime,
            "is_rented": is_rented,
        }
        remediation = None
        cause = ctx.state.dind_probe_error
        template = Msg.SYSBOX_MISSING
        nvidia_template = _NVIDIA_HOOK_TEMPLATE_BY_REASON_CODE.get(cause.code) if cause else None
        if nvidia_template is not None:
            # DAH-3634: the probe's `docker run` was refused by the NVIDIA container hook, so
            # no sysbox verdict was measured and the node cannot start any GPU container;
            # "install sysbox" cannot fix it. The check still fails: the score stays 0.
            template = nvidia_template
            what["dind_probe_error"] = cause.text
            remediation = f"The sysbox check could not run: {cause.text}. {template.remediation}"
        elif cause is not None:
            # DAH-2856: the probe's container came up but its sshd never answered, so no sysbox
            # verdict was measured at all; the cause read from the container's logs replaces
            # "install sysbox", which sent ticket-0309's provider through three reinstalls.
            what["dind_probe_error"] = cause.text
            remediation = f"The sysbox check could not run: {cause.text}."
            # "not sysbox" is said only when dockerd's own line was read: an inner-dockerd code
            # with no line names the symptom, not the cause, so the generic guidance stays.
            if cause.code in DIND_INNER_DOCKERD_CODES and cause.dockerd_line:
                remediation += " Fix that on the host first; reinstalling sysbox does not change it."
            elif cause.code == DIND_SSHD_NOT_READY.code:
                # DAH-5844: the container never answered at all, so the notice must not say sysbox is missing
                template = Msg.DIND_PROBE_UNREACHABLE
                remediation += f" {template.remediation}"
        return render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
            remediation=remediation,
        )
