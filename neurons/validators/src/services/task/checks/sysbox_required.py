from __future__ import annotations

from core.config import settings
from services.executor_connectivity.models import DIND_INNER_DOCKERD_CODES, DindLogCause

from ..messages import MessageTemplate, render_message
from ..messages import SysboxRequiredMessages as Msg
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context

# DAH-3634: probe causes that get their own reason code instead of SYSBOX_REQUIRED_MISSING, keyed
# by `ContextState.dind_probe_error.code`, written by dind_probe.diagnose_docker_run_error.
_NVIDIA_HOOK_TEMPLATES = {
    Msg.NVIDIA_RUNTIME_MISMATCH.reason: Msg.NVIDIA_RUNTIME_MISMATCH,
    Msg.NVIDIA_CONTAINER_HOOK_FAILED.reason: Msg.NVIDIA_CONTAINER_HOOK_FAILED,
}


def _nvidia_hook_template(cause: DindLogCause | None) -> MessageTemplate | None:
    """The NVIDIA_* template for a probe cause whose code is one of ours; None otherwise."""
    if cause is None:
        return None
    return _NVIDIA_HOOK_TEMPLATES.get(cause.code)


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
        if not settings.REQUIRE_SYSBOX_FOR_UNRENTED:
            event = render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        # Determine rental status the same way as port_count.py / rented_machine.py.
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        if not ctx.state.sysbox_runtime and not is_rented:
            # DAH-2742: the verification is deliberately kept. ConnectivityOrchestrator.verify
            # forces sysbox_runtime=False whenever the DinD probe fails at all, not only on a
            # genuinely missing runtime, so a transient miss must not flip the executor instantly.
            return CheckResult(passed=False, event=self._render_missing_sysbox_event(ctx, is_rented))

        event = render_message(
            Msg.SYSBOX_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"sysbox_runtime": ctx.state.sysbox_runtime, "is_rented": is_rented},
        )
        return CheckResult(passed=True, event=event)

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
        nvidia_template = _nvidia_hook_template(cause)
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
        return render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
            remediation=remediation,
        )
