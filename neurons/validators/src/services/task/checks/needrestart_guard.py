from __future__ import annotations

from ..messages import NeedrestartGuardMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

GUARDED = ("present", "no_needrestart")


class NeedrestartGuardCheck:
    """Warn when the host's apt (needrestart) may restart the container runtime under running pods.

    Pure data: reads ``specs.needrestart_guard`` from the scrape, which looks for the
    /etc/needrestart/conf.d/lium-container-runtime.conf the setup script writes since 16 Nov 2026 (DAH-6384).
    Hosts set up earlier have needrestart and no guard. Non-fatal, before the rented short-circuit so rented
    hosts, where a runtime restart breaks live pods, are reported too. The score is not changed. The warning is
    read from the validator log and its providers are told directly, as with PidHostRequiredCheck's rented warning;
    it is not published to the backend.
    """

    check_id = "executor.validate.needrestart_guard"
    replays_after_session_drop = True
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        guard = specs.get("needrestart_guard")
        what = {"needrestart_guard": guard, "unguarded": specs.get("needrestart_guard_unguarded")}
        if not isinstance(guard, str):
            what = {"scrape_error": specs.get("needrestart_guard_scrape_error")}
            return CheckResult(passed=True, event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what))
        if guard not in GUARDED:
            return CheckResult(passed=True, event=render_message(Msg.MISSING, ctx=ctx, check_id=self.check_id, what=what))
        return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
