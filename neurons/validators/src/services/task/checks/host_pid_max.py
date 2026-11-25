"""DAH-7041: a node whose host answers an unusable kernel.pid_max is not listed.

The rental create reads the host's kernel.pid_max before anything else and refuses the rental when it is unusable
(docker_service `HOST_READ_PID_MAX_STEP`): without a safe pids limit a tenant fork bomb starves the host's sshd and
dockerd. No other check read it, so such a node passed every cycle, was listed, failed its renter's create, was paused
by the backend and listed again by the next clean cycle (24 Nov 2026, Portal node 0e25f768: "unparsable output").
This reads the same file with the same parser and threshold (services/host_pid_max.py).

Only an answer the host gave decides: a read that timed out or closed without an exit status says nothing about the
host's setting and passes; the create's own read still refuses that rental. Like NodeAccountCheck's refusal, a
failure is an availability error: the backend hides the node and refuses new rentals on its free GPUs, while a rental
already on it runs on (not fatal there, score unchanged).
"""

from __future__ import annotations

from services.host_pid_max import READ_HOST_PID_MAX_COMMAND, parse_host_pid_max, pid_max_refused

from core.config import settings

from ..availability import ReachSource, ReachTarget
from ..messages import HostPidMaxMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental

# the create's own bound on this read (docker_service _HOST_READ_TIMEOUT_SECONDS)
READ_TIMEOUT_SECONDS = 15


class HostPidMaxCheck:
    check_id = "executor.validate.host_pid_max"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if settings.RENTAL_PIDS_LIMIT_PER_CPU <= 0 or ctx.runner is None:
            # the create sets no pids limit then, so it reads nothing and refuses nothing
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id))
        result = await ctx.runner.run(READ_HOST_PID_MAX_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        what: dict = {"executor_uuid": ctx.executor.uuid}
        if result.error_type is not None:
            what["read_error"] = result.error_type
            return CheckResult(passed=True, event=render_message(Msg.NOT_READ, ctx=ctx, check_id=self.check_id, what=what))
        host_pid_max, why = parse_host_pid_max(result.exit_code, result.stdout)
        what["host_pid_max"] = host_pid_max
        if not pid_max_refused(host_pid_max):
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
        what.update(
            {
                "why": why or f"below {settings.RENTAL_PIDS_LIMIT_HOST_MARGIN} reserved for the host",
                "reach_source": str(ReachSource.VALIDATOR),
                "reach_target": str(ReachTarget.NODE),
            }
        )
        event = render_message(Msg.UNUSABLE, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=False, event=event, fatal=not has_live_rental(ctx))
