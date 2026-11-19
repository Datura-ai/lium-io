"""The validator's SSH login on a node must be root.

Every command the validator runs on a node writes its scripts into a fresh root-only directory under the executor
image's own /root/app and runs them with the image's interpreter (services.executor_payload). A non-root login cannot
do that, so it is refused here with its own reason, right after the account check, before anything is installed or
run; a node that is not validated is never listed or rented, so no later path (backup, restore) runs there either.
No executor image logs the validator in as anything but root.
"""

from __future__ import annotations

from core.utils import _m, get_extra_info, get_logger

from ..messages import LoginUserMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental

logger = get_logger(__name__)

READ_UID_COMMAND = "id -u"
READ_TIMEOUT_SECONDS = 15


class LoginUserCheck:
    check_id = "executor.validate.login_user"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        result = await ctx.runner.run(READ_UID_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        uid = (result.stdout or "").strip()
        what = {"login_uid": uid, "ssh_username": ctx.executor.ssh_username}
        if not result.success:
            # an unreadable uid proves nothing; the SSH failure surfaces in the next check that runs a command
            what["error"] = result.error_type or f"exit {result.exit_code}"
            logger.info(_m("LOGIN_USER_UNREAD", extra=get_extra_info(what)))
            event = render_message(Msg.UNREAD, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=True, event=event)
        if uid == "0":
            return CheckResult(passed=True, event=render_message(Msg.ROOT, ctx=ctx, check_id=self.check_id, what=what))
        event = render_message(Msg.NOT_ROOT, ctx=ctx, check_id=self.check_id, what=what)
        if has_live_rental(ctx):
            # clearing the verified job resets the node inactive, which would end the renter's pod
            return CheckResult(passed=False, event=event, fatal=False)
        return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True}, fatal=True)
