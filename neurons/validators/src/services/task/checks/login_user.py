"""The validator's SSH login on a node must be root.

Every command the validator runs on a node writes its scripts into a fresh root-only directory under the executor
image's own /root/app and runs them with the image's interpreter (services.executor_payload). A non-root login cannot
do that, so it is refused here with its own reason, right after the account check, before anything is installed or
run; a node that is not validated is never listed or rented, so no later path (backup, restore) runs there either.
No executor image logs the validator in as anything but root.
"""

from __future__ import annotations

from core.utils import get_logger

from ..messages import LoginUserMessages as Msg, TenantEnforcementMessages, render_message
from ..pipeline import CheckResult, Context

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
        if not result.success or not uid.isdigit():
            # no confirmed uid (a timeout, an error, empty or malformed output): the node's first command failed, the
            # same transport failure as on any other first command; nothing is installed and no stored state is kept
            error = (result.error_type or f"exit {result.exit_code}") if not result.success else "not a uid"
            what = {"executor_uuid": ctx.executor.uuid, "transport_error": f"`{READ_UID_COMMAND}` failed ({error})"}
            event = render_message(TenantEnforcementMessages.EXECUTOR_TRANSPORT_UNREACHABLE, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event, fatal=True)
        if uid == "0":
            return CheckResult(passed=True, event=render_message(Msg.ROOT, ctx=ctx, check_id=self.check_id, what=what))
        # the run ends here and nothing is reset; a non-root login never passed validation (every later step needs root),
        # so there is no verified state to clear. It hides the node at once (REASONS_THAT_HIDE_AT_ONCE): a listed node
        # that switched its login must not take a new rental while it fails
        event = render_message(Msg.NOT_ROOT, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=False, event=event, fatal=True)
