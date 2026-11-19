"""DAH-6466: stage the node's local s3fs plugin copy outside any rental, so a Volume rental reads only local files.

Non-fatal and never scored. A completion step: it runs only once the run's verdict is settled, and only when the
task has time left for its whole bound, so it can never take the time a scored check needs. A node the check has not reached yet (or a rented one) is not held off Volume
rentals: its first Volume rental stages the copy itself, the one registry contact for that node.
"""

from __future__ import annotations

import logging
import time

from core.utils import _m, get_extra_info
from services.s3fs_plugin_copy import S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS, s3fs_plugin_command

from ..messages import S3fsPluginCopyMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_egress_guard import BUDGET_MARGIN_SECONDS
from .rental_owner import rental_of_this_node
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

# a staged copy is looked at again this often (someone may have wiped the Docker data root); a failed
# staging, typically a registry path that is down, is tried again sooner
RECHECK_STAGED_SECONDS = 6 * 60 * 60
RETRY_FAILED_SECONDS = 60 * 60


class S3fsPluginCopyCheck:
    check_id = "executor.prep.s3fs_plugin_copy"
    fatal = False
    on_completion = True

    def __init__(self) -> None:
        # (miner hotkey, executor uuid) -> (monotonic time of the last run, whether it staged)
        self._last_run: dict[tuple[str, str], tuple[float, bool]] = {}

    async def run(self, ctx: Context) -> CheckResult:
        key = (ctx.miner_hotkey, ctx.executor.uuid)
        now = time.monotonic()
        last = self._last_run.get(key)
        if last is not None and now - last[0] < (RECHECK_STAGED_SECONDS if last[1] else RETRY_FAILED_SECONDS):
            return self._result(ctx, Msg.SKIPPED, {"why": "not due"})
        rented = rental_of_this_node(ctx)
        if rented is not None and rented.pods:
            # no plugin install beside a renter's pod; the copy is staged once the node is idle
            return self._result(ctx, Msg.SKIPPED, {"why": "rented"})
        if task_seconds_left(ctx) < S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS + BUDGET_MARGIN_SECONDS:
            # not a run: the next cycle with time for the whole bound tries again
            return self._result(ctx, Msg.SKIPPED, {"why": "not enough task time left"})

        result = await ctx.runner.run(
            s3fs_plugin_command(), timeout=S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS, retryable=False
        )
        staged = result.error_type is None and result.exit_code == 0
        self._last_run[key] = (now, staged)
        if staged:
            return self._result(ctx, Msg.STAGED, {})
        detail = result.error_type or (result.stderr or "")[-300:]
        logger.warning(_m("s3fs plugin copy not staged", extra=get_extra_info({**ctx.default_extra, "detail": detail})))
        return self._result(ctx, Msg.NOT_STAGED, {"detail": detail})

    def _result(self, ctx: Context, template, what: dict) -> CheckResult:
        return CheckResult(passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what))
