import time

from core.config import settings

from ..pipeline import Context

# the executor task is cancelled at JOB_TIME_OUT - 120 (miner_service)
TASK_TIMEOUT_MARGIN_SECONDS = 120


def task_seconds_left(ctx: Context) -> float:
    """Seconds before the executor task is cancelled: by its own timeout, or by an express recheck's earlier deadline."""
    left = settings.JOB_TIME_OUT - TASK_TIMEOUT_MARGIN_SECONDS - (time.monotonic() - ctx.started_at_monotonic)
    if ctx.deadline_monotonic is None:
        return left
    return min(left, ctx.deadline_monotonic - time.monotonic())
