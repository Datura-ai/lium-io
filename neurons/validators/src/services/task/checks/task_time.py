import time

from core.config import settings

from ..pipeline import Context

# the executor task is cancelled at JOB_TIME_OUT - 120 (miner_service)
TASK_TIMEOUT_MARGIN_SECONDS = 120


def task_seconds_left(ctx: Context) -> float:
    """Seconds before the executor task is cancelled."""
    return settings.JOB_TIME_OUT - TASK_TIMEOUT_MARGIN_SECONDS - (time.monotonic() - ctx.started_at_monotonic)
