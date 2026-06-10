"""Shared incentive eligibility checks."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:
    from services.task_service import JobResult


def is_missing_discord_after_cutoff(job_result: JobResult) -> bool:
    return (
        datetime.utcnow() >= settings.DISCORD_INCENTIVE_CUTOFF
        and not job_result.provider_discord_connected
    )
