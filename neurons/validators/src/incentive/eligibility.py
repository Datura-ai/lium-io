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


def earns_spot_rental_incentive(job_result: JobResult) -> bool:
    """A rented spot node that earns rental incentive (ENABLE_SPOT_RENTAL_INCENTIVE).

    The same nodes as spot-node pay, while rented: the backend's provider_spot_executor_ids, so a
    node that is spot as a restriction (banned, pinned, or under a self-rent or untrusted-renter
    rental) keeps earning nothing. The free remainder of a partially rented one is not rented.
    """
    return (
        settings.ENABLE_SPOT_RENTAL_INCENTIVE
        and job_result.is_spot
        and job_result.is_on_spot_pay_list
        and job_result.is_rented
        and not job_result.is_split_remainder
    )
