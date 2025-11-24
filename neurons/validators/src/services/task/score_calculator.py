"""Score calculation logic for executor validation.

This module contains the business logic for calculating actual and job scores
based on collateral status, port availability, rental state, and contract versions.
"""

from typing import Tuple

from core.config import settings
from services.const import MIN_PORT_COUNT


SCORE_PORTION_FOR_OLD_CONTRACT = 0


def calculate_scores(
    gpu_model: str,
    collateral_deposited: bool,
    is_rental_succeed: bool,
    contract_version: str,
    rented: bool = False,
    port_count: int = 0,
) -> Tuple[float, float, str]:
    """Calculate actual and job scores for an executor validation.

    Args:
        gpu_model: GPU model identifier
        collateral_deposited: Whether collateral has been deposited
        is_rental_succeed: Whether rental verification succeeded
        contract_version: Version of the collateral contract
        rented: Whether the executor is currently rented
        port_count: Number of available ports

    Returns:
        Tuple of (actual_score, job_score, warning_message)
        - actual_score: Score used for validator rewards
        - job_score: Score reported in logs (1.0 for rented machines)
        - warning_message: Empty string or warning message with leading " WARNING: "
    """
    warning_messages = []
    job_score = 1.0
    actual_score = 1.0

    if not is_rental_succeed and not settings.SKIP_RENTAL_VERIFICATION:
        actual_score = 0.0
        warning_messages.append("Score set to 0 pending rental verification")

    if port_count < MIN_PORT_COUNT and not rented:
        actual_score = 0.0
        job_score = 0.0
        warning_messages.append(
            f"Insufficient ports: {port_count} available, {MIN_PORT_COUNT} required"
        )

    # Early return for collateral-excluded GPU types
    if gpu_model in settings.COLLATERAL_EXCLUDED_GPU_TYPES:
        return _format_return(actual_score, job_score, warning_messages, rented)

    # Collateral checks
    if not collateral_deposited:
        if settings.ENABLE_NO_COLLATERAL:
            warning_messages.append("No collateral deposited")
        else:
            actual_score = 0.0
            job_score = 0.0
            warning_messages.append("Collateral required but not deposited")
    elif (
        contract_version
        and contract_version != settings.get_latest_contract_version()
        and not settings.ENABLE_NO_COLLATERAL
    ):
        actual_score = actual_score * SCORE_PORTION_FOR_OLD_CONTRACT
        job_score = job_score * SCORE_PORTION_FOR_OLD_CONTRACT
        warning_messages.append(
            f"Outdated contract version (current: {contract_version}, "
            f"latest: {settings.get_latest_contract_version()})"
        )

    return _format_return(actual_score, job_score, warning_messages, rented)


def _format_return(
    actual_score: float,
    job_score: float,
    warning_messages: list[str],
    rented: bool,
) -> Tuple[float, float, str]:
    """Format the return values for score calculation."""
    # Rented machines always report job_score=1.0
    final_job_score = 1.0 if rented else job_score

    warning_text = ""
    if warning_messages:
        warning_text = " WARNING: " + " | ".join(warning_messages)

    return actual_score, final_job_score, warning_text
