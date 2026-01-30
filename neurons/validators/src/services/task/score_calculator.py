"""Score calculation logic for executor validation.

This module contains the business logic for calculating actual and job scores
based on collateral status, rental state, and contract versions.
"""

from typing import Tuple

from core.config import settings
from services.const import MACHINE_PRICES
from services.task.pipeline import Context


SCORE_PORTION_FOR_OLD_CONTRACT = 0


def calculate_scores(
    ctx: Context,
    rented: bool = False,
) -> Tuple[float, float, str]:
    """Calculate actual and job scores for an executor validation.

    Args:
        ctx: pipeline context
        rented: Whether the executor is currently rented

    Returns:
        Tuple of (actual_score, job_score, warning_message)
        - actual_score: Score used for validator rewards
        - job_score: Score reported in logs (1.0 for rented machines)
        - warning_message: Empty string or warning message with leading " WARNING: "
    """
    gpu_model = ctx.state.gpu_model or ""
    collateral_deposited = ctx.collateral_deposited
    is_rental_succeed = ctx.is_rental_succeed
    contract_version = ctx.contract_version or ""
    price_per_gpu = ctx.executor.price_per_gpu

    warning_messages = []
    job_score = 1.0
    actual_score = 1.0

    if not is_rental_succeed and not settings.SKIP_RENTAL_VERIFICATION:
        actual_score = 0.0
        warning_messages.append("Score set to 0 pending rental verification")

    # Machine price check
    base_price = MACHINE_PRICES.get(gpu_model, 0)
    if price_per_gpu and price_per_gpu > base_price * settings.MACHINE_MAX_PRICE_RATE:
        actual_score = 0.0
        warning_messages.append(
            f"GPU price exceeds the limit. limit: {base_price * settings.MACHINE_MAX_PRICE_RATE}, actual: {price_per_gpu}"
        )

    # Early return for collateral-excluded GPU types
    if gpu_model in settings.COLLATERAL_EXCLUDED_GPU_TYPES:
        return _format_return(actual_score, job_score, warning_messages, rented)

    # Collateral checks
    if not collateral_deposited:
        collateral_error = ctx.collateral_error_message
        if settings.ENABLE_NO_COLLATERAL:
            warning_messages.append(collateral_error or "No collateral deposited")
        else:
            actual_score = 0.0
            job_score = 0.0
            warning_messages.append(collateral_error or "Collateral required but not deposited")
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
