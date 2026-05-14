"""Score calculation logic for executor validation.

This module contains the business logic for calculating actual and job scores
based on collateral status, rental state, and contract versions.
"""


from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG
from services.const import MACHINE_PRICES
from services.task.pipeline import Context

from core.config import settings

SCORE_PORTION_FOR_OLD_CONTRACT = 0


def calculate_scores(
    ctx: Context,
    rented: bool = False,
) -> tuple[float, float, str]:
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
    contract_version = ctx.contract_version or ""
    price_per_gpu = ctx.executor.price_per_gpu

    warning_messages = []
    job_score = 1.0
    actual_score = 1.0

    shared_config_client = getattr(ctx.services, "shared_config_client", None)
    shared_config = shared_config_client.config if shared_config_client else DEFAULT_SHARED_CONFIG

    # Machine price check
    base_price = MACHINE_PRICES.get(gpu_model, 0)
    if price_per_gpu and price_per_gpu > base_price * settings.MACHINE_MAX_PRICE_RATE:
        actual_score = 0.0
        warning_messages.append(
            f"GPU price exceeds the limit. limit: {base_price * settings.MACHINE_MAX_PRICE_RATE}, actual: {price_per_gpu}"
        )

    # EMA verifyx download speed check — threshold enforced upstream in VerifyXCheck
    ema_verifyx_download = ((ctx.state.specs or {}).get("network") or {}).get(
        "ema_verifyx_download_speed"
    )
    if not rented and ema_verifyx_download is None:
        actual_score = 0.0
        job_score = 0.0
        warning_messages.append(
            "EMA verifyx download speed unavailable (probe failed or never measured)"
        )

    if (
        shared_config.require_storage_limit_supported
        and not rented
        and (ctx.state.specs or {}).get("storage_limit_supported") is not True
    ):
        actual_score = 0.0
        job_score = 0.0
        warning_messages.append("Storage limit support required but unavailable")

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
) -> tuple[float, float, str]:
    """Format the return values for score calculation."""
    # Rented machines always report job_score=1.0
    final_job_score = 1.0 if rented else job_score

    warning_text = ""
    if warning_messages:
        warning_text = " WARNING: " + " | ".join(warning_messages)

    return actual_score, final_job_score, warning_text
