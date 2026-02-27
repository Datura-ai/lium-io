from time import time

from core.utils import get_logger, _m
from incentive.config import BASE_GPU_MAP, DEFAULT_PRICE
from services.const import TOTAL_BURN_EMISSION
from services.task import JobResult


logger = get_logger(__name__)


def get_hourly_rate(
    gpu_model: str,
    gpu_count: int,
    custom_prices: dict[str, dict[str, float | str]],
    default_prices: dict[str, float],
) -> float:
    """Resolve hourly rate in USD for a (gpu_model, gpu_count) pair.

    Lookup in custom_prices: specific GPU name > "*" fallback.
    Within a GPU config: specific count > "*" fallback.
    If resolved value is DEFAULT_PRICE sentinel, falls back to default_prices[gpu_model].

    Returns 0.0 if no matching config found (not eligible for rental incentive).
    """
    if gpu_model in custom_prices:
        gpu_config = custom_prices[gpu_model]
    elif "*" in custom_prices:
        gpu_config = custom_prices["*"]
    else:
        return 0.0

    count_key = str(gpu_count)
    if count_key in gpu_config:
        value = gpu_config[count_key]
    elif "*" in gpu_config:
        value = gpu_config["*"]
    else:
        return 0.0

    if value == DEFAULT_PRICE:
        return default_prices.get(gpu_model, 0.0)

    return float(value)


def log_for_monitoring(
    job_results: dict[str, list[JobResult]],
    started_at: float,
    unrented_count_by_type: dict | None = None,
) -> None:
    try:
        first_with_rental = next(
            (r for results in job_results.values() for r in results if r.rental_share),
            None,
        )
        rental_share = first_with_rental.rental_share if first_with_rental else 0
        burn_share = first_with_rental.burn_share if first_with_rental else float(TOTAL_BURN_EMISSION)
        total_rental_cost = first_with_rental.total_rental_cost if first_with_rental else 0

        unrented_count_by_group = unrented_count_by_type or {}

        logger.info(_m("Incentive_results", extra={
            "duration": f"{time() - started_at:.2f}s",
            "unrented_count_by_group": unrented_count_by_group,
            "rental_share": rental_share,
            "burn_share": burn_share,
            "total_rental_cost": total_rental_cost,
        }))

        # Rental breakdown by (base_model, gpu_count)
        rental_breakdown: dict[str, dict] = {}
        for results in job_results.values():
            for r in results:
                if not r.eligible_for_rental_share:
                    continue
                base = BASE_GPU_MAP.get(r.gpu_model, r.gpu_model)
                key = f"{r.gpu_count}x{base}"
                if key not in rental_breakdown:
                    rental_breakdown[key] = {
                        "unrented_cap_multiplier": r.unrented_cap_multiplier,
                        "hourly_rate": r.hourly_rate,
                        "effective_rate": r.effective_rate,
                        "executor_count": 0,
                        "total_gpus": 0,
                        "total_cost": 0.0,
                    }
                rental_breakdown[key]["executor_count"] += 1
                rental_breakdown[key]["total_gpus"] += r.gpu_count
                rental_breakdown[key]["total_cost"] += r.gpu_count * (r.effective_rate or 0)

        for key, info in sorted(rental_breakdown.items(), key=lambda x: -x[1]["total_cost"]):
            cap = info["unrented_cap_multiplier"]
            rate = info["hourly_rate"]
            eff = info["effective_rate"]
            exs = info["executor_count"]
            cost = info["total_cost"]
            logger.info(_m(
                f"Rental_breakdown | {key} - {exs}ex | ${rate:.2f} * {cap:.2f} = ${eff:.3f}/gpu | total=${cost:.2f}",
                extra={"group": key, **info},
            ))

        for job_list in job_results.values():
            for job_result in job_list:
                logger.info(_m("", extra=job_result.model_dump(exclude={"incentive_logs"})))
    except Exception as e:
        logger.error(f"Error logging for monitoring: {e}", exc_info=True)
