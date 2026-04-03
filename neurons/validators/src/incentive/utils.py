from time import time

from core.utils import get_logger, _m
from incentive.config import BASE_GPU_MAP, DefaultPrice
from services.const import TOTAL_BURN_EMISSION
from services.task import JobResult


logger = get_logger(__name__)


def get_hourly_rate(
    gpu_model: str,
    gpu_count: int,
    custom_prices: dict[str, dict[str, float | DefaultPrice]],
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

    if isinstance(value, DefaultPrice):
        return default_prices.get(gpu_model, 0.0) * value.multiplier

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

        # Rental breakdown: one line per executor, sorted by gpu name then gpu count
        if any(r.eligible_for_rental_share for results in job_results.values() for r in results):
            logger.info(_m("Rental_breakdown | format: rate * cap * sysbox = eff/gpu * gpus = cost"))
        rental_executors: list[tuple[str, JobResult]] = []
        for results in job_results.values():
            for r in results:
                if not r.eligible_for_rental_share:
                    continue
                base = BASE_GPU_MAP.get(r.gpu_model, r.gpu_model)
                rental_executors.append((base, r))

        for base, r in sorted(rental_executors, key=lambda x: (x[0], x[1].gpu_count)):
            key = f"{r.gpu_count}x{base}"
            cap = r.unrented_cap_multiplier or 0
            rate = r.hourly_rate or 0
            eff = r.effective_rate or 0
            sysbox = r.sysbox_multiplier or 0
            ex_cost = r.gpu_count * eff
            ex_id = r.executor_info.uuid[:8] if r.executor_info.uuid else "?"
            logger.info(_m(
                f"Rental_breakdown | {key} [{ex_id}] | ${rate:.2f} * {cap:.2f} * {sysbox:.2f} = ${eff:.3f}/gpu * {r.gpu_count}gpu = ${ex_cost:.2f}",
                extra={"group": key, "executor_id": ex_id, "hourly_rate": rate,
                        "unrented_cap_multiplier": cap, "sysbox_multiplier": sysbox,
                        "effective_rate": eff, "gpu_count": r.gpu_count, "executor_cost": ex_cost},
            ))

        for job_list in job_results.values():
            for job_result in job_list:
                logger.info(_m("", extra=job_result.model_dump(exclude={"incentive_logs"})))
    except Exception as e:
        logger.error(f"Error logging for monitoring: {e}", exc_info=True)
