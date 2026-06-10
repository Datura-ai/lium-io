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
    unrented_count_by_bucket: dict | None = None,
) -> None:
    try:
        first_with_rental = next(
            (r for results in job_results.values() for r in results if r.rental_share),
            None,
        )
        rental_share = first_with_rental.rental_share if first_with_rental else 0
        burn_share = first_with_rental.burn_share if first_with_rental else float(TOTAL_BURN_EMISSION)
        total_rental_cost = first_with_rental.total_rental_cost if first_with_rental else 0

        # Bucket-keyed map uses tuple keys; serialize as "{base}_{bucket}" for Loki unwrap compatibility.
        unrented_count_by_group = {
            f"{base}_{bucket}": count
            for (base, bucket), count in (unrented_count_by_bucket or {}).items()
        }

        # Aggregate per-bucket stats from eligible unrented executors for dashboard-friendly logging.
        unrented_by_bucket: dict[str, dict[str, float]] = {}
        for results in job_results.values():
            for r in results:
                if not r.eligible_for_rental_share:
                    continue
                base = BASE_GPU_MAP.get(r.gpu_model, r.gpu_model)
                bucket = r.count_bucket if r.count_bucket is not None else 0
                key = f"{base}_{bucket}"
                eff = r.effective_rate or 0
                agg = unrented_by_bucket.setdefault(key, {
                    "base_model": base,
                    "bucket": bucket,
                    "count": 0,
                    "hourly_rate": r.hourly_rate or 0,
                    "cap_multiplier": r.unrented_cap_multiplier or 0,
                    "sysbox_multiplier_sum": 0.0,
                    "discord_multiplier_sum": 0.0,
                    "cost_per_h": 0.0,
                })
                agg["count"] += r.gpu_count
                agg["cost_per_h"] += r.gpu_count * eff
                agg["sysbox_multiplier_sum"] += r.gpu_count * (r.sysbox_multiplier or 0)
                agg["discord_multiplier_sum"] += r.gpu_count * (r.discord_multiplier or 0)

        for agg in unrented_by_bucket.values():
            cnt = agg["count"] or 1
            agg["sysbox_multiplier_avg"] = agg.pop("sysbox_multiplier_sum") / cnt
            agg["discord_multiplier_avg"] = agg.pop("discord_multiplier_sum") / cnt
            agg["share_of_rental_pool"] = (
                agg["cost_per_h"] / total_rental_cost if total_rental_cost else 0.0
            )
            agg["share_of_emission"] = agg["share_of_rental_pool"] * rental_share

        logger.info(_m("Incentive_results", extra={
            "duration": f"{time() - started_at:.2f}s",
            "unrented_count_by_group": unrented_count_by_group,
            "unrented_by_bucket": unrented_by_bucket,
            "rental_share": rental_share,
            "burn_share": burn_share,
            "total_rental_cost": total_rental_cost,
        }))

        for key, agg in sorted(unrented_by_bucket.items()):
            logger.info(_m("Unrented_bucket_summary", extra={"bucket_key": key, **agg}))

        # Rental breakdown: one line per executor, sorted by gpu name then gpu count
        if any(r.eligible_for_rental_share for results in job_results.values() for r in results):
            logger.info(_m("Rental_breakdown | format: rate * cap * sysbox * discord = eff/gpu * gpus = cost"))
        rental_executors: list[tuple[str, JobResult]] = []
        for results in job_results.values():
            for r in results:
                if not r.eligible_for_rental_share:
                    continue
                base = BASE_GPU_MAP.get(r.gpu_model, r.gpu_model)
                rental_executors.append((base, r))

        for base, r in sorted(rental_executors, key=lambda x: (x[0], x[1].gpu_count)):
            key = f"{r.gpu_count}x{base}"
            bucket = r.count_bucket if r.count_bucket is not None else 0
            bucket_key = f"{base}·{bucket}"
            cap = r.unrented_cap_multiplier or 0
            rate = r.hourly_rate or 0
            eff = r.effective_rate or 0
            sysbox = r.sysbox_multiplier or 0
            discord = r.discord_multiplier or 0
            ex_cost = r.gpu_count * eff
            ex_id = r.executor_info.uuid[:8] if r.executor_info.uuid else "?"
            logger.info(_m(
                f"Rental_breakdown | {key} [{ex_id}] {bucket_key} | ${rate:.2f} * {cap:.2f} * {sysbox:.2f} * {discord:.2f} = ${eff:.3f}/gpu * {r.gpu_count}gpu = ${ex_cost:.2f}",
                extra={"group": key, "bucket_key": bucket_key, "executor_id": ex_id,
                        "hourly_rate": rate, "unrented_cap_multiplier": cap,
                        "sysbox_multiplier": sysbox, "discord_multiplier": discord, "effective_rate": eff,
                        "gpu_count": r.gpu_count, "executor_cost": ex_cost},
            ))

        for job_list in job_results.values():
            for job_result in job_list:
                logger.info(_m("", extra=job_result.model_dump(exclude={"incentive_logs"})))
    except Exception as e:
        logger.error(f"Error logging for monitoring: {e}", exc_info=True)
