from time import time

from core.utils import get_logger, _m
from services.const import TOTAL_BURN_EMISSION
from services.task import JobResult


logger = get_logger(__name__)


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

        unrented_count_by_group = {
            base_model: sum(gpu_types.values())
            for base_model, gpu_types in unrented_count_by_type.items()
        } if unrented_count_by_type else {}

        logger.info(_m("Incentive_results", extra={
            "duration": f"{time() - started_at:.2f}s",
            "unrented_count_by_group": unrented_count_by_group,
            "rental_share": rental_share,
            "burn_share": burn_share,
            "total_rental_cost": total_rental_cost,
        }))
        for job_list in job_results.values():
            for job_result in job_list:
                logger.info(_m("", extra=job_result.model_dump(exclude={"incentive_logs"})))
    except Exception as e:
        logger.error(f"Error logging for monitoring: {e}", exc_info=True)
