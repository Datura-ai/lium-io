"""DAH-3601 (P157/P164): the B300 1× bucket pays for 4 idle cards, the 8× bucket for 32.

Regression: on 17 Sep 2026 the 13:28Z cycle held 12 idle single-card B300 nodes
against a bucket cap of 10 (multiplier 10/12) while every 8-card node was rented.
This test replays that cycle shape against the production `IncentiveConfig`
defaults: with the cap at 4 the twelve 1× nodes share 4 cards of pay (4/12 each),
and an idle 8× node in the same cycle is not diluted. The second case, six idle
1× nodes, is paid in full under the old cap of 10 and at 4/6 under the new one, so
a cap back at 10, or at the drafts' 7 or 5, fails both multiplier assertions.
"""

from unittest.mock import AsyncMock

import pytest
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

B300 = "NVIDIA B300 SXM6 AC"


async def _run_with_production_config(job_results: dict[str, list[JobResult]]) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)

    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, job_results,
        total_gpu_model_count_map={B300: sum(j.gpu_count for jobs in job_results.values() for j in jobs)},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()
    return incentive


@pytest.mark.asyncio
async def test_six_idle_1x_b300_are_paid_four_sixths(make_pcc_job) -> None:
    """Six idle cards sit under the old cap of 10 (paid in full) and over the new
    cap of 4: the multiplier is 4/6. With the cap back at 10, or at the drafts'
    7 or 5, this test fails."""
    jobs = {f"miner_{i}": [make_pcc_job(f"exec-1x-{i:02d}", B300, 1)] for i in range(6)}

    incentive = await _run_with_production_config(jobs)

    assert incentive.cap_multiplier_by_bucket[("B300", 1)] == pytest.approx(4 / 6)
    for jobs_of_miner in jobs.values():
        assert jobs_of_miner[0].cap_dilution_applied is True
        assert jobs_of_miner[0].max_cap == 4
