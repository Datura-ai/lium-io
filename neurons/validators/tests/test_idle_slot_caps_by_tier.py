"""DAH-3620 (P163): the A100 8× bucket pays for 40 idle GPUs, the L40S 8× bucket for 16.

Regression: with both buckets at 64 the validator would pay six idle 8-card A100 nodes
(48 GPUs, the most renters ever held at once over 3–17 Sep 2026; p95 was 32) and three
idle 8-card L40S nodes (24 GPUs; renters never held more than 8) in full. These tests
replay the production `IncentiveConfig` defaults against those fills: the A100 bucket
dilutes to 40/48 and the L40S bucket to 16/24, while today's A100 fill (one idle
8-card node, the 17 Sep 14:29Z cycle) and a one-node L40S fill are still paid in full
and the 1× buckets and the other families are untouched. With either cap back at 64 the dilution assertions fail.
"""

from unittest.mock import AsyncMock

import pytest
from incentive.config import MAX_UNRENTED_GPUS_BY_TYPE, IncentiveConfig
from incentive.rental_price import RentalPriceIncentive

from tests.test_rental_price_incentive_flow import _make_pcc_job

A100 = "NVIDIA A100-SXM4-80GB"
L40S = "NVIDIA L40S"


async def _run_with_production_config(job_results: dict) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)

    counts: dict[str, int] = {}
    for jobs in job_results.values():
        for j in jobs:
            counts[j.gpu_model] = counts.get(j.gpu_model, 0) + j.gpu_count
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, job_results, total_gpu_model_count_map=counts
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()
    return incentive


def test_caps_are_the_measured_demand_and_nothing_else_moved():
    assert MAX_UNRENTED_GPUS_BY_TYPE["A100"] == {1: 10, 8: 40}
    assert MAX_UNRENTED_GPUS_BY_TYPE["L40S"] == {1: 10, 8: 16}
    for family in (
        "H100",
        "H200",
        "B200",
        "RTX 4090",
        "RTX 5090",
        "RTX PRO 6000",
        "RTX 6000 Ada Generation",
    ):
        assert MAX_UNRENTED_GPUS_BY_TYPE[family] == {1: 10, 8: 64}, family


@pytest.mark.asyncio
async def test_six_idle_8x_a100_nodes_share_forty_gpus_of_pay():
    """48 idle A100 GPUs on 8-card nodes (the 14 d peak demand) against a cap of 40."""
    jobs = {f"miner_{i}": [_make_pcc_job(f"a100-8x-{i}", A100, 8)] for i in range(6)}
    jobs["miner_1x"] = [_make_pcc_job("a100-1x", A100, 1)]

    incentive = await _run_with_production_config(jobs)

    assert incentive.unrented_count_by_bucket[("A100", 8)] == 48
    assert incentive.cap_multiplier_by_bucket[("A100", 8)] == pytest.approx(40 / 48)
    for i in range(6):
        result = jobs[f"miner_{i}"][0]
        assert result.count_bucket == 8
        assert result.max_cap == 40
        assert result.cap_dilution_applied is True
        assert result.effective_rate == pytest.approx(result.hourly_rate * 40 / 48)

    single = jobs["miner_1x"][0]
    assert single.count_bucket == 1
    assert single.max_cap == 10
    assert single.cap_dilution_applied is False
    assert single.unrented_cap_multiplier == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_three_idle_8x_l40s_nodes_share_sixteen_gpus_of_pay():
    """24 idle L40S GPUs on 8-card nodes against a cap of 16: two nodes' worth of pay."""
    jobs = {f"miner_{i}": [_make_pcc_job(f"l40s-8x-{i}", L40S, 8)] for i in range(3)}

    incentive = await _run_with_production_config(jobs)

    assert incentive.unrented_count_by_bucket[("L40S", 8)] == 24
    assert incentive.cap_multiplier_by_bucket[("L40S", 8)] == pytest.approx(16 / 24)
    for jobs_of_miner in jobs.values():
        assert jobs_of_miner[0].max_cap == 16
        assert jobs_of_miner[0].cap_dilution_applied is True


@pytest.mark.asyncio
async def test_todays_fill_is_paid_in_full():
    """Today's A100 fill (one idle 8-card node in the 17 Sep 2026 14:29Z cycle; no L40S
    8-card node was idle) and a one-node L40S fill: both sit under the new caps, so
    nothing changes for them at deploy."""
    jobs = {
        "miner_a100": [_make_pcc_job("a100-8x-0", A100, 8)],
        "miner_l40s": [_make_pcc_job("l40s-8x-0", L40S, 8)],
    }

    incentive = await _run_with_production_config(jobs)

    for key in (("A100", 8), ("L40S", 8)):
        assert incentive.unrented_count_by_bucket[key] == 8
        assert incentive.cap_multiplier_by_bucket[key] == pytest.approx(1.0)
    for jobs_of_miner in jobs.values():
        assert jobs_of_miner[0].cap_dilution_applied is False
        assert jobs_of_miner[0].effective_rate == pytest.approx(jobs_of_miner[0].hourly_rate)
