"""The A100 and L40S 8-card buckets use a lower idle cap than the default 8-card bucket.

These tests replay the production `IncentiveConfig` defaults: a fill above the cap is
diluted by the cap multiplier from settings, a fill under it is paid in full, and the 1×
buckets and the other families are untouched. With either cap back at the default the
dilution assertions fail.
"""

from unittest.mock import AsyncMock

import pytest
from incentive.config import MAX_UNRENTED_GPUS_BY_TYPE, IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task import JobResult

from tests.test_rental_price_incentive_flow import _make_pcc_job, _total_gpu_counts

A100 = "NVIDIA A100-SXM4-80GB"
L40S = "NVIDIA L40S"


async def _run_with_production_config(
    job_results: dict[str, list[JobResult]],
) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)

    incentive = RentalPriceIncentive(
        IncentiveConfig(),
        redis,
        job_results,
        total_gpu_model_count_map=_total_gpu_counts(job_results),
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()
    return incentive


def test_a100_and_l40s_caps_and_nothing_else_moved():
    assert MAX_UNRENTED_GPUS_BY_TYPE["A100"] == {1: 10, 8: 40}
    assert MAX_UNRENTED_GPUS_BY_TYPE["L40S"] == {1: 10, 8: 16}
    for family, buckets in MAX_UNRENTED_GPUS_BY_TYPE.items():
        if family in ("B300", "A100", "L40S"):
            continue
        assert buckets in ({}, {1: 10, 8: 64}), family


@pytest.mark.asyncio
async def test_idle_8_card_a100_nodes_over_the_cap_share_the_capped_pay():
    """Six idle 8-card A100 nodes exceed the bucket cap and are diluted by the cap multiplier."""
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
async def test_idle_8_card_l40s_nodes_over_the_cap_share_the_capped_pay():
    """Three idle 8-card L40S nodes exceed the bucket cap and are diluted by the cap multiplier."""
    jobs = {f"miner_{i}": [_make_pcc_job(f"l40s-8x-{i}", L40S, 8)] for i in range(3)}

    incentive = await _run_with_production_config(jobs)

    assert incentive.unrented_count_by_bucket[("L40S", 8)] == 24
    assert incentive.cap_multiplier_by_bucket[("L40S", 8)] == pytest.approx(16 / 24)
    for jobs_of_miner in jobs.values():
        assert jobs_of_miner[0].max_cap == 16
        assert jobs_of_miner[0].cap_dilution_applied is True


@pytest.mark.asyncio
async def test_one_idle_node_per_family_is_paid_in_full():
    """One idle 8-card node per family sits under the cap, so it is paid in full."""
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
