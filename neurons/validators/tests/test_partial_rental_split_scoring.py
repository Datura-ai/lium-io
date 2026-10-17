"""DAH-2467 — mixed scoring for partially rented GPU-split nodes.

A split-opted-in executor with only part of its GPUs rented earns in BOTH pools:
the rented GPUs in the mining pool, the free GPUs in the unrented (rental-share)
pool. The engine models it as two virtual JobResults and merges them back into
one executor result after scoring.
"""

from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult


H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default
MINER_HOTKEY = "miner-hotkey-1"


def _make_job(
    *,
    gpu_count: int = 8,
    rented_gpu_count: int | None = 1,
    is_rented: bool = True,
    supports_gpu_splitting: bool = True,
    gpu_splitting_min_count: int | None = 1,
) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="exec-split-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=gpu_count,
        is_rented=is_rented,
        rented_gpu_count=rented_gpu_count,
        supports_gpu_splitting=supports_gpu_splitting,
        gpu_splitting_min_count=gpu_splitting_min_count,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def _build_incentive(job: JobResult) -> RentalPriceIncentive:
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    return RentalPriceIncentive(
        IncentiveConfig(),
        redis_service,
        {MINER_HOTKEY: [job]},
        {H200: job.gpu_count},
    )


@pytest.mark.asyncio
async def test_calculate_mining_scores_pays_both_pools_and_merges_back(monkeypatch):
    # Arrange — 8-GPU split node, 1 GPU rented; rental share pinned so the test is deterministic.
    job = _make_job(gpu_count=8, rented_gpu_count=1)
    incentive = _build_incentive(job)
    monkeypatch.setattr(incentive, "_calculate_rental_share", AsyncMock(return_value=0.1))

    # Act
    await incentive.calculate_mining_scores()

    # Assert — merged back into ONE result with the full GPU count.
    results = incentive.job_results[MINER_HOTKEY]
    assert results == [job]
    assert job.gpu_count == 8
    assert job.is_rented is True

    # The rented GPU earned in the mining pool: mining_score computed on 1 of 8 GPUs.
    expected_mining_score = 1.0 * 0.3 * 1 / 8
    assert job.mining_score == pytest.approx(expected_mining_score)

    # The free GPUs earned in the unrented pool: with a single unrented participant the whole
    # (pinned) rental share lands on this executor, on top of its mining incentive.
    mining_incentive = incentive.mining_share  # sole miner in the mining pool
    assert job.incentive == pytest.approx(mining_incentive + 0.1)

    # Miner weight aggregation saw both portions.
    assert incentive.miner_incentives[MINER_HOTKEY] == pytest.approx(job.incentive)

    # The unrented accounting counted the 7 free GPUs in the min-count bucket.
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 7


@pytest.mark.asyncio
async def test_plain_idle_node_earns_only_in_the_unrented_pool(monkeypatch):
    # Arrange
    job = _make_job(gpu_count=8, rented_gpu_count=None, is_rented=False)
    incentive = _build_incentive(job)
    monkeypatch.setattr(incentive, "_calculate_rental_share", AsyncMock(return_value=0.1))

    # Act
    await incentive.calculate_mining_scores()

    # Assert
    assert job.incentive_rented == 0.0
    assert job.incentive_idle == pytest.approx(job.incentive)
    assert job.incentive_source == "idle_incentive"
    assert job.node_state_at_cycle == "idle"
    assert job.incentive_formula_version == "rental_price_v2"


