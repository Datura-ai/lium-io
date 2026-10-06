from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive import rental_price as rental_price_module
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult


def _make_job(
    *,
    executor_id: str = "executor-discord-test",
    gpu_model: str = "H100",
    gpu_count: int = 1,
    is_rented: bool = False,
    provider_discord_connected: bool = True,
) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=executor_id,
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="discord-test-batch",
        log_status="success",
        log_text="ok",
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=True,
        provider_discord_connected=provider_discord_connected,
    )


def _redis_service() -> AsyncMock:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    return redis


def _set_discord_cutoff(monkeypatch, *, active: bool) -> None:
    cutoff = datetime.utcnow() - timedelta(days=1) if active else datetime.utcnow() + timedelta(days=1)
    monkeypatch.setattr(settings, "DISCORD_INCENTIVE_CUTOFF", cutoff)


@pytest.mark.asyncio
async def test_rental_price_incentive_excludes_executor_without_discord_after_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _make_job(provider_discord_connected=False)
    incentive = _make_rental_price_incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.eligible_for_rental_share is False
    assert job.mining_score == 0
    assert job.effective_rate is None
    assert job.incentive == 0
    assert incentive.unrented_count_by_bucket == {}
    assert incentive.total_rental_cost == 0
    assert incentive.get_snapshot().mining.total_gpu_model_count_map == {}


@pytest.mark.asyncio
async def test_rental_price_incentive_keeps_effective_rate_with_discord_after_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _make_job(provider_discord_connected=True)
    incentive = _make_rental_price_incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.effective_rate == pytest.approx(4.0)
    assert job.incentive > 0


@pytest.mark.asyncio
async def test_rental_price_incentive_does_not_penalize_missing_discord_before_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=False)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _make_job(provider_discord_connected=False)
    incentive = _make_rental_price_incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.effective_rate == pytest.approx(4.0)
    assert job.incentive > 0


@pytest.mark.asyncio
async def test_rental_price_incentive_mixed_discord_bucket_excludes_disconnected_executor(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    connected = _make_job(
        executor_id="connected-executor",
        provider_discord_connected=True,
    )
    disconnected = _make_job(
        executor_id="disconnected-executor",
        provider_discord_connected=False,
    )
    incentive = _make_rental_price_incentive(
        {
            "connected-miner": [connected],
            "disconnected-miner": [disconnected],
        }
    )

    await incentive.calculate_mining_scores()

    # Match spot behavior: disconnected supply is excluded from incentive pools.
    assert incentive.unrented_count_by_bucket[("H100", 1)] == 1
    assert incentive.cap_multiplier_by_bucket[("H100", 1)] == pytest.approx(1.0)
    assert incentive.total_rental_cost == pytest.approx(4.0)
    assert incentive.get_snapshot().mining.total_gpu_model_count_map == {"H100": 1}

    assert connected.effective_rate == pytest.approx(4.0)
    assert connected.incentive > 0
    assert incentive.miner_incentives["connected-miner"] == pytest.approx(connected.incentive)

    assert disconnected.eligible_for_rental_share is False
    assert disconnected.mining_score == 0
    assert disconnected.effective_rate is None
    assert disconnected.incentive == 0
    assert incentive.miner_incentives["disconnected-miner"] == 0


def _make_rental_price_incentive(job_results: dict[str, list[JobResult]]) -> RentalPriceIncentive:
    incentive = RentalPriceIncentive(
        IncentiveConfig(
            algorithm="rental_price",
            rental_incentive_gpu_types=["H100"],
            max_unrented_gpus={"H100": {1: 1}},
            rental_prices_per_hour={"H100": 4.0},
            gpu_count_custom_prices={"H100": {"1": 4.0}},
        ),
        _redis_service(),
        job_results,
        total_gpu_model_count_map={"H100": 1},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 1.0
    price_provider.get_alpha_rate.return_value = 1.0
    incentive.price_provider = price_provider
    return incentive
