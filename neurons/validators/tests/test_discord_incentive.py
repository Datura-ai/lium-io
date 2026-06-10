from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive import rental_price as rental_price_module
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
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
    monkeypatch.setattr(settings, "PORTION_FOR_DISCORD", 1)


@pytest.mark.asyncio
async def test_default_incentive_sets_zero_multiplier_without_discord_after_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=True)
    job = _make_job(is_rented=True, provider_discord_connected=False)
    incentive = DefaultIncentive(
        IncentiveConfig(algorithm="default"),
        _redis_service(),
        {"miner": [job]},
        total_gpu_model_count_map={"H100": 1},
    )

    result = await incentive.calculate_executor_score(job)

    assert result.discord_multiplier == 0
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_default_incentive_does_not_penalize_missing_discord_before_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=False)
    job = _make_job(is_rented=True, provider_discord_connected=False)
    incentive = DefaultIncentive(
        IncentiveConfig(algorithm="default"),
        _redis_service(),
        {"miner": [job]},
        total_gpu_model_count_map={"H100": 1},
    )

    result = await incentive.calculate_executor_score(job)

    assert result.discord_multiplier == 1
    assert result.mining_score == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_rental_price_incentive_sets_zero_effective_rate_without_discord_after_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _make_job(provider_discord_connected=False)
    incentive = _make_rental_price_incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.discord_multiplier == 0
    assert job.effective_rate == 0
    assert job.incentive == 0


@pytest.mark.asyncio
async def test_rental_price_incentive_keeps_effective_rate_with_discord_after_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _make_job(provider_discord_connected=True)
    incentive = _make_rental_price_incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.discord_multiplier == 1
    assert job.effective_rate == pytest.approx(4.0)
    assert job.incentive > 0


@pytest.mark.asyncio
async def test_rental_price_incentive_does_not_penalize_missing_discord_before_cutoff(monkeypatch):
    _set_discord_cutoff(monkeypatch, active=False)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _make_job(provider_discord_connected=False)
    incentive = _make_rental_price_incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.discord_multiplier == 1
    assert job.effective_rate == pytest.approx(4.0)
    assert job.incentive > 0


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
