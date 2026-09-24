"""Every reason a node earns 0 is recorded, not only the first one found.

A node blocked by one rule used to hide every later rule: an 8x flagship with no Discord
learned about the flagship gate only after connecting Discord.
"""
import logging
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from core.config import settings, shared_client
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"
# a scrape without the NCU observation: the flagship gate reads it as "not unrestricted"
FLAGSHIP_SPEC: dict = {"hard_disk": {"total": 200 * 1024**2}, "gpu": {"details": [{"capacity": 141 * 1024}] * 8}}


def _incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _idle_flagship(**overrides) -> JobResult:
    fields: dict = dict(
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
        ),
        spec=dict(FLAGSHIP_SPEC),
        score=1.0,
        job_score=1.0,
        job_batch_id="2026-09-24 00:00:00",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=8,
        collateral_deposited=True,
        sysbox_runtime=True,
    )
    fields.update(overrides)
    return JobResult(**fields)


def _failed(**overrides) -> JobResult:
    return _idle_flagship(
        spec=None, score=0, job_score=0, log_status="error", gpu_model=None, gpu_count=0, **overrides
    )


def _codes(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


@pytest.fixture
def discord_cutoff_passed(monkeypatch):
    monkeypatch.setattr(settings, "DISCORD_INCENTIVE_CUTOFF", datetime(2020, 1, 1))


@pytest.mark.asyncio
async def test_flagship_gate_is_recorded_even_when_discord_blocks(monkeypatch, discord_cutoff_passed):
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)

    result = await _incentive().calculate_executor_score(_idle_flagship(provider_discord_connected=False))

    # the first entry is the one the old first-match evaluation reported
    assert _codes(result) == ["provider_discord_not_connected", "flagship_without_ncu_or_split"]
    assert result.mining_score == 0
    assert result.eligible_for_rental_share is False


@pytest.mark.asyncio
async def test_every_hard_exclusion_and_every_enforced_gate_is_recorded_in_order(
    monkeypatch, discord_cutoff_passed
):
    for flag in (
        "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT",
        "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT",
        "ENABLE_UNRENTED_SOFT_PRICE_LIMIT",
    ):
        monkeypatch.setattr(settings, flag, True)
    new_cfg = shared_client.config.model_copy(update={"machine_prices_p90": {H200: 0.5}})
    monkeypatch.setattr(shared_client, "_config", new_cfg)

    result = await _incentive().calculate_executor_score(
        _idle_flagship(provider_discord_connected=False, is_new_rentals_paused=True, is_spot=True)
    )

    assert _codes(result) == [
        "spot_tier",
        "provider_discord_not_connected",
        "new_rentals_paused",
        "price_above_market_p90_soft_limit",
        "insufficient_disk_for_vram",
        "flagship_without_ncu_or_split",
    ]
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_two_idle_pool_gates_are_both_recorded(monkeypatch):
    # price used to stop the chain: a node over the price limit never heard about its disk
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    new_cfg = shared_client.config.model_copy(update={"machine_prices_p90": {H200: 0.5}})
    monkeypatch.setattr(shared_client, "_config", new_cfg)

    result = await _incentive().calculate_executor_score(_idle_flagship(gpu_count=4))

    assert _codes(result) == ["price_above_market_p90_soft_limit", "insufficient_disk_for_vram"]
    assert result.eligible_for_rental_share is False


@pytest.mark.asyncio
async def test_a_shadow_gate_is_not_a_reason_and_logs_nothing_on_a_blocked_node(
    monkeypatch, caplog, discord_cutoff_passed
):
    # flag off: not a blocking reason; its shadow line keeps counting eligible nodes only
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", False)

    with caplog.at_level(logging.INFO):
        result = await _incentive().calculate_executor_score(
            _idle_flagship(provider_discord_connected=False)
        )

    assert _codes(result) == ["provider_discord_not_connected"]
    assert not [
        record
        for record in caplog.records
        if getattr(record.msg, "extra", {}).get("reason") == "flagship_without_ncu_or_split"
    ]


@pytest.mark.asyncio
async def test_driver_and_sysbox_are_recorded_on_a_node_blocked_before_pricing(
    monkeypatch, discord_cutoff_passed
):
    monkeypatch.setattr(settings, "MIN_DRIVER_CUTOFF", datetime(2020, 1, 1))
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX_UNRENTED", 1)

    result = await _incentive().calculate_executor_score(
        _idle_flagship(
            gpu_count=4,
            provider_discord_connected=False,
            nvidia_driver_version="470.0.1",
            sysbox_runtime=False,
        )
    )

    assert _codes(result) == [
        "provider_discord_not_connected",
        "nvidia_driver_below_minimum",
        "sysbox_not_enabled",
    ]
    assert result.zero_incentive_reasons[1].context["driver_multiplier"] == 0.0


@pytest.mark.asyncio
async def test_an_eligible_node_whose_rate_collapses_on_two_factors_gets_both(monkeypatch):
    incentive = _incentive()
    job = _idle_flagship(gpu_count=1)
    job.eligible_for_rental_share = True
    job.hourly_rate = 5.0
    job.sysbox_multiplier = 0.0
    job.sysbox_runtime = False
    job.driver_multiplier = 0.0
    job.count_bucket = 1
    job.max_cap = 0

    await incentive._post_process_job_result("hk", job)

    assert _codes(job) == [
        "no_unrented_capacity_for_gpu_count",
        "nvidia_driver_below_minimum",
        "sysbox_not_enabled",
    ]


@pytest.mark.asyncio
async def test_a_healthy_idle_node_has_no_reason():
    result = await _incentive().calculate_executor_score(_idle_flagship(gpu_count=4, spec=None))

    assert result.zero_incentive_reasons == []
    assert result.eligible_for_rental_share is True
