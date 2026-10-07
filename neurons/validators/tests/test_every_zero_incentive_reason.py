"""Every reason a node earns 0 is recorded, not only the first one found.

A node blocked by one rule used to hide every later rule: an 8x flagship with no Discord
learned about the flagship gate only after connecting Discord. And a node whose validation
failed earned 0 with an empty reason list. Both cases now reach the backend as data.
"""
from unittest.mock import AsyncMock

import pytest
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


def _logged(caplog, reason: str) -> int:
    return sum(1 for record in caplog.records if getattr(record.msg, "extra", {}).get("reason") == reason)


# ── a zero caused by a failed validation carries the failing check's code ─────


# ── a run that passed every check but scored 0 is not a failed check ─────────


# ── excluded nodes: scored 0, never an exception ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusion", [{"is_spot": True}, {"is_provider_banned": True}], ids=["spot", "banned"])
async def test_a_rented_excluded_node_scores_zero(exclusion):
    rented_healthy = _idle_flagship(gpu_count=4, spec=None, is_rented=True)
    redis = AsyncMock(get_portion_per_gpu_type=AsyncMock(return_value=0.3))
    incentive = RentalPriceIncentive(IncentiveConfig(), redis, {"hk": [rented_healthy]}, {H200: 4})
    assert (await incentive.calculate_executor_score(rented_healthy)).mining_score > 0

    result = await incentive.calculate_executor_score(
        _idle_flagship(gpu_count=4, spec=None, is_rented=True, **exclusion)
    )

    assert result.mining_score == 0
    assert result.eligible_for_rental_share is False
    assert len(_codes(result)) == 1


@pytest.mark.asyncio
async def test_an_excluded_node_with_a_model_the_validator_does_not_know_scores_zero():
    # get_base_model_for_gpu raises on an unknown model; the exclusion is its reason, not a crash
    result = await _incentive().calculate_executor_score(_idle_flagship(gpu_model="NVIDIA FAKE 9000", is_spot=True))

    assert _codes(result) == ["spot_tier"]
    assert result.mining_score == 0
    assert result.eligible_for_rental_share is False
