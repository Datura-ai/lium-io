"""DAH-2835 — unrented incentive Docker Hub egress gate.

An unrented executor that could not reach registry-1.docker.io for
MIN_REGISTRY_UNREACHABLE_CYCLES_TO_PENALIZE consecutive validation cycles forfeits the
unrented rental incentive while staying active. The gate ships in shadow mode
(ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT off): the breach is only logged and the payout is
unchanged until the flag is turned on.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import MIN_REGISTRY_UNREACHABLE_CYCLES_TO_PENALIZE, RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    *,
    unreachable_cycles: int | None = MIN_REGISTRY_UNREACHABLE_CYCLES_TO_PENALIZE,
    is_rented: bool = False,
    spec: dict | None = None,
) -> JobResult:
    # the power cap gate ships enforced but fails open on a spec without its keys, so a spec
    # carrying only this probe's key reaches the egress gate
    if spec is None:
        spec = {}
        if unreachable_cycles is not None:
            spec["registry_unreachable_cycles"] = unreachable_cycles

    return JobResult(
        spec=spec,
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
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=8,
        is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


@pytest.mark.parametrize("cycles", [0, 1, 2])
def test_short_outage_is_not_flagged(cycles):
    # Arrange — a blip of one or two cycles must never cost a miner the incentive
    incentive = _build_incentive()

    # Act
    measured = incentive._registry_unreachable(_make_job(unreachable_cycles=cycles))

    # Assert
    assert measured is None


def test_sustained_outage_is_flagged():
    # Arrange — three consecutive cycles (~45 minutes) is the threshold
    incentive = _build_incentive()

    # Act
    measured = incentive._registry_unreachable(_make_job(unreachable_cycles=3))

    # Assert
    assert measured is not None
    assert measured.unreachable_cycles == 3


UNPROVEN_SPECS: list[dict] = [
    {},                                       # probe skipped: no measurement this cycle
    {"registry_unreachable_cycles": None},    # probe reported nothing
    {"registry_unreachable_cycles": "3"},     # count as a string
    {"registry_unreachable_cycles": True},    # bool is not a count
]


@pytest.mark.parametrize("spec", UNPROVEN_SPECS)
def test_unproven_probe_is_never_flagged(spec):
    # Fail open: only a proven outage is penalized, so every unreadable value keeps the payout.
    incentive = _build_incentive()

    assert incentive._registry_unreachable(_make_job(spec=spec)) is None


def test_spec_less_job_result_is_not_flagged():
    # estimate_executor builds a spec-less JobResult for every GPU model every cycle.
    incentive = _build_incentive()
    job = _make_job()
    job.spec = None

    assert incentive._registry_unreachable(job) is None


def test_enforcement_is_off_by_default():
    # Rollout contract: this gate ships in shadow mode, so the flag default - not the
    # env-resolved value - must stay False.
    default = type(settings).model_fields["ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT"].default

    assert default is False


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — sustained outage and flag on
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — excluded from the rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_enforced_appends_customer_facing_incentive_log(monkeypatch):
    # DAH-2327: the zero-incentive reason must reach the customer-facing incentive log and
    # name the host the executor cannot reach.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job())

    log = "\n".join(result.incentive_logs)
    assert "registry_unreachable" in log
    assert "registry-1.docker.io" in log
    assert [reason.reason for reason in result.zero_incentive_reasons] == ["registry_unreachable"]


@pytest.mark.asyncio
async def test_short_outage_keeps_eligibility_when_enforced(monkeypatch):
    # Arrange — flag on but only two bad cycles
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(unreachable_cycles=2))

    # Assert
    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_unmeasured_executor_keeps_eligibility_when_enforced(monkeypatch):
    # An absent key is every node the probe skipped, which is most of the fleet on any cycle.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(spec={}))

    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch):
    # Arrange — sustained outage but flag off → shadow only
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", False)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — still eligible for the unrented rental pool, nothing told to the miner
    assert result.eligible_for_rental_share is True
    assert "registry_unreachable" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_shadow_mode_emits_the_measurement_log(monkeypatch, caplog):
    # The shadow log is the whole deliverable of the first deploy: the rollout decision is
    # made from these fields, so they are a dashboard contract.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", False)
    incentive = _build_incentive()

    with caplog.at_level(logging.INFO):
        await incentive.calculate_executor_score(_make_job())

    breach = next(
        r.msg for r in caplog.records
        if hasattr(r.msg, "extra") and r.msg.extra.get("reason") == "registry_unreachable"
    )
    assert "shadow only - flag off" in breach.message
    assert breach.extra["unreachable_cycles"] == 3
    assert breach.extra["enforced"] is False
    assert breach.extra["pool"] == "rental_kept_shadow"


@pytest.mark.asyncio
async def test_rented_executor_is_not_gated(monkeypatch):
    # A rented machine earns from the rented path and must not be touched by this gate.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_REGISTRY_EGRESS_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "registry_unreachable" not in "\n".join(result.incentive_logs)
