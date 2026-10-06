"""DAH-2546 / DAH-2594 — unrented incentive flagship capability gate.

An unrented 8x H200/B200/B300 executor that has none of NCU profiling counters
open on the host (ncu_profiling_access == "unrestricted"), real GPU splitting
enabled (min_gpu_count below the full node size), or a verified TDX quote
forfeits the unrented rental incentive while staying active. Enforcement is gated by
ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT; while the flag is off the breach is
only logged (shadow mode) and the payout is unchanged.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive

from core.config import settings
from services.task_service import JobResult

H200 = "NVIDIA H200"
B200 = "NVIDIA B200"
B300 = "NVIDIA B300 SXM6 AC"
H100 = "NVIDIA H100 80GB HBM3"

# a scrape carrying no NCU observation at all: a machine not re-scraped since DAH-2182 shipped
SCRAPE_WITHOUT_NCU: dict[str, str] = {}

# what prepare_host_policy returns for a verified TDX quote: os_image_hash + compose_hash
ATTESTATION_DIGEST = "0f1e2d:9a8b7c"


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    *,
    gpu_model: str = H200,
    gpu_count: int = 8,
    # spec=None models the spec-less synthetic JobResult the estimate path builds
    spec: dict[str, str] | None = SCRAPE_WITHOUT_NCU,
    supports_gpu_splitting: bool = False,
    gpu_splitting_min_count: int | None = None,
    is_rented: bool = False,
    attestation_digest: str | None = None,
    gpu_attestation_passed: bool | None = None,
    tdx_quote: str | None = None,
) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
            tdx_quote=tdx_quote,
        ),
        spec=spec,
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        is_rented=is_rented,
        supports_gpu_splitting=supports_gpu_splitting,
        attestation_digest=attestation_digest,
        gpu_attestation_passed=gpu_attestation_passed,
        gpu_splitting_min_count=gpu_splitting_min_count,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


@pytest.mark.parametrize(
    "job_kwargs, is_missing",
    [
        ({}, True),  # neither capability
        ({"spec": None}, False),  # spec-less synthetic/estimate result: fail open
        ({"spec": {"ncu_profiling_access": "unrestricted"}}, False),
        ({"spec": {"ncu_profiling_access": "restricted"}}, True),
        ({"spec": {"ncu_profiling_access": "unknown"}}, True),
        ({"supports_gpu_splitting": True, "gpu_splitting_min_count": 1}, False),
        # min equal to the node size is whole-host-only in practice, not real splitting
        ({"supports_gpu_splitting": True, "gpu_splitting_min_count": 8}, True),
        # a configured min without hardware support does not count either
        ({"gpu_splitting_min_count": 1}, True),
        ({"gpu_model": B200}, True),
        ({"gpu_model": B300}, True),
        ({"gpu_model": B300, "supports_gpu_splitting": True, "gpu_splitting_min_count": 1}, False),
        ({"gpu_model": H100}, False),  # not a flagship model
        ({"gpu_count": 4}, False),  # only full 8x nodes are gated
        ({"attestation_digest": ATTESTATION_DIGEST}, False),
        ({"gpu_model": B200, "attestation_digest": ATTESTATION_DIGEST}, False),
        # a CVM whose GPU-CC evidence verified bad still counts: GPU attestation is a separate
        # (observe-only) lever, and withholding evidence must never pay more than submitting it
        ({"attestation_digest": ATTESTATION_DIGEST, "gpu_attestation_passed": False}, False),
    ],
)
def test_missing_flagship_capability(job_kwargs, is_missing):
    # Arrange
    incentive = _build_incentive()
    job = _make_job(**job_kwargs)
    base_model = incentive.get_base_model_for_gpu(job.gpu_model)

    # Act
    missing = incentive._missing_flagship_capability(job, base_model)

    # Assert
    assert (missing is not None) is is_missing


def test_enforcement_defaults_to_shadow_mode():
    # Arrange / Act — rollout contract: first deploy must be shadow-only, so the
    # flag default (not the env-resolved value) must stay False.
    default = type(settings).model_fields["ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT"].default

    # Assert
    assert default is False


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch):
    # Arrange — neither capability but flag off → shadow only
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", False)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — still eligible for the unrented rental pool, nothing told to the miner
    assert result.eligible_for_rental_share is True
    assert "flagship_without_ncu_or_split" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_shadow_mode_emits_the_capability_log(monkeypatch, caplog):
    # The shadow log is the whole deliverable of the first deploy: the rollout decision is
    # made from these fields, and the scrape error is what separates a failed probe from a
    # provider who deliberately left the counters closed.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", False)
    incentive = _build_incentive()

    with caplog.at_level(logging.INFO):
        await incentive.calculate_executor_score(
            _make_job(
                spec={
                    "ncu_profiling_access": "unknown",
                    "ncu_profiling_scrape_error": "Cannot read /proc/driver/nvidia/params",
                }
            )
        )

    breach = next(
        record.msg
        for record in caplog.records
        if record.msg.extra.get("reason") == "flagship_without_ncu_or_split"
    )
    assert "shadow only - flag off" in breach.message
    assert breach.extra["ncu_profiling_access"] == "unknown"
    assert breach.extra["ncu_profiling_scrape_error"] == "Cannot read /proc/driver/nvidia/params"
    assert breach.extra["enforced"] is False
    assert breach.extra["pool"] == "rental_kept_shadow"
    assert breach.extra["tdx_quote_present"] is False


@pytest.mark.asyncio
async def test_self_declared_cvm_is_still_excluded(monkeypatch, caplog):
    # A machine that submitted a quote the validator did not verify still fails the gate; the
    # log has to separate it from an ordinary host, since attestation_digest is None on every
    # line this gate emits.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    with caplog.at_level(logging.INFO):
        result = await incentive.calculate_executor_score(
            _make_job(tdx_quote='{"quote": "..."}', attestation_digest=None)
        )

    breach = next(
        record.msg
        for record in caplog.records
        if record.msg.extra.get("reason") == "flagship_without_ncu_or_split"
    )
    assert result.eligible_for_rental_share is False
    assert breach.extra["tdx_quote_present"] is True


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — neither capability and flag on
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — excluded from the rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_enforced_appends_customer_facing_incentive_log(monkeypatch):
    # The zero-incentive reason must reach the miner-facing incentive log with
    # both remediation paths spelled out.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job())

    log = "\n".join(result.incentive_logs)
    assert "flagship_without_ncu_or_split" in log
    assert "NCU profiling" in log
    assert "GPU splitting" in log
    assert "confidential computing" in log
    assert [reason.reason for reason in result.zero_incentive_reasons] == [
        "flagship_without_ncu_or_split"
    ]


@pytest.mark.parametrize(
    "job_kwargs, expect_eligible",
    [
        ({"spec": {"ncu_profiling_access": "unrestricted"}}, True),
        # min equal to the node size (a live prod case: a B200 at min=8) is whole-host-only
        ({"supports_gpu_splitting": True, "gpu_splitting_min_count": 8}, False),
        # estimate_executor feeds spec-less synthetic results through the same scoring
        # path every cycle; the gate must fail open on those (DAH-2520 precedent)
        ({"spec": None}, True),
        # DAH-2594 — an attested CVM cannot open host NCU counters, attestation is its path
        ({"attestation_digest": ATTESTATION_DIGEST}, True),
    ],
)
@pytest.mark.asyncio
async def test_enforced_eligibility_by_capability(monkeypatch, job_kwargs, expect_eligible):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(**job_kwargs))

    # Assert
    assert result.eligible_for_rental_share is expect_eligible


@pytest.mark.asyncio
async def test_rented_flagship_without_capabilities_untouched(monkeypatch):
    # The gate governs only the unrented pool: a rented machine keeps normal scoring
    # and is never told about the missing capabilities.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "flagship_without_ncu_or_split" not in "\n".join(result.incentive_logs)
