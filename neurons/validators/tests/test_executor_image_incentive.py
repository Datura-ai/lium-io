from unittest.mock import AsyncMock

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from services.task.models import JobResult


@pytest.fixture
def enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", True)


@pytest.fixture
def warn_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", False)


def failed_idle_result() -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="outdated-executor",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=0,
        job_score=0,
        job_batch_id="batch",
        log_status="error",
        log_text="validation failed",
        executor_image_report={
            "status": "OUTDATED",
            "observed_digest": f"sha256:{'b' * 64}",
            "expected_digest": f"sha256:{'a' * 64}",
            "expected_ref": "daturaai/compute-subnet-executor:latest",
        },
    )


@pytest.mark.asyncio
async def test_default_incentive_records_outdated_reason_before_score_early_return(enforce):
    result = failed_idle_result()
    incentive = DefaultIncentive(
        IncentiveConfig(),
        AsyncMock(),
        {"miner": [result]},
        {},
    )

    await incentive._pre_process_job_result("miner", result)

    assert result.mining_score == 0
    assert [reason.reason for reason in result.zero_incentive_reasons] == [
        ZeroIncentiveReason.OUTDATED_EXECUTOR_IMAGE
    ]


@pytest.mark.asyncio
async def test_rental_price_incentive_records_reason_before_unsuccessful_filter(enforce):
    result = failed_idle_result()
    incentive = RentalPriceIncentive(
        IncentiveConfig(),
        AsyncMock(),
        {"miner": [result]},
        {},
    )

    await incentive._pre_process_job_result("miner", result)

    assert [reason.reason for reason in result.zero_incentive_reasons] == [
        ZeroIncentiveReason.OUTDATED_EXECUTOR_IMAGE
    ]


@pytest.mark.asyncio
async def test_rented_outdated_executor_keeps_job_result_but_gets_zero_mining_score(enforce):
    result = failed_idle_result().model_copy(
        update={
            "job_score": 1.0,
            "is_rented": True,
            "gpu_model": "NVIDIA H200",
            "gpu_count": 8,
        }
    )
    incentive = RentalPriceIncentive(
        IncentiveConfig(),
        AsyncMock(),
        {"miner": [result]},
        {"NVIDIA H200": 8},
    )

    await incentive._pre_process_job_result("miner", result)

    assert result.mining_score == 0
    assert result.zero_incentive_reasons[0].reason == "outdated_executor_image"


def idle_result_on_old_image() -> JobResult:
    # A validated idle H200 node whose executor container still runs the previous image.
    return failed_idle_result().model_copy(
        update={
            "score": 1.0,
            "job_score": 1.0,
            "log_status": "success",
            "log_text": "ok",
            "gpu_model": "NVIDIA H200",
            "gpu_count": 8,
        }
    )


def redis_with_full_portion() -> AsyncMock:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type.return_value = 1.0
    return redis


@pytest.mark.asyncio
async def test_warning_only_default_incentive_scores_old_image_node(warn_only):
    # Regression: DAH-2701 zeroed idle pay for ~100 nodes on the old image (11 Sep).
    result = idle_result_on_old_image()
    incentive = DefaultIncentive(
        IncentiveConfig(),
        redis_with_full_portion(),
        {"miner": [result]},
        {"NVIDIA H200": 8},
    )

    await incentive._pre_process_job_result("miner", result)

    assert result.mining_score > 0
    assert ZeroIncentiveReason.OUTDATED_EXECUTOR_IMAGE.value not in [
        reason.reason for reason in result.zero_incentive_reasons
    ]


@pytest.mark.asyncio
async def test_warning_only_rental_price_incentive_records_no_outdated_reason(warn_only):
    result = failed_idle_result()
    incentive = RentalPriceIncentive(
        IncentiveConfig(),
        AsyncMock(),
        {"miner": [result]},
        {},
    )

    await incentive._pre_process_job_result("miner", result)

    assert result.zero_incentive_reasons == []


@pytest.mark.asyncio
async def test_warning_only_rented_old_image_executor_keeps_mining_score(warn_only):
    result = idle_result_on_old_image().model_copy(update={"is_rented": True})
    incentive = RentalPriceIncentive(
        IncentiveConfig(),
        redis_with_full_portion(),
        {"miner": [result]},
        {"NVIDIA H200": 8},
    )

    await incentive._pre_process_job_result("miner", result)

    assert result.mining_score > 0
    assert ZeroIncentiveReason.OUTDATED_EXECUTOR_IMAGE.value not in [
        reason.reason for reason in result.zero_incentive_reasons
    ]
