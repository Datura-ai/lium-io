from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task.models import JobResult


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


def _rented(status: str) -> JobResult:
    result = failed_idle_result().model_copy(
        update={"job_score": 1.0, "is_rented": True, "gpu_model": "NVIDIA H200", "gpu_count": 8}
    )
    result.executor_image_report = {**result.executor_image_report, "status": status, "rented_when_read": True}
    return result


@pytest.mark.asyncio
async def test_rented_outdated_executor_keeps_its_incentive():
    # DAH-6845: an old image never costs a renter's node its share; it leaves the rent list once free
    result = _rented("OUTDATED")
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {"miner": [result]}, {"NVIDIA H200": 8})

    await incentive._pre_process_job_result("miner", result)

    assert "outdated_executor_image" not in [r.reason for r in result.zero_incentive_reasons]


@pytest.mark.asyncio
async def test_rented_unofficial_executor_keeps_job_result_but_gets_zero_mining_score():
    result = _rented("UNOFFICIAL")
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {"miner": [result]}, {"NVIDIA H200": 8})

    await incentive._pre_process_job_result("miner", result)

    assert result.mining_score == 0
    assert result.zero_incentive_reasons[0].reason == "outdated_executor_image"




@pytest.mark.asyncio
async def test_outdated_image_read_while_rented_keeps_its_incentive_when_the_rental_ended_later():
    # DAH-6845 review: the rental ended between the image reading and the cycle end
    result = _rented("OUTDATED").model_copy(update={"is_rented": False})
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {"miner": [result]}, {"NVIDIA H200": 8})

    await incentive._pre_process_job_result("miner", result)

    assert "outdated_executor_image" not in [r.reason for r in result.zero_incentive_reasons]


@pytest.mark.asyncio
async def test_outdated_image_read_on_a_free_node_is_excluded():
    result = failed_idle_result()
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {"miner": [result]}, {"NVIDIA H200": 8})

    await incentive._pre_process_job_result("miner", result)

    assert "outdated_executor_image" in [r.reason for r in result.zero_incentive_reasons]
