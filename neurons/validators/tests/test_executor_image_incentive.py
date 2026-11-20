from unittest.mock import AsyncMock

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
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


