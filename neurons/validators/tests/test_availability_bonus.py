"""Tests for availability bonus calculation in calc_job_score.

The availability bonus provides a 20% score multiplier for unrented H100/H200/B200 GPUs
to incentivize keeping these high-value GPUs available on the platform.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.src.services.task.models import JobResult
from neurons.validators.src.services.const import (
    AVAILABILITY_BONUS_MULTIPLIER,
    AVAILABILITY_BONUS_GPU_TYPES,
)


@pytest.mark.parametrize(
    "gpu_model,rented,gpu_count,total_gpu_count,expected_multiplier",
    [
        # Target GPU types - unrented should get 1.2x bonus
        ("NVIDIA H100 80GB HBM3", False, 8, 100, 1.2),
        ("NVIDIA H200", False, 4, 50, 1.2),
        ("NVIDIA H200 NVL", False, 2, 20, 1.2),
        ("NVIDIA B200", False, 1, 10, 1.2),
        # Target GPU types - rented should NOT get bonus
        ("NVIDIA H100 80GB HBM3", True, 8, 100, 1.0),
        ("NVIDIA H200", True, 4, 50, 1.0),
        ("NVIDIA H200 NVL", True, 2, 20, 1.0),
        ("NVIDIA B200", True, 1, 10, 1.0),
        # Non-target GPU types - never get bonus
        ("NVIDIA RTX 4090", False, 8, 100, 1.0),
        ("NVIDIA RTX 4090", True, 8, 100, 1.0),
        ("NVIDIA A100 80GB PCIe", False, 4, 50, 1.0),
        ("NVIDIA H100 PCIe", False, 2, 20, 1.0),  # Not in target list
        ("NVIDIA H800 80GB HBM3", False, 2, 20, 1.0),  # Not in target list
        # Edge cases
        ("NVIDIA H100 80GB HBM3", False, 1, 1, 1.2),  # Single GPU
        ("NVIDIA B200", False, 0, 10, 0.0),  # Zero GPU count (should return 0)
    ],
)
@pytest.mark.asyncio
async def test_calc_job_score_availability_bonus(
    gpu_model,
    rented,
    gpu_count,
    total_gpu_count,
    expected_multiplier,
):
    """Test that availability bonus is correctly applied based on GPU type and rented status.

    Arrange:
        - Create a Validator instance with mocked dependencies
        - Create a JobResult with specific GPU model, rented status, and count
        - Set up total GPU count map

    Act:
        - Call calc_job_score with the test data

    Assert:
        - The score includes the availability bonus multiplier when applicable
        - Non-target GPUs or rented GPUs don't receive the bonus
    """
    from core.validator import Validator
    from core.config import settings

    # Create validator and mock dependencies
    validator = Validator()

    # Mock redis service
    validator.redis_service = AsyncMock()
    validator.redis_service.get_portion_per_gpu_type = AsyncMock(return_value=1.0)
    validator.redis_service.get_executor_uptime = AsyncMock(return_value=10000)

    # Create executor info
    executor_info = ExecutorSSHInfo(
        uuid="test-executor-uuid",
        address="192.168.1.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )

    # Base score for calculation (before multiplier)
    base_score = 1.0
    base_job_score = 1.0

    # Create JobResult
    job_result = JobResult(
        spec=None,
        executor_info=executor_info,
        score=base_score,
        job_score=base_job_score,
        collateral_deposited=True,
        job_batch_id="test-batch-123",
        log_status="info",
        log_text="Test",
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        sysbox_runtime=True,
        ssh_pub_keys=None,
        rented=rented,
    )

    # Total GPU count map
    total_gpu_model_count_map = {gpu_model: total_gpu_count}

    # Act
    if gpu_count == 0:
        # When GPU count is 0, score should be 0
        result = await validator.calc_job_score(total_gpu_model_count_map, job_result)
        assert result == 0, "Score should be 0 when GPU count is 0"
    else:
        result = await validator.calc_job_score(total_gpu_model_count_map, job_result)

        # Expected score calculation:
        # base_score * score_portion * gpu_count / total_gpu_count * (1 + bonus)
        expected_score = (
            base_score
            * 1.0  # score_portion (mocked)
            * gpu_count
            / total_gpu_count
            * expected_multiplier
        )

        assert abs(result - expected_score) < 0.0001, (
            f"Expected score {expected_score} for {gpu_model} "
            f"(rented={rented}, bonus={expected_multiplier}), got {result}"
        )


def test_availability_bonus_constants():
    """Verify that availability bonus constants are correctly defined."""
    from neurons.validators.src.services.const import (
        AVAILABILITY_BONUS_MULTIPLIER,
        AVAILABILITY_BONUS_GPU_TYPES,
    )

    # Bonus multiplier should be 0.2 (20%)
    assert AVAILABILITY_BONUS_MULTIPLIER == 0.2, (
        f"AVAILABILITY_BONUS_MULTIPLIER should be 0.2, got {AVAILABILITY_BONUS_MULTIPLIER}"
    )

    # GPU types should include expected models
    expected_models = {
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200",
        "NVIDIA H200 NVL",
        "NVIDIA B200",
    }

    assert AVAILABILITY_BONUS_GPU_TYPES == expected_models, (
        f"AVAILABILITY_BONUS_GPU_TYPES should contain {expected_models}, "
        f"got {AVAILABILITY_BONUS_GPU_TYPES}"
    )

    # Verify it's a frozenset (immutable)
    assert isinstance(AVAILABILITY_BONUS_GPU_TYPES, frozenset), (
        "AVAILABILITY_BONUS_GPU_TYPES should be a frozenset"
    )


def test_job_result_has_rented_field():
    """Verify that JobResult model has the rented field."""
    from neurons.validators.src.services.task.models import JobResult
    from datura.requests.miner_requests import ExecutorSSHInfo
    from pydantic import ValidationError

    executor_info = ExecutorSSHInfo(
        uuid="test-uuid",
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )

    # Should accept rented=False
    job_result_unrented = JobResult(
        spec=None,
        executor_info=executor_info,
        score=1.0,
        job_score=1.0,
        collateral_deposited=True,
        job_batch_id="batch-123",
        log_status="info",
        log_text="Test",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=8,
        sysbox_runtime=True,
        rented=False,
    )
    assert job_result_unrented.rented is False

    # Should accept rented=True
    job_result_rented = JobResult(
        spec=None,
        executor_info=executor_info,
        score=1.0,
        job_score=1.0,
        collateral_deposited=True,
        job_batch_id="batch-123",
        log_status="info",
        log_text="Test",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=8,
        sysbox_runtime=True,
        rented=True,
    )
    assert job_result_rented.rented is True

    # Default value should be False
    job_result_default = JobResult(
        spec=None,
        executor_info=executor_info,
        score=1.0,
        job_score=1.0,
        collateral_deposited=True,
        job_batch_id="batch-123",
        log_status="info",
        log_text="Test",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=8,
        sysbox_runtime=True,
    )
    assert job_result_default.rented is False


@pytest.mark.parametrize(
    "gpu_model,should_receive_bonus",
    [
        # Target GPU types - should receive bonus
        ("NVIDIA H100 80GB HBM3", True),
        ("NVIDIA H200", True),
        ("NVIDIA H200 NVL", True),
        ("NVIDIA B200", True),
        # Non-target GPU types - should NOT receive bonus
        ("NVIDIA H100 PCIe", False),
        ("NVIDIA H100 NVL", False),
        ("NVIDIA H800 80GB HBM3", False),
        ("NVIDIA H800 NVL", False),
        ("NVIDIA RTX 4090", False),
        ("NVIDIA RTX 6000 Ada Generation", False),
        ("NVIDIA A100 80GB PCIe", False),
        ("NVIDIA A100-SXM4-80GB", False),
        ("NVIDIA B300 SXM6 AC", False),
        ("", False),  # Empty string
    ],
)
def test_gpu_model_in_bonus_set(gpu_model, should_receive_bonus):
    """Test that GPU models are correctly identified for availability bonus."""
    is_in_bonus_set = gpu_model in AVAILABILITY_BONUS_GPU_TYPES
    assert is_in_bonus_set == should_receive_bonus, (
        f"GPU model '{gpu_model}' bonus status mismatch: "
        f"expected {should_receive_bonus}, got {is_in_bonus_set}"
    )
