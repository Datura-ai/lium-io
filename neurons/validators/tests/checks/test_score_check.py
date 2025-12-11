import pytest

from neurons.validators.src.services.task.checks.score import ScoreCheck
from neurons.validators.src.services.task.messages import ScoreMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


class DummyScoreCalculator:
    """Mock score calculator that mimics the real calc_scores function."""

    def __init__(self, *, actual_score: float, job_score: float, warning_message: str = ""):
        """
        Args:
            actual_score: The actual score to return
            job_score: The job score to return
            warning_message: Optional warning message
        """
        self.actual_score = actual_score
        self.job_score = job_score
        self.warning_message = warning_message
        self.called_with: dict | None = None

    def __call__(
        self,
        gpu_model: str,
        collateral_deposited: bool,
        is_rental_succeed: bool,
        contract_version: str,
        rented: bool,
        port_count: int,
    ) -> tuple[float, float, str]:
        """Mock score calculator that tracks calls and returns configured scores."""
        self.called_with = {
            "gpu_model": gpu_model,
            "collateral_deposited": collateral_deposited,
            "is_rental_succeed": is_rental_succeed,
            "contract_version": contract_version,
            "rented": rented,
            "port_count": port_count,
        }
        return self.actual_score, self.job_score, self.warning_message


@pytest.mark.parametrize(
    "gpu_model,collateral_deposited,is_rental_succeed,contract_version,rented,port_count,actual_score,job_score,warning_msg,expected_pass",
    [
        # Normal success case: full scores, no warnings
        ("NVIDIA RTX 4090", True, True, "v1.0.0", False, 10, 1.0, 1.0, "", True),
        # Success with warning
        ("NVIDIA RTX 3090", True, False, "v1.0.0", False, 10, 0.0, 1.0, " WARNING: Score set to 0 pending rental verification", True),
        # Low collateral warning
        ("NVIDIA RTX 3080", False, True, "v1.0.0", False, 10, 1.0, 1.0, " WARNING: No collateral deposited", True),
        # Insufficient ports - zero score
        ("NVIDIA RTX 3070", True, True, "v1.0.0", False, 2, 0.0, 0.0, " WARNING: Insufficient ports: 2 available, 3 required", True),
        # Rented machine
        ("NVIDIA RTX 4090", True, True, "v1.0.0", True, 10, 1.0, 1.0, "", True),
        # Old contract version
        ("NVIDIA RTX 3090", True, True, "v0.9.0", False, 10, 0.0, 0.0, " WARNING: Outdated contract version (current: v0.9.0, latest: v1.0.0)", True),
        # Edge case: no GPU model
        ("", True, True, "v1.0.0", False, 10, 1.0, 1.0, "", True),
    ],
)
@pytest.mark.asyncio
async def test_score_check(
    gpu_model,
    collateral_deposited,
    is_rental_succeed,
    contract_version,
    rented,
    port_count,
    actual_score,
    job_score,
    warning_msg,
    expected_pass,
    context_factory,
):
    # Create mock score calculator
    score_calculator = DummyScoreCalculator(
        actual_score=actual_score,
        job_score=job_score,
        warning_message=warning_msg,
    )

    # Setup services with mock score calculator
    services = build_services(score_calculator=score_calculator)

    # Setup config
    config = build_context_config()

    # Setup state with GPU model
    state = build_state(gpu_model=gpu_model)

    # Create context with score-related fields
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        collateral_deposited=collateral_deposited,
        is_rental_succeed=is_rental_succeed,
        contract_version=contract_version,
        rented=rented,
        port_count=port_count,
    )

    # Run the check
    result = await ScoreCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == Msg.SCORE_COMPUTED.reason

    # Verify score calculator was called with correct parameters
    assert score_calculator.called_with is not None
    assert score_calculator.called_with["gpu_model"] == gpu_model
    assert score_calculator.called_with["collateral_deposited"] is collateral_deposited
    assert score_calculator.called_with["is_rental_succeed"] is is_rental_succeed
    assert score_calculator.called_with["contract_version"] == contract_version
    assert score_calculator.called_with["rented"] is rented
    assert score_calculator.called_with["port_count"] == port_count

    # Verify updates
    assert "score" in result.updates
    assert result.updates["score"] == actual_score
    assert result.updates["job_score"] == job_score
    assert result.updates["score_warning"] == (warning_msg or None)

    # Verify impact message
    assert f"Job score={job_score}" in result.event.impact
    assert f"actual score={actual_score}" in result.event.impact

    # Verify what_we_saw contains scores and warning
    assert result.event.what_we_saw.get("actual_score") == actual_score
    assert result.event.what_we_saw.get("job_score") == job_score
    assert result.event.what_we_saw.get("warning_message") == warning_msg
