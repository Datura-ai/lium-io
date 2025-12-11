import pytest

from neurons.validators.src.services.task.checks.finalize import FinalizeCheck
from neurons.validators.src.services.task.messages import FinalizeMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "score,job_score,score_warning,gpu_model,gpu_count,contract_version,sysbox_runtime,expected_success,expected_severity",
    [
        # Success case: positive score, no warnings
        (1.0, 1.0, None, "NVIDIA RTX 4090", 2, "v1.0.0", True, True, "info"),
        # Success case: positive score with warning
        (0.5, 0.5, " WARNING: Low collateral", "NVIDIA RTX 3090", 1, "v0.9.0", False, True, "info"),
        # Failure case: zero score, no warnings
        (0.0, 0.0, None, "NVIDIA RTX 3080", 1, "v1.0.0", True, False, "warning"),
        # Failure case: zero score with warning
        (0.0, 0.0, " WARNING: Insufficient ports", "NVIDIA RTX 3070", 4, "v1.0.0", False, False, "warning"),
        # Edge case: no GPU info
        (1.0, 1.0, None, None, None, None, False, True, "info"),
    ],
)
@pytest.mark.asyncio
async def test_finalize_check(
    score,
    job_score,
    score_warning,
    gpu_model,
    gpu_count,
    contract_version,
    sysbox_runtime,
    expected_success,
    expected_severity,
    context_factory,
):
    # Setup services
    services = build_services()

    # Setup config
    config = build_context_config()

    # Setup state with GPU info
    state = build_state(
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        sysbox_runtime=sysbox_runtime,
    )

    # Create context with scores and warnings
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        score=score,
        job_score=job_score,
        score_warning=score_warning,
        contract_version=contract_version,
    )

    # Run the check
    result = await FinalizeCheck().run(ctx)

    # Verify result
    assert result.passed is True  # FinalizeCheck always passes
    assert result.event.reason_code == Msg.COMPLETED.reason
    assert result.event.severity == expected_severity

    # Verify updates
    assert "success" in result.updates
    assert result.updates["success"] is expected_success
    assert result.updates["log_status"] == expected_severity
    assert result.updates["log_text"] == result.event.event

    # Verify impact message
    assert f"Job score={job_score}" in result.event.impact
    assert f"actual score={score}" in result.event.impact

    # Verify remediation based on success and warnings
    if expected_success:
        assert "No action needed" in result.event.remediation
        if score_warning:
            assert score_warning in result.event.remediation
    else:
        assert "Address issues" in result.event.remediation
        if score_warning:
            assert score_warning in result.event.remediation

    # Verify what_we_saw
    assert result.event.what_we_saw.get("gpu_model") == gpu_model
    assert result.event.what_we_saw.get("gpu_count") == (gpu_count if gpu_count is not None else 0)
    assert result.event.what_we_saw.get("contract_version") == contract_version
    assert result.event.what_we_saw.get("sysbox_runtime") is sysbox_runtime
    assert "unrented_multiplier" in result.event.what_we_saw
