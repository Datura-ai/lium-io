import pytest

from neurons.validators.src.services.task.checks.spec_change import SpecChangeCheck

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "verified_spec,new_spec,expected_pass,expected_reason,expect_clear",
    [
        ("", "model:2", True, "SPEC_UNCHANGED", False),
        ("model:2", "model:2", True, "SPEC_UNCHANGED", False),
        ("model:2", "model:4", False, "SPEC_CHANGED", True),
    ],
)
@pytest.mark.asyncio
async def test_spec_change_check(verified_spec, new_spec, expected_pass, expected_reason, expect_clear, context_factory):
    services = build_services()
    config = build_context_config()
    state = build_state(gpu_model_count=new_spec)

    verified = {"spec": verified_spec} if verified_spec else {}
    ctx = context_factory(services=services, config=config, state=state, verified=verified)

    result = await SpecChangeCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
