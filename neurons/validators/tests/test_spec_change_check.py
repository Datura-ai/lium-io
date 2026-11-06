import pytest

from neurons.validators.src.services.task.checks.spec_change import SpecChangeCheck
from neurons.validators.src.services.task.messages import SpecChangeMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "verified_spec,new_spec,expected_pass,expected_reason,expect_clear",
    [
        ("", "model:2", True, Msg.SPEC_UNCHANGED.reason, False),
        ("model:2", "model:2", True, Msg.SPEC_UNCHANGED.reason, False),
        ("model:2", "model:4", False, Msg.SPEC_CHANGED.reason, True),
    ],
)
@pytest.mark.asyncio
async def test_spec_change_check(
    verified_spec, new_spec, expected_pass, expected_reason, expect_clear, context_factory
):
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


@pytest.mark.asyncio
async def test_spec_change_non_fatal_after_rented_fingerprint_failure(context_factory):
    state = build_state(gpu_model_count="model:3")
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        verified={"spec": "model:4"},
    )
    ctx = ctx.model_copy(update={"gpu_fingerprint_passed": False})

    result = await SpecChangeCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.SPEC_CHANGED.reason
    assert result.fatal is False
