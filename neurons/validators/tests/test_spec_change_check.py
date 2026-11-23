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
        # DAH-6546: one card under two datacenter names, as the backend's GPU-set move treats it
        ("NVIDIA B300 SXM6 AC:8", "NVIDIA B300 SXM6 PC:8", True, Msg.SPEC_UNCHANGED.reason, False),
        ("NVIDIA B300 SXM6 PC:8", "NVIDIA B300 SXM6 AC:8", True, Msg.SPEC_UNCHANGED.reason, False),
        # every other model change, or a count change across the pair, still resets
        ("NVIDIA B300 SXM6 AC:8", "NVIDIA B300 SXM6 PC:4", False, Msg.SPEC_CHANGED.reason, True),
        ("NVIDIA B300 SXM6 AC:8", "NVIDIA B200:8", False, Msg.SPEC_CHANGED.reason, True),
        ("NVIDIA B300 SXM6 PC:8", "NVIDIA GB300:8", False, Msg.SPEC_CHANGED.reason, True),
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
        assert result.updates.get("gpu_respec") is True
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


@pytest.mark.parametrize(
    "verified_spec,new_spec",
    [
        ("NVIDIA H100 80GB HBM3:8", "NVIDIA B300 SXM6 AC:8"),
        ("NVIDIA B300 SXM6 AC:8", "NVIDIA H100 80GB HBM3:8"),
    ],
)
@pytest.mark.parametrize("rented", [False, True])
@pytest.mark.asyncio
async def test_other_cards_of_another_model_stay_unverified(verified_spec, new_spec, rented, context_factory):
    # the fingerprint check re-anchored a full set of other cards this cycle: another machine, not a rename
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(gpu_model_count=new_spec),
        verified={"spec": verified_spec},
    ).model_copy(update={"gpu_reanchor": True, "gpu_fingerprint_passed": True, "rented": rented})

    result = await SpecChangeCheck().run(ctx)

    assert result.passed is False
    assert result.fatal is True
    assert result.event.reason_code == Msg.SPEC_CHANGED.reason
    assert result.updates.get("clear_verified_job_info") is True
    # the record keeps the node's model and cards, so the next cycle fails the same way
    assert not result.updates.get("gpu_respec")
    assert result.updates.get("gpu_reanchor") is False


@pytest.mark.asyncio
async def test_other_cards_of_the_same_card_pair_still_pass(context_factory):
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(gpu_model_count="NVIDIA B300 SXM6 PC:8"),
        verified={"spec": "NVIDIA B300 SXM6 AC:8"},
    ).model_copy(update={"gpu_reanchor": True})

    result = await SpecChangeCheck().run(ctx)

    assert result.passed is True
    assert "gpu_reanchor" not in result.updates
