"""DAH-6241: a node whose card is not the model it reports earns nothing."""

import pytest

from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.gpu_model_truth import (
    EXPECTED_CC_AND_SM,
    gpu_model_mismatch,
)
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg

from tests.helpers import build_context_config, build_services, build_state
from tests.test_capability_check import DummyValidationService


def _specs(model: str) -> dict:
    return {"gpu": {"count": 1, "details": [{"name": model}]}}


# prod, 12 Nov 2026: the three nodes of one hotkey that report a model their card is not
FAKED = [
    ("NVIDIA H100 80GB HBM3", "8.0", 108),  # an A100
    ("NVIDIA GeForce RTX 5070 Ti", "8.6", 48),  # an Ampere card
    ("NVIDIA RTX 6000 Ada Generation", "8.6", 84),  # an RTX A6000
]


@pytest.mark.parametrize("model", sorted(EXPECTED_CC_AND_SM))
def test_every_model_on_the_fleet_passes_with_its_own_card(model):
    # the table is what the sealed matmul read on the fleet's honest nodes of each name
    cc, sm_count = EXPECTED_CC_AND_SM[model]
    assert gpu_model_mismatch(_specs(model), {"cc": cc, "sm_count": sm_count}) is None


@pytest.mark.parametrize("model, cc, sm_count", FAKED)
def test_a_card_that_is_not_the_reported_model_fails(model, cc, sm_count):
    mismatch = gpu_model_mismatch(_specs(model), {"cc": cc, "sm_count": sm_count, "device": "x"})

    assert mismatch is not None
    assert mismatch["measured_compute_capability"] == cc
    assert mismatch["measured_sm_count"] == sm_count
    assert (mismatch["expected_compute_capability"], mismatch["expected_sm_count"]) == (
        EXPECTED_CC_AND_SM[model]
    )


def test_the_same_generation_with_fewer_sms_fails():
    # a cut-down card of the same generation is still not the reported model
    assert gpu_model_mismatch(_specs("NVIDIA GeForce RTX 4090"), {"cc": "8.9", "sm_count": 76})


@pytest.mark.parametrize(
    "specs, metrics",
    [
        (_specs("NVIDIA Some Future GPU"), {"cc": "8.0", "sm_count": 108}),
        (_specs("NVIDIA H100 80GB HBM3"), None),
        (_specs("NVIDIA H100 80GB HBM3"), {}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "8.0"}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "", "sm_count": 108}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "8.0", "sm_count": "108"}),
        ({"gpu": {"count": 1}}, {"cc": "8.0", "sm_count": 108}),
    ],
    ids=["unknown_model", "no_metrics", "empty_metrics", "no_sm", "empty_cc", "sm_text", "no_name"],
)
def test_an_unknown_model_or_a_run_without_both_numbers_never_fails(specs, metrics):
    assert gpu_model_mismatch(specs, metrics) is None


async def _run(context_factory, model: str, cc: str, sm_count: int):
    service = DummyValidationService(success=True, metrics={"cc": cc, "sm_count": sm_count})
    ctx = context_factory(
        services=build_services(validation=service),
        config=build_context_config(),
        state=build_state(specs=_specs(model)),
    )
    return await CapabilityCheck().run(ctx)


@pytest.mark.asyncio
async def test_the_capability_check_fails_a_faked_model_with_its_own_reason(context_factory):
    result = await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    assert "gpu_metrics" not in str(result.updates)


@pytest.mark.asyncio
async def test_the_capability_check_passes_the_real_model(context_factory):
    result = await _run(context_factory, "NVIDIA H100 80GB HBM3", "9.0", 132)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_OK.reason
