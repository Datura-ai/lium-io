import pytest

from neurons.validators.src.services.task.checks.gpu_model_valid import GpuModelValidCheck

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "gpu_model_rates,state_kwargs,expected_pass,expected_reason",
    [
        (None, {"specs": {}, "gpu_count": 1, "gpu_details": [{"name": "NVIDIA RTX 3090"}]}, False, "GPU_MODEL_POLICY_MISSING"),
        (
            {"NVIDIA RTX 3090": 0.05},
            {"specs": {"gpu": {"count": 1, "details": [{"name": "Unsupported"}]}}, "gpu_count": 1, "gpu_details": [{"name": "Unsupported"}]},
            False,
            "GPU_MODEL_UNSUPPORTED",
        ),
        (
            {None: 1.0, "NVIDIA RTX 3090": 0.05},
            {"specs": {"gpu": {"count": 0, "details": []}}, "gpu_count": 0, "gpu_details": []},
            False,
            "GPU_COUNT_ZERO",
        ),
        (
            {"NVIDIA RTX 3090": 0.05},
            {"specs": {"gpu": {"count": 2, "details": [{"name": "NVIDIA RTX 3090"}]}}, "gpu_count": 2, "gpu_details": [{"name": "NVIDIA RTX 3090"}]},
            False,
            "GPU_DETAILS_MISMATCH",
        ),
        (
            {"NVIDIA RTX 3090": 0.05},
            {"specs": {"gpu": {"count": 1, "details": [{"name": "NVIDIA RTX 3090"}]}}, "gpu_count": 1, "gpu_details": [{"name": "NVIDIA RTX 3090"}]},
            True,
            "GPU_MODEL_OK",
        ),
    ],
)
@pytest.mark.asyncio
async def test_gpu_model_valid_check(gpu_model_rates, state_kwargs, expected_pass, expected_reason, context_factory):
    services = build_services()
    config = build_context_config(gpu_model_rates=gpu_model_rates)
    state = build_state(**state_kwargs)

    ctx = context_factory(services=services, config=config, state=state)

    result = await GpuModelValidCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
