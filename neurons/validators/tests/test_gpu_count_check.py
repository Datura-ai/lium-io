import pytest

from neurons.validators.src.services.task.checks.gpu_count import GpuCountCheck
from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "gpu_count,max_gpu_count,expected_pass,expected_reason",
    [
        (2, 3, True, "GPU_COUNT_OK"),
        (5, None, False, "GPU_COUNT_POLICY_MISSING"),
        (5, 4, False, "GPU_COUNT_EXCEEDS_MAX"),
    ],
)
@pytest.mark.asyncio
async def test_gpu_count_check(gpu_count, max_gpu_count, expected_pass, expected_reason, context_factory):
    services = build_services()
    config = build_context_config(max_gpu_count=max_gpu_count)
    specs = {"gpu": {"count": gpu_count or 0}}
    state = build_state(specs=specs, gpu_count=gpu_count)

    ctx = context_factory(services=services, config=config, state=state)
    result = await GpuCountCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
