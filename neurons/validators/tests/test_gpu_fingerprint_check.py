import pytest

from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "prev_uuids,current_uuids,expected_pass,expected_reason,expect_clear",
    [
        ("", "", True, "GPU_UUID_OK", False),
        ("", "gpu-001", True, "GPU_UUID_OK", False),
        ("gpu-001", "", True, "GPU_UUID_OK", False),
        ("gpu-001,gpu-002", "gpu-001,gpu-002", True, "GPU_UUID_OK", False),
        ("gpu-002,gpu-001", "gpu-001,gpu-002", True, "GPU_UUID_OK", False),  # sorted comparison
        ("gpu-001,gpu-002", "gpu-001,gpu-003", False, "GPU_UUID_CHANGED", True),
        ("gpu-001", "gpu-001,gpu-002", False, "GPU_UUID_CHANGED", True),
    ],
)
@pytest.mark.asyncio
async def test_gpu_fingerprint_check(
    prev_uuids,
    current_uuids,
    expected_pass,
    expected_reason,
    expect_clear,
    context_factory,
):
    services = build_services()
    config = build_context_config()
    state = build_state(gpu_uuids=current_uuids)

    verified = {"uuids": prev_uuids} if prev_uuids else {}
    ctx = context_factory(services=services, config=config, state=state, verified=verified)

    result = await GpuFingerprintCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
