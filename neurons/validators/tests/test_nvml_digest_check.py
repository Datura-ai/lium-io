import pytest

from neurons.validators.src.services.task.checks.nvml_digest import NvmlDigestCheck

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "driver_version,lib_digest,nvml_digest_map,expected_pass,expected_reason,expect_clear",
    [
        # No driver version - should pass
        (
            "",
            "abc123",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            True,
            "NVML_DIGEST_OK",
            False,
        ),
        # Matching digest - should pass
        (
            "535.183.01",
            "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            True,
            "NVML_DIGEST_OK",
            False,
        ),
        # Mismatched digest - should fail
        (
            "535.183.01",
            "wrong_digest_here",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            "NVML_DIGEST_MISMATCH",
            True,
        ),
        # Driver version not in map - should fail
        (
            "999.999.99",
            "any_digest",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            "NVML_DIGEST_MISMATCH",
            True,
        ),
        # Empty digest - should fail if driver version is known
        (
            "535.183.01",
            "",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            "NVML_DIGEST_MISMATCH",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_nvml_digest_check(
    driver_version,
    lib_digest,
    nvml_digest_map,
    expected_pass,
    expected_reason,
    expect_clear,
    context_factory,
):
    services = build_services()
    config = build_context_config(nvml_digest_map=nvml_digest_map)
    specs = {
        "gpu": {"driver": driver_version},
        "md5_checksums": {"libnvidia_ml": lib_digest},
    }
    state = build_state(specs=specs)

    ctx = context_factory(services=services, config=config, state=state)

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
