import pytest

from neurons.validators.src.services.task.checks.nvml_digest import NvmlDigestCheck
from neurons.validators.src.services.task.messages import NvmlDigestMessages as Msg

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
            Msg.DIGEST_OK.reason,
            False,
        ),
        # Matching digest - should pass
        (
            "535.183.01",
            "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            True,
            Msg.DIGEST_OK.reason,
            False,
        ),
        # Mismatched digest - should fail
        (
            "535.183.01",
            "wrong_digest_here",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            Msg.DIGEST_MISMATCH.reason,
            True,
        ),
        # Driver version not in map - should fail with DRIVER_UNKNOWN
        (
            "999.999.99",
            "any_digest",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            Msg.DRIVER_UNKNOWN.reason,
            False,  # DAH-2742: unknown driver is not tampering
        ),
        # Real-world unknown driver case - 535.274.02
        (
            "535.274.02",
            "939800fdf0d88c143e416203d68a7d39:25e82746a4eb51597e9e901bc59d5a4e05c5971f8e0069df49c6d4f6cfeb4b51",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            Msg.DRIVER_UNKNOWN.reason,
            False,  # DAH-2742: unknown driver is not tampering
        ),
        # Empty digest - should fail if driver version is known
        (
            "535.183.01",
            "",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            Msg.DIGEST_MISMATCH.reason,
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


@pytest.mark.asyncio
async def test_nvml_digest_check_allows_driver_595_71_05(context_factory):
    specs = {
        "gpu": {"driver": "595.71.05"},
        "md5_checksums": {
            "libnvidia_ml": "020cd1156cbce5ebbf12963d0c70496e:9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5",
        },
    }
    ctx = context_factory(state=build_state(specs=specs))

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_OK.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_unknown_driver_is_reported_to_backend(context_factory):
    # An unknown driver (not in the map, not a known spoof) is reported for verification.
    services = build_services()
    config = build_context_config(nvml_digest_map={}, nvml_invalid_drivers=[])
    state = build_state(specs={"gpu": {"driver": "999.99"}, "md5_checksums": {"libnvidia_ml": "abc"}})
    ctx = context_factory(services=services, config=config, state=state)

    result = await NvmlDigestCheck().run(ctx)

    assert result.event.reason_code == Msg.DRIVER_UNKNOWN.reason
    services.backend.report_unknown_driver.assert_awaited_once_with("999.99")


@pytest.mark.asyncio
async def test_known_spoof_driver_is_not_reported(context_factory):
    # A driver already confirmed invalid is rejected but never re-reported to the backend.
    services = build_services()
    config = build_context_config(nvml_digest_map={}, nvml_invalid_drivers=["591.86"])
    state = build_state(specs={"gpu": {"driver": "591.86"}, "md5_checksums": {"libnvidia_ml": "abc"}})
    ctx = context_factory(services=services, config=config, state=state)

    result = await NvmlDigestCheck().run(ctx)

    assert result.event.reason_code == Msg.DRIVER_UNKNOWN.reason
    services.backend.report_unknown_driver.assert_not_awaited()


@pytest.mark.asyncio
async def test_nvml_digest_check_allows_driver_580_167_08(context_factory):
    specs = {
        "gpu": {"driver": "580.167.08"},
        "md5_checksums": {
            "libnvidia_ml": "fa0c084327835d0369e5307a1ba3a882:c7eac74626efce631035360d6a5d1f9d72b02d81739ab0adbf0521f6c6a0f10a",
        },
    }
    ctx = context_factory(state=build_state(specs=specs))

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_OK.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_nvml_digest_check_allows_driver_610_43_02(context_factory):
    specs = {
        "gpu": {"driver": "610.43.02"},
        "md5_checksums": {
            "libnvidia_ml": "5ad6c02411f730682597558ae8f3a9f8:2dc828b3f5027f98e05c7607c1d8129d11bd28de4c2091c5cd7e32dbc21ec172",
        },
    }
    ctx = context_factory(state=build_state(specs=specs))

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_OK.reason
    assert "clear_verified_job_info" not in result.updates
