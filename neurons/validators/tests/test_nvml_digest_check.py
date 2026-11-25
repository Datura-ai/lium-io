import hashlib
import logging
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from neurons.validators.src.services.task.checks import nvml_digest
from neurons.validators.src.services.task.checks.nvml_digest import NvmlDigestCheck
from neurons.validators.src.services.task.messages import NvmlDigestMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


@pytest.mark.parametrize(
    "driver_version,lib_digest,nvml_digest_map,expected_pass,expected_reason,expect_clear",
    [
        # Matching digest - should pass
        (
            "535.183.01",
            "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            True,
            Msg.DIGEST_OK.reason,
            False,
        ),
        # Mismatched digest with no second read (no SSH): fails, but one read is no proof, so no reset (DAH-7050)
        (
            "535.183.01",
            "wrong_digest_here",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            Msg.DIGEST_UNCONFIRMED.reason,
            False,
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
        # Empty digest - should fail if driver version is known (reset only once a second read confirms it)
        (
            "535.183.01",
            "",
            {"535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"},
            False,
            Msg.DIGEST_UNCONFIRMED.reason,
            False,
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
    # DAH-2742: a confirmed spoof keeps the instant reset that an unrecognized-but-innocent
    # driver no longer gets.
    assert result.updates["clear_verified_job_info"] is True


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
async def test_nvml_digest_check_allows_driver_595_91_07(context_factory):
    # The Deep Learning Base OSS NVIDIA Driver AMI (AWS g5/g6) ships this driver (DAH-3339).
    specs = {
        "gpu": {"driver": "595.91.07"},
        "md5_checksums": {
            "libnvidia_ml": "815eeecaf87fd8f947c66b2ef1ca7525:7515da5b856b805fc07811dfd72a37545c1bd9e78f4d8c16421e155ac8f4aec4",
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


@pytest.mark.asyncio
async def test_a_missing_driver_is_logged_and_fails(context_factory, caplog):
    specs = {"gpu": {"driver": ""}, "md5_checksums": {"libnvidia_ml": "abc123"}}
    ctx = context_factory(state=build_state(specs=specs))

    with caplog.at_level(logging.WARNING, logger=nvml_digest.logger.name):
        result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DRIVER_MISSING.reason
    assert any("NVML_DRIVER_MISSING" in record.getMessage() for record in caplog.records)


X86_580_95_05 = "abb3628879c4801b3e5f5e3351d01c96:9933cfe943ca96d9b3c352221154dd0afd5ed5406fcb9a3d15935ff4f2c380a9"
AARCH64_580_95_05 = "2a216a28f68e552b436a334b4bb26c91:c99a74c7399c6ff8fd0b446d6d08dded76bc858d413f6beedd01212a4b4b2f6c"


@pytest.mark.parametrize(
    "arch,lib_digest,expected_pass,expected_reason",
    [
        # DAH-5053: a GB300 (Grace) host carries NVIDIA's aarch64 libnvidia-ml
        ("aarch64", AARCH64_580_95_05, True, Msg.DIGEST_OK.reason),
        ("aarch64", X86_580_95_05, False, Msg.DIGEST_MISMATCH.reason),
        ("x86_64", X86_580_95_05, True, Msg.DIGEST_OK.reason),
        ("x86_64", AARCH64_580_95_05, False, Msg.DIGEST_MISMATCH.reason),
        # hosts on an older scrape report no arch and keep the x86_64 list
        (None, X86_580_95_05, True, Msg.DIGEST_OK.reason),
    ],
)
@pytest.mark.asyncio
async def test_nvml_digest_check_uses_the_host_arch_list(
    arch, lib_digest, expected_pass, expected_reason, context_factory
):
    specs = {
        "gpu": {"driver": "580.95.05"},
        "md5_checksums": {"libnvidia_ml": lib_digest},
        "machine_arch": arch,
    }
    # the shared-config map is x86_64 only and must not apply to an aarch64 host
    config = build_context_config(nvml_digest_map={"580.95.05": X86_580_95_05})
    # the second read of the same file returns the same digest
    ctx = context_factory(config=config, state=build_state(specs=specs), ssh=_ssh(lib_digest + "\n"))

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason


@pytest.mark.asyncio
async def test_unknown_aarch64_driver_is_not_reported_to_the_x86_resolver(context_factory):
    services = build_services()
    specs = {
        "gpu": {"driver": "999.99"},
        "md5_checksums": {"libnvidia_ml": "abc"},
        "machine_arch": "aarch64",
    }
    ctx = context_factory(services=services, state=build_state(specs=specs))

    result = await NvmlDigestCheck().run(ctx)

    assert result.event.reason_code == Msg.DRIVER_UNKNOWN.reason
    services.backend.report_unknown_driver.assert_not_awaited()


KNOWN = "535.183.01"
GOOD = "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"
OTHER = "e3795cab4c63f3974e2c5c3f98539680:0d5d550ebfae598379af36c4a49c1e3f63f44d0ab1efa69d6a205d7909ee0856"
ZEROS = "00000000000000000000000000000000:" + "0" * 64


def _ssh(stdout="", exit_status=0, raises=None):
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=raises) if raises else AsyncMock(
        return_value=MagicMock(exit_status=exit_status, stdout=stdout, stderr="")
    )
    return ssh


def _mismatch_ctx(context_factory, ssh, reported=ZEROS):
    config = build_context_config(nvml_digest_map={KNOWN: GOOD})
    state = build_state(specs={"gpu": {"driver": KNOWN}, "md5_checksums": {"libnvidia_ml": reported}})
    return context_factory(config=config, state=state, ssh=ssh)


@pytest.mark.asyncio
async def test_a_mismatch_the_second_read_confirms_resets_the_node(context_factory):
    ctx = _mismatch_ctx(context_factory, _ssh(OTHER + "\n"), reported=OTHER)

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.updates["clear_verified_job_info"] is True
    assert result.event.what_we_saw["second_read_md5"] == OTHER


@pytest.mark.asyncio
async def test_one_bad_read_the_second_read_contradicts_keeps_the_verification(context_factory):
    # DAH-7050: node 1778f7ba, 23 Nov: one wrong digest, intact file, the next cycle passed
    ctx = _mismatch_ctx(context_factory, _ssh(GOOD + "\n"))

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_UNCONFIRMED.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh",
    [
        _ssh(raises=TimeoutError()),
        _ssh(raises=OSError("connection lost")),
        _ssh(stdout="", exit_status=1),  # no library found
        _ssh(stdout="d41d8cd98f00b204e9800998ecf8427e\n"),  # not "<md5>:<sha256>"
    ],
    ids=["timeout", "ssh error", "no library", "partial output"],
)
async def test_an_unreadable_second_read_is_unknown_never_a_fault(context_factory, ssh):
    result = await NvmlDigestCheck().run(_mismatch_ctx(context_factory, ssh))

    assert result.event.reason_code == Msg.DIGEST_UNCONFIRMED.reason
    assert "clear_verified_job_info" not in result.updates
    assert result.event.what_we_saw["second_read_md5"] is None


def test_the_second_read_hashes_the_library_the_scrape_picks(tmp_path):
    usr = tmp_path / "usr"
    (usr / "local" / "cuda" / "compat").mkdir(parents=True)
    (usr / "local" / "cuda" / "compat" / "libnvidia-ml.so.1").write_bytes(b"not this one")
    lib_dir = usr / "lib" / "x86_64-linux-gnu"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libnvidia-ml.so.595.84").write_bytes(b"the real library")
    (lib_dir / "libnvidia-ml.so.1").symlink_to("libnvidia-ml.so.595.84")
    cmd = nvml_digest.LIBNVIDIA_ML_DIGEST_CMD.replace("find /usr ", f"find {usr} ")

    out = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, check=True).stdout.strip()

    data = b"the real library"
    assert out == f"{hashlib.md5(data).hexdigest()}:{hashlib.sha256(data).hexdigest()}"
