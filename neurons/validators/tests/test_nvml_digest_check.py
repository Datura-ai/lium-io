import hashlib
import logging
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from neurons.validators.src.services.task.checks import nvml_digest
from neurons.validators.src.services.task.checks.nvml_digest import NvmlDigestCheck
from neurons.validators.src.services.task.messages import NvmlDigestMessages as Msg

from tests.helpers import build_context_config, build_services, build_state

KNOWN = "535.183.01"
GOOD = "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f"
OTHER = "e3795cab4c63f3974e2c5c3f98539680:0d5d550ebfae598379af36c4a49c1e3f63f44d0ab1efa69d6a205d7909ee0856"


def _ssh(stdout="", exit_status=0, raises=None):
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=raises) if raises else AsyncMock(
        return_value=MagicMock(exit_status=exit_status, stdout=stdout, stderr="")
    )
    return ssh


def _read(digest, size=1854672):
    """The node's answer to a complete read of a library with this digest."""
    return _ssh(f"DIGEST {digest} {size}\n")


def _ctx(context_factory, driver, ssh, scrape="", digest_map=None, arch=None, services=None, invalid=None):
    specs = {"gpu": {"driver": driver}, "md5_checksums": {"libnvidia_ml": scrape}}
    if arch is not None:
        specs["machine_arch"] = arch
    config = build_context_config(nvml_digest_map=digest_map, nvml_invalid_drivers=invalid or [])
    return context_factory(
        config=config, state=build_state(specs=specs), ssh=ssh, services=services or build_services()
    )


@pytest.mark.asyncio
async def test_a_complete_read_of_the_right_library_passes(context_factory):
    ctx = _ctx(context_factory, KNOWN, _read(GOOD), digest_map={KNOWN: GOOD})

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_OK.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_a_complete_read_of_a_wrong_library_resets_at_once(context_factory):
    ctx = _ctx(context_factory, KNOWN, _read(OTHER), digest_map={KNOWN: GOOD})

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.updates["clear_verified_job_info"] is True
    assert result.event.what_we_saw["actual_md5"] == OTHER
    assert ctx.ssh.run.await_count == 1


@pytest.mark.asyncio
async def test_a_wrong_scrape_digest_no_longer_decides(context_factory):
    # DAH-7050, 23 Nov: one scrape reported a digest that matched no file on an intact host; the read decides
    ctx = _ctx(context_factory, KNOWN, _read(GOOD), scrape=OTHER, digest_map={KNOWN: GOOD})

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["scrape_md5"] == OTHER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh",
    [
        None,  # no session
        _ssh(raises=TimeoutError()),
        _ssh(raises=OSError("connection lost")),
        _ssh(exit_status=3),  # no library found
        _ssh(exit_status=4),  # stat, md5sum or sha256sum failed
        _ssh(exit_status=5),  # the file changed while it was hashed
        _ssh("DIGEST " + GOOD[:40] + "\n"),  # output cut short
        _ssh(f"DIGEST {GOOD} 0\n"),  # an empty file
        _ssh(f"{GOOD}\n"),  # no completeness marker
    ],
    ids=["no ssh", "timeout", "ssh error", "no library", "failed read", "changed mid-read", "cut short", "empty",
         "no marker"],
)
async def test_a_failed_or_partial_read_is_unknown_never_a_fault(context_factory, ssh):
    ctx = _ctx(context_factory, KNOWN, ssh, scrape=OTHER, digest_map={KNOWN: GOOD})

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_UNREAD.reason
    assert "clear_verified_job_info" not in result.updates
    assert result.event.what_we_saw["actual_md5"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "driver,digest",
    [
        ("595.71.05", "020cd1156cbce5ebbf12963d0c70496e:9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5"),
        ("580.167.08", "fa0c084327835d0369e5307a1ba3a882:c7eac74626efce631035360d6a5d1f9d72b02d81739ab0adbf0521f6c6a0f10a"),
        # The Deep Learning Base OSS NVIDIA Driver AMI (AWS g5/g6) ships this driver (DAH-3339).
        ("595.91.07", "815eeecaf87fd8f947c66b2ef1ca7525:7515da5b856b805fc07811dfd72a37545c1bd9e78f4d8c16421e155ac8f4aec4"),
        ("610.43.02", "5ad6c02411f730682597558ae8f3a9f8:2dc828b3f5027f98e05c7607c1d8129d11bd28de4c2091c5cd7e32dbc21ec172"),
    ],
)
async def test_the_built_in_map_allows_these_drivers(context_factory, driver, digest):
    result = await NvmlDigestCheck().run(_ctx(context_factory, driver, _read(digest)))

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_OK.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", ["999.999.99", "535.274.02"])
async def test_an_unknown_driver_fails_without_a_reset_and_without_a_read(context_factory, driver):
    services = build_services()
    ssh = _read(GOOD)
    ctx = _ctx(context_factory, driver, ssh, scrape="abc", digest_map={KNOWN: GOOD}, services=services)

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DRIVER_UNKNOWN.reason
    assert "clear_verified_job_info" not in result.updates  # DAH-2742: unknown driver is not tampering
    services.backend.report_unknown_driver.assert_awaited_once_with(driver)
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_known_spoof_driver_is_not_reported(context_factory):
    # A driver already confirmed invalid is rejected but never re-reported to the backend.
    services = build_services()
    ctx = _ctx(context_factory, "591.86", _read(GOOD), scrape="abc", digest_map={}, services=services,
               invalid=["591.86"])

    result = await NvmlDigestCheck().run(ctx)

    assert result.event.reason_code == Msg.DRIVER_UNKNOWN.reason
    services.backend.report_unknown_driver.assert_not_awaited()
    # DAH-2742: a confirmed spoof keeps the instant reset that an unrecognized-but-innocent driver no longer gets.
    assert result.updates["clear_verified_job_info"] is True


@pytest.mark.asyncio
async def test_a_missing_driver_is_logged_and_fails(context_factory, caplog):
    ctx = _ctx(context_factory, "", _read(GOOD), scrape="abc123")

    with caplog.at_level(logging.WARNING, logger=nvml_digest.logger.name):
        result = await NvmlDigestCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DRIVER_MISSING.reason
    assert any("NVML_DRIVER_MISSING" in record.getMessage() for record in caplog.records)


X86_580_95_05 = "abb3628879c4801b3e5f5e3351d01c96:9933cfe943ca96d9b3c352221154dd0afd5ed5406fcb9a3d15935ff4f2c380a9"
AARCH64_580_95_05 = "2a216a28f68e552b436a334b4bb26c91:c99a74c7399c6ff8fd0b446d6d08dded76bc858d413f6beedd01212a4b4b2f6c"


@pytest.mark.parametrize(
    "arch,read_digest,expected_pass,expected_reason",
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
    arch, read_digest, expected_pass, expected_reason, context_factory
):
    # the shared-config map is x86_64 only and must not apply to an aarch64 host
    ctx = _ctx(context_factory, "580.95.05", _read(read_digest), digest_map={"580.95.05": X86_580_95_05}, arch=arch)

    result = await NvmlDigestCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason


@pytest.mark.asyncio
async def test_unknown_aarch64_driver_is_not_reported_to_the_x86_resolver(context_factory):
    services = build_services()
    ctx = _ctx(context_factory, "999.99", _read(GOOD), scrape="abc", arch="aarch64", services=services)

    result = await NvmlDigestCheck().run(ctx)

    assert result.event.reason_code == Msg.DRIVER_UNKNOWN.reason
    services.backend.report_unknown_driver.assert_not_awaited()


def _fake_usr(tmp_path):
    usr = tmp_path / "usr"
    (usr / "local" / "cuda" / "compat").mkdir(parents=True)
    (usr / "local" / "cuda" / "compat" / "libnvidia-ml.so.1").write_bytes(b"not this one")
    lib_dir = usr / "lib" / "x86_64-linux-gnu"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libnvidia-ml.so.595.84").write_bytes(b"the real library")
    (lib_dir / "libnvidia-ml.so.1").symlink_to("libnvidia-ml.so.595.84")
    return usr, lib_dir / "libnvidia-ml.so.595.84"


def _run_read(usr, cmd=None):
    cmd = (cmd or nvml_digest.LIBNVIDIA_ML_DIGEST_CMD).replace("find /usr ", f"find {usr} ")
    return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)


def test_the_read_hashes_the_library_the_scrape_picks_and_marks_it_complete(tmp_path):
    usr, _ = _fake_usr(tmp_path)

    out = _run_read(usr)

    data = b"the real library"
    assert out.returncode == 0
    assert out.stdout.strip() == f"DIGEST {hashlib.md5(data).hexdigest()}:{hashlib.sha256(data).hexdigest()} {len(data)}"


def test_a_file_replaced_while_it_is_hashed_prints_no_digest(tmp_path):
    usr, lib = _fake_usr(tmp_path)
    # stands in for a driver update landing between the hash and the second stat
    cmd = nvml_digest.LIBNVIDIA_ML_DIGEST_CMD.replace(
        "&& b=$(stat", f"&& {{ rm -f {lib}; printf 'new driver' > {lib}; }} && b=$(stat"
    )

    out = _run_read(usr, cmd)

    assert out.returncode == 5
    assert "DIGEST" not in out.stdout


def test_an_unreadable_library_prints_no_digest(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    usr, lib = _fake_usr(tmp_path)
    lib.chmod(0)

    out = _run_read(usr)

    assert out.returncode == 4
    assert "DIGEST" not in out.stdout


def test_no_library_prints_no_digest(tmp_path):
    (tmp_path / "usr").mkdir()

    out = _run_read(tmp_path / "usr")

    assert out.returncode == 3
    assert out.stdout == ""
