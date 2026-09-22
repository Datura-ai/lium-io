"""DAH-2774: a libverifyx.so bump accepts the library it replaces for a transition window.

The SSH path fails VerifyX (`OUTDATED_LIBRARY_ERROR`) for any executor whose /usr/lib/libverifyx.so
sha256 differs from the validator's, and the local `/verify` path falls back to SSH on the same
mismatch. Executors reach a new image only as fast as their updater (EXECUTOR_IMAGE_CHECK_ENFORCE
is off because Watchtower stalled), so without a window every executor that has not pulled the
new image fails a fatal check the moment the validator ships the new library.

What each test pins:
- the accepted set is {current, previous} until VERIFYX_PREVIOUS_LIB_ACCEPTED_UNTIL, {current} after;
- the defaults name lium-io main's library and the two committed copies are one build;
- an executor on the previous library passes the gate and is measured as main's validator measured
  it: the package download, never that library's unchecked `speedtest.download_mbps`;
- a digest outside the set still fails with OUTDATED_LIBRARY_ERROR before any SSH run.
"""

from __future__ import annotations

import hashlib
import pathlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.services.verifyx_validation_service import (
    MIN_CIPHER_LEN,
    OUTDATED_LIBRARY_ERROR,
    VerifyXValidationService,
    settings,
)

SERVICE = "neurons.validators.src.services.verifyx_validation_service"
REPO = pathlib.Path(__file__).resolve().parents[3]
MAIN_LIBRARY_SHA256 = "16b9a5012f8e6b4438fbedfe722b9c30de9e2f2e98373aed33094c6ff6be564f"

CURRENT = hashlib.sha256(b"current build").hexdigest()
PREVIOUS = hashlib.sha256(b"previous build").hexdigest()
UNTIL = datetime(2026, 10, 13, tzinfo=UTC)

PACKAGE = {"pkg": "distilbert-base-uncased.tar", "size": 268_000_000, "hash": "sha256:abc"}
PACKAGE_MBPS = 420.0
# main's library divides its requested 100 MB by the elapsed time whatever came back: a 403 in
# 17 ms reads 48 000 Mbps
UNCHECKED_SPEEDTEST_MBPS = 48_000.0


def _payload() -> dict:
    return {
        "challenge_data": {
            "network_challenge": {"download": dict(PACKAGE), "timeout_seconds": 120}
        },
        "response_data": {
            "network_execution": {
                "speedtest": {"download_mbps": UNCHECKED_SPEEDTEST_MBPS, "upload_mbps": 310.0},
                "download": {
                    **PACKAGE,
                    "status": "success",
                    "speed_mbps": PACKAGE_MBPS,
                    "time_ms": 5_100,
                    "error": None,
                },
                "success": True,
                "error": "",
                "execution_time_ms": 9_800,
            }
        },
    }


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(settings.verifyx, "PREVIOUS_LIB_SHA256", PREVIOUS)
    monkeypatch.setattr(settings.verifyx, "PREVIOUS_LIB_ACCEPTED_UNTIL", UNTIL)


def _service() -> VerifyXValidationService:
    service = VerifyXValidationService()
    service._lib_sha256 = CURRENT
    return service


def test_the_previous_library_is_accepted_until_the_window_closes(window, monkeypatch):
    service = _service()

    assert service.accepted_lib_sha256s(UNTIL - timedelta(seconds=1)) == {CURRENT, PREVIOUS}
    assert service.accepted_lib_sha256s(UNTIL) == {CURRENT}
    assert service.accepted_lib_sha256s(UNTIL + timedelta(days=30)) == {CURRENT}

    # an env value without an offset is read as UTC
    monkeypatch.setattr(settings.verifyx, "PREVIOUS_LIB_ACCEPTED_UNTIL", UNTIL.replace(tzinfo=None))
    assert service.accepted_lib_sha256s(UNTIL - timedelta(seconds=1)) == {CURRENT, PREVIOUS}

    # "" closes the window; a previous digest equal to the current one adds nothing
    monkeypatch.setattr(settings.verifyx, "PREVIOUS_LIB_SHA256", "")
    assert service.accepted_lib_sha256s(UNTIL - timedelta(days=1)) == {CURRENT}
    monkeypatch.setattr(settings.verifyx, "PREVIOUS_LIB_SHA256", CURRENT)
    assert service.accepted_lib_sha256s(UNTIL - timedelta(days=1)) == {CURRENT}


def test_the_defaults_name_mains_library_and_the_committed_copies_are_one_build():
    validator_lib = hashlib.sha256(
        (REPO / "neurons/validators/libverifyx.so").read_bytes()
    ).hexdigest()
    executor_lib = hashlib.sha256(
        (REPO / "neurons/executor/libverifyx.so").read_bytes()
    ).hexdigest()

    assert validator_lib == executor_lib
    assert settings.verifyx.PREVIOUS_LIB_SHA256 == MAIN_LIBRARY_SHA256 != validator_lib
    assert settings.verifyx.PREVIOUS_LIB_ACCEPTED_UNTIL == UNTIL


async def _run(service: VerifyXValidationService, executor_digest: str):
    shell = MagicMock()
    shell.get_sha256_checksum_by_path = AsyncMock(return_value=executor_digest)
    shell.ssh_client = SimpleNamespace(
        run=AsyncMock(
            return_value=SimpleNamespace(stdout="ab" * MIN_CIPHER_LEN, stderr="", exit_status=0)
        )
    )
    with (
        patch(f"{SERVICE}.VerifyXValidator") as fake_validator_cls,
        patch(f"{SERVICE}._verify_memory_test", return_value=({"success": True}, [])),
        patch(f"{SERVICE}._verify_storage_test", return_value=({"success": True}, [])),
    ):
        fake_validator_cls.return_value.generate_challenge.return_value = "cd" * MIN_CIPHER_LEN
        fake_validator_cls.return_value.verify_response.return_value = _payload()
        response = await service.validate_verifyx_and_process_job(
            shell=shell,
            executor_info=SimpleNamespace(
                python_path="/usr/bin/python3", root_dir="/root/app", uuid="exec-1"
            ),
            default_extra={"executor_uuid": "exec-1"},
            machine_spec={"gpu": {"count": 1, "details": [{"uuid": "u", "name": "H100"}]}},
        )
    return response, shell.ssh_client.run


@pytest.mark.asyncio
async def test_an_executor_on_the_previous_library_passes_and_is_measured_by_its_package_download(
    window,
):
    with patch(f"{SERVICE}.datetime") as clock:
        clock.now.return_value = UNTIL - timedelta(days=1)
        previous, ran = await _run(_service(), PREVIOUS)
        current, _ = await _run(_service(), CURRENT)

    assert previous.error is None and ran.await_count == 1
    assert previous.data["verifyx_library"] == "previous"
    assert previous.data["network"]["download_speed"] == PACKAGE_MBPS
    assert previous.data["network"]["package_download_speed"] == PACKAGE_MBPS
    # the same answer from the current library is read as the capacity probe it is
    assert "verifyx_library" not in current.data
    assert current.data["network"]["download_speed"] == UNCHECKED_SPEEDTEST_MBPS


@pytest.mark.asyncio
async def test_after_the_window_or_for_any_other_digest_the_gate_refuses_before_ssh(window):
    with patch(f"{SERVICE}.datetime") as clock:
        clock.now.return_value = UNTIL
        closed, ran_closed = await _run(_service(), PREVIOUS)
        clock.now.return_value = UNTIL - timedelta(days=1)
        stranger, ran_stranger = await _run(_service(), hashlib.sha256(b"tampered").hexdigest())

    assert closed.error == OUTDATED_LIBRARY_ERROR and ran_closed.await_count == 0
    assert stranger.error == OUTDATED_LIBRARY_ERROR and ran_stranger.await_count == 0
