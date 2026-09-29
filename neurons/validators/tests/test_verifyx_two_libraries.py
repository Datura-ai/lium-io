"""DAH-2774: libverifyx.so is the Cloudflare capacity build.

Library refresh is off by default. When VERIFYX_LIBRARY_REFRESH_ENABLED is on, a mismatch
checks that /usr/lib is writable, then curls the raw GitHub URL, installs, and retries
once. A read-only root does not take the fatal outdated path after a failed write.
"""

from __future__ import annotations

import copy
import hashlib
import pathlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.services.verifyx_validation_service import (
    LIB_PATH,
    MIN_CIPHER_LEN,
    OUTDATED_LIBRARY_ERROR,
    VerifyXValidationService,
    settings,
)
from tests.test_verifyx_capacity_probe import _challenge_data, _probe_payload

SERVICE = "neurons.validators.src.services.verifyx_validation_service"
REPO = pathlib.Path(__file__).resolve().parents[3]
# celium-gpu-verifier main @ dd0f994 (#25 merged), "verifyx build" run 35977025920 SHA256SUMS
LIBRARY_SHA256 = "c44146556cd0a415c6f6da888fdd6597d6aa3cfa342dfc9b66d4797cb216e08a"
STALE_SHA256 = "16b9a5012f8e6b4438fbedfe722b9c30de9e2f2e98373aed33094c6ff6be564f"


class Libraries:
    """Stands in for the native library: the answer it gives, and every challenge built."""

    def __init__(self, answers: dict[str, dict]):
        self.answers = answers
        self.built: list[SimpleNamespace] = []
        library = self

        class FakeValidator:
            def __init__(self, lib_name, seed):
                self.record = SimpleNamespace(lib_name=lib_name, seed=seed, config=None)
                library.built.append(self.record)

            def generate_challenge(self, challenge_input):
                self.record.config = challenge_input["config"]
                return "cd" * MIN_CIPHER_LEN

            def verify_response(self, response):
                return {
                    "challenge_data": _challenge_data(),
                    "response_data": {
                        "network_execution": copy.deepcopy(library.answers[self.record.lib_name])
                    },
                }

        self.validator_cls = FakeValidator


def executor_shell(digests: dict[str, str]) -> MagicMock:
    shell = MagicMock()
    shell.get_sha256_checksum_by_path = AsyncMock(side_effect=lambda path: digests.get(path, ""))
    shell.ssh_client = SimpleNamespace(
        run=AsyncMock(
            return_value=SimpleNamespace(stdout="ab" * MIN_CIPHER_LEN, stderr="", exit_status=0)
        )
    )
    return shell


def asked_for(shell: MagicMock) -> list[str]:
    return [call.args[0] for call in shell.get_sha256_checksum_by_path.await_args_list]


def commands(shell: MagicMock) -> list[str]:
    return [call.args[0] for call in shell.ssh_client.run.await_args_list]


@contextmanager
def patched_libraries():
    library = Libraries({LIB_PATH: _probe_payload()})
    with (
        patch(f"{SERVICE}.VerifyXValidator", library.validator_cls),
        patch(f"{SERVICE}.sha256_from_path", return_value=LIBRARY_SHA256),
        patch(f"{SERVICE}._verify_memory_test", return_value=({"success": True}, [])),
        patch(f"{SERVICE}._verify_storage_test", return_value=({"success": True}, [])),
    ):
        yield library


@pytest.fixture
def libraries():
    with patched_libraries() as library:
        yield library


EXECUTOR_INFO = SimpleNamespace(python_path="/usr/bin/python3", root_dir="/root/app", uuid="exec-1")
MACHINE_SPEC = {"gpu": {"count": 1, "details": [{"uuid": "u", "name": "H100"}]}}
MAIN_COMMAND_PREFIX = "/usr/bin/python3 /root/app/src/verifyx_executor.py --seed "


async def gated_run(shell):
    return await VerifyXValidationService().validate_verifyx_and_process_job(
        shell=shell,
        executor_info=EXECUTOR_INFO,
        default_extra={"executor_uuid": "exec-1"},
        machine_spec=MACHINE_SPEC,
    )


def _sha256(relative: str) -> str:
    return hashlib.sha256((REPO / relative).read_bytes()).hexdigest()


def test_one_libverifyx_so_in_both_images_and_no_capacity_library():
    assert _sha256("neurons/validators/libverifyx.so") == LIBRARY_SHA256
    assert _sha256("neurons/executor/libverifyx.so") == LIBRARY_SHA256
    assert not (REPO / "neurons/validators/libverifyx_capacity.so").exists()
    assert not (REPO / "neurons/executor/libverifyx_capacity.so").exists()
    assert LIB_PATH == "/usr/lib/libverifyx.so"
    validator_dockerfile = (REPO / "neurons/validators/Dockerfile").read_text()
    executor_dockerfile = (REPO / "neurons/executor/Dockerfile").read_text()
    assert "RUN mv libverifyx.so /usr/lib/" in validator_dockerfile
    assert "libverifyx_capacity.so" not in validator_dockerfile
    assert "mv /root/app/libverifyx.so /usr/lib/" in executor_dockerfile
    assert "libverifyx_capacity.so" not in executor_dockerfile
    env_template = (REPO / "neurons/validators/.env.template").read_text()
    assert "VERIFYX_NETWORK_GATE_MODE" not in env_template


@pytest.mark.asyncio
async def test_matching_hash_runs_libverifyx_so_with_no_lib_flag(libraries):
    shell = executor_shell({LIB_PATH: LIBRARY_SHA256})
    result = await gated_run(shell)
    assert result.error is None and result.data["success"] is True
    assert asked_for(shell) == [LIB_PATH]
    command = commands(shell)[0]
    assert command.startswith(MAIN_COMMAND_PREFIX)
    assert "--lib" not in command
    assert [record.lib_name for record in libraries.built] == [LIB_PATH]


def _refresh_on():
    return patch.object(settings.verifyx, "LIBRARY_REFRESH_ENABLED", True)


@pytest.mark.asyncio
async def test_hash_mismatch_does_not_refresh_when_opt_in_is_off(libraries):
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    with patch.object(
        VerifyXValidationService, "_refresh_executor_library", AsyncMock(return_value=True)
    ) as refresh:
        result = await gated_run(shell)
    assert refresh.await_count == 0
    assert result.error == OUTDATED_LIBRARY_ERROR
    assert (result.diagnostics or {}).get("event") == "VERIFYX_LIBRARY_MISMATCH_NO_REFRESH"
    assert commands(shell) == []


@pytest.mark.asyncio
async def test_hash_mismatch_does_not_refresh_when_usr_lib_is_not_writable(libraries):
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    with (
        _refresh_on(),
        patch.object(
            VerifyXValidationService, "_executor_can_write_lib", AsyncMock(return_value=False)
        ) as writable,
        patch.object(
            VerifyXValidationService, "_refresh_executor_library", AsyncMock(return_value=True)
        ) as refresh,
    ):
        result = await gated_run(shell)
    assert writable.await_count == 1
    assert refresh.await_count == 0
    assert result.error != OUTDATED_LIBRARY_ERROR
    assert "not writable" in result.error
    assert (result.diagnostics or {}).get("event") == "VERIFYX_LIBRARY_WRITE_DENIED"


@pytest.mark.asyncio
async def test_hash_mismatch_fetches_then_retries(libraries):
    digests = {LIB_PATH: STALE_SHA256}

    async def after_refresh(_shell, expected, _extra):
        digests[LIB_PATH] = expected
        return True

    shell = executor_shell(digests)
    with (
        _refresh_on(),
        patch.object(
            VerifyXValidationService, "_executor_can_write_lib", AsyncMock(return_value=True)
        ),
        patch.object(
            VerifyXValidationService, "_refresh_executor_library", side_effect=after_refresh
        ) as refresh,
    ):
        result = await gated_run(shell)
    assert refresh.await_count == 1
    assert result.error is None and result.data["success"] is True
    assert asked_for(shell) == [LIB_PATH, LIB_PATH]


@pytest.mark.asyncio
async def test_hash_mismatch_and_failed_fetch_stays_outdated(libraries):
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    with (
        _refresh_on(),
        patch.object(
            VerifyXValidationService, "_executor_can_write_lib", AsyncMock(return_value=True)
        ),
        patch.object(
            VerifyXValidationService, "_refresh_executor_library", AsyncMock(return_value=False)
        ) as refresh,
    ):
        result = await gated_run(shell)
    assert refresh.await_count == 1
    assert result.error == OUTDATED_LIBRARY_ERROR
    assert commands(shell) == []


@pytest.mark.asyncio
async def test_curl_installs_when_the_fetched_hash_matches():
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    shell.ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(
            stdout=f"CURL_RC:0\n{LIBRARY_SHA256}  /tmp/libverifyx.so.fetch\n",
            stderr="",
            exit_status=0,
        )
    )
    ok = await VerifyXValidationService()._refresh_executor_library(
        shell, LIBRARY_SHA256, {"executor_uuid": "exec-1"}
    )
    assert ok is True
    ran = commands(shell)
    assert any("curl -fsSL" in cmd for cmd in ran)
    assert any(cmd.startswith("mv ") and LIB_PATH in cmd for cmd in ran)


@pytest.mark.asyncio
async def test_executor_can_write_lib_reads_the_probe():
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    shell.ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(stdout="WRITE_OK:1\n", stderr="", exit_status=0)
    )
    assert await VerifyXValidationService()._executor_can_write_lib(shell) is True
    shell.ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(stdout="WRITE_OK:0\n", stderr="", exit_status=0)
    )
    assert await VerifyXValidationService()._executor_can_write_lib(shell) is False


@pytest.mark.asyncio
async def test_curl_error_is_logged_and_validator_file_is_installed():
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    shell.ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(stdout="CURL_RC:22\n", stderr="404", exit_status=0)
    )
    put = AsyncMock(return_value=None)
    with patch.object(VerifyXValidationService, "_put_validator_library", put):
        ok = await VerifyXValidationService()._refresh_executor_library(
            shell, LIBRARY_SHA256, {"executor_uuid": "exec-1"}
        )
    assert ok is True
    put.assert_awaited_once()
