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
    VerifyXValidationService,
    settings,
)
from tests.test_verifyx_capacity_probe import _challenge_data, _probe_payload

SERVICE = "neurons.validators.src.services.verifyx_validation_service"
REPO = pathlib.Path(__file__).resolve().parents[3]
# celium-gpu-verifier main @ cb94a7f (#28 merged), "verifyx build" run 37360385667 SHA256SUMS
LIBRARY_SHA256 = "c393ae12e7e81dee4d66c190e3bdc361fb238337715b5ff30169161e426841bb"
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
    shell.executor_arch = "x86_64"  # an x86 host whose CPU was already read: no `uname -m` in the command list
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


def _refresh_on():
    return patch.object(settings.verifyx, "LIBRARY_REFRESH_ENABLED", True)


ARM_SHA256 = "e" * 64


def arm_shell(digests: dict[str, str]) -> MagicMock:
    shell = executor_shell(digests)
    shell.executor_arch = None
    shell.ssh_client.run = AsyncMock(
        side_effect=lambda cmd, **_kw: SimpleNamespace(
            stdout="aarch64\n" if cmd == "uname -m" else "ab" * MIN_CIPHER_LEN, stderr="", exit_status=0
        )
    )
    return shell


@pytest.mark.asyncio
async def test_arm_executor_without_an_aarch64_build_fails_clearly_and_uploads_nothing(libraries):
    shell = arm_shell({LIB_PATH: ARM_SHA256})
    put = AsyncMock(return_value=None)
    with _refresh_on(), patch.object(VerifyXValidationService, "_put_validator_library", put):
        result = await gated_run(shell)
    assert result.error.startswith("ARM_VERIFIER_NOT_AVAILABLE")
    assert result.diagnostics == {"event": "ARM_VERIFIER_NOT_AVAILABLE", "executor_arch": "aarch64"}
    put.assert_not_awaited()
    assert commands(shell) == ["uname -m"]


@pytest.mark.asyncio
async def test_arm_executor_is_checked_against_and_sent_the_aarch64_build(libraries, tmp_path):
    arm_lib = tmp_path / "libverifyx-aarch64.so"
    arm_lib.write_bytes(b"arm")
    digests = {LIB_PATH: STALE_SHA256}
    sources = []

    async def after_refresh(_shell, expected, _extra, source):
        sources.append((source, expected))
        digests[LIB_PATH] = expected
        return True

    shell = arm_shell(digests)
    with (
        _refresh_on(),
        patch(f"{SERVICE}.aarch64_lib_path", return_value=str(arm_lib)),
        patch(f"{SERVICE}.sha256_from_path", side_effect=lambda p: ARM_SHA256 if p == str(arm_lib) else LIBRARY_SHA256),
        patch.object(VerifyXValidationService, "_executor_can_write_lib", AsyncMock(return_value=True)),
        patch.object(VerifyXValidationService, "_refresh_executor_library", side_effect=after_refresh),
    ):
        result = await gated_run(shell)
    assert sources == [(str(arm_lib), ARM_SHA256)]
    assert result.error is None
    assert [record.lib_name for record in libraries.built] == [LIB_PATH]  # the validator still runs its own build


@pytest.mark.asyncio
async def test_refresh_of_the_aarch64_build_fetches_and_puts_that_file():
    shell = executor_shell({LIB_PATH: STALE_SHA256})
    shell.ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(stdout="CURL_RC:22\n", stderr="404", exit_status=0)
    )
    put = AsyncMock(return_value=None)
    with patch.object(VerifyXValidationService, "_put_validator_library", put):
        ok = await VerifyXValidationService()._refresh_executor_library(
            shell, ARM_SHA256, {"executor_uuid": "exec-1"}, source="/usr/lib/libverifyx-aarch64.so"
        )
    assert ok is True
    assert "/neurons/executor/libverifyx-aarch64.so" in commands(shell)[0]
    put.assert_awaited_once_with(shell, "/usr/lib/libverifyx-aarch64.so")


@pytest.mark.parametrize(
    ("arch", "has_arm_build", "expected"),
    [
        ("aarch64", True, ARM_SHA256),
        ("aarch64", False, LIBRARY_SHA256),
        ("x86_64", True, LIBRARY_SHA256),
        (None, True, LIBRARY_SHA256),
    ],
)
def test_local_verify_expects_the_library_the_executor_holds(libraries, tmp_path, arch, has_arm_build, expected):
    """`/verify` and the SSH checksum gate accept the same library for one architecture, while the
    challenge itself is still generated by the validator's native build."""
    arm_lib = str(tmp_path / "libverifyx-aarch64.so")
    spec = {**MACHINE_SPEC, "machine_arch": arch} if arch else MACHINE_SPEC
    with (
        patch(f"{SERVICE}.aarch64_lib_path", return_value=arm_lib if has_arm_build else None),
        patch(f"{SERVICE}.sha256_from_path", side_effect=lambda p: ARM_SHA256 if p == arm_lib else LIBRARY_SHA256),
    ):
        challenge = VerifyXValidationService().prepare_verifyx_challenge(spec, {})
    assert challenge.expected_lib_sha256 == expected
    assert [record.lib_name for record in libraries.built] == [LIB_PATH]


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_digest", [STALE_SHA256, LIBRARY_SHA256])
async def test_an_unreadable_cpu_never_writes_a_library(libraries, executor_digest):
    """A failed `uname -m` is not taken for x86: a stale library stays put (no x86 build on a possible ARM host), and a
    library that already matches the native build still runs."""
    shell = executor_shell({LIB_PATH: executor_digest})
    shell.executor_arch = None
    refresh = AsyncMock(return_value=True)
    with _refresh_on(), patch.object(VerifyXValidationService, "_refresh_executor_library", refresh):
        result = await gated_run(shell)
    refresh.assert_not_awaited()
    if executor_digest == STALE_SHA256:
        assert result.error.startswith("EXECUTOR_ARCH_UNKNOWN")
    else:
        assert result.error is None


@pytest.mark.asyncio
async def test_an_arm_host_holding_the_x86_build_is_caught_and_sent_the_aarch64_build(libraries, tmp_path):
    """The CPU is read before the hashes: the x86 build on an ARM host matches the native digest and must not pass."""
    arm_lib = tmp_path / "libverifyx-aarch64.so"
    arm_lib.write_bytes(b"arm")
    sources = []

    async def after_refresh(_shell, expected, _extra, source):
        sources.append(source)
        return True

    with (
        _refresh_on(),
        patch(f"{SERVICE}.aarch64_lib_path", return_value=str(arm_lib)),
        patch(f"{SERVICE}.sha256_from_path", side_effect=lambda p: ARM_SHA256 if p == str(arm_lib) else LIBRARY_SHA256),
        patch.object(VerifyXValidationService, "_executor_can_write_lib", AsyncMock(return_value=True)),
        patch.object(VerifyXValidationService, "_refresh_executor_library", side_effect=after_refresh),
    ):
        await gated_run(arm_shell({LIB_PATH: LIBRARY_SHA256}))
    assert sources == [str(arm_lib)]
