"""DAH-2774: both images vendor two VerifyX builds, and executors are asked for nothing new until enforce.

The SSH path fails VerifyX (`OUTDATED_LIBRARY_ERROR`) for any executor whose library sha256 differs
from the validator's, and executors reach a new image only as fast as their updater. So:

- libverifyx.so stays byte-identical to main (same name, same sha256 the validator and every executor
  check today) and is the only library off and shadow gate on: the checksum gate reads the executor's
  /usr/lib/libverifyx.so and the command is main's, with no `--lib`;
- libverifyx_capacity.so (celium-gpu-verifier#25) is committed beside it, one build in both images,
  and both Dockerfiles install it next to libverifyx.so;
- enforce gates on libverifyx_capacity.so: an executor without the validator's build of it fails as
  outdated before any run, and the run names it with `--lib`;
- the shadow run (`measure_capacity_shadow`) asks for libverifyx_capacity.so only, runs only when
  the executor has the validator's build, runs light under the timeout it is given, and every
  failure is a record, never an exception.
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
    CAPACITY_LIB_PATH,
    LIB_PATH,
    MIN_CIPHER_LEN,
    OUTDATED_LIBRARY_ERROR,
    VerifyXValidationService,
    settings,
)
from tests.test_verifyx_capacity_probe import _challenge_data, _probe_payload

SERVICE = "neurons.validators.src.services.verifyx_validation_service"
REPO = pathlib.Path(__file__).resolve().parents[3]
# lium-io main's libverifyx.so: what every executor presents today
MAIN_LIBRARY_SHA256 = "16b9a5012f8e6b4438fbedfe722b9c30de9e2f2e98373aed33094c6ff6be564f"
# celium-gpu-verifier#25 built reproducibly (celium-gpu-verifier#26 prints this digest)
CAPACITY_LIBRARY_SHA256 = "cb3686363514c2d43be699b1572ece53c8e4d44f19c9c79fe560ece1f2926fbe"
VALIDATOR_DIGESTS = {LIB_PATH: MAIN_LIBRARY_SHA256, CAPACITY_LIB_PATH: CAPACITY_LIBRARY_SHA256}
# an executor on today's image: libverifyx.so only
TODAYS_EXECUTOR = {LIB_PATH: MAIN_LIBRARY_SHA256}
UPDATED_EXECUTOR = dict(VALIDATOR_DIGESTS)


class Libraries:
    """Stands in for both native libraries on both sides: the answer each library gives, and every
    challenge the validator built (which library, which config)."""

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
    """Both libraries patched into the service module; `answers` is set per test."""
    library = Libraries({LIB_PATH: _probe_payload(), CAPACITY_LIB_PATH: _probe_payload()})
    with (
        patch(f"{SERVICE}.VerifyXValidator", library.validator_cls),
        patch(f"{SERVICE}.sha256_from_path", side_effect=VALIDATOR_DIGESTS.__getitem__),
        patch(f"{SERVICE}._verify_memory_test", return_value=({"success": True}, [])),
        patch(f"{SERVICE}._verify_storage_test", return_value=({"success": True}, [])),
    ):
        yield library


@pytest.fixture
def libraries():
    with patched_libraries() as library:
        yield library


@pytest.fixture
def mode(monkeypatch):
    def set_mode(value: str) -> None:
        monkeypatch.setattr(settings.verifyx, "NETWORK_GATE_MODE", value)

    return set_mode


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


async def shadow_run(shell, timeout_seconds: float = 240.0):
    return await VerifyXValidationService().measure_capacity_shadow(
        shell=shell,
        executor_info=EXECUTOR_INFO,
        default_extra={"executor_uuid": "exec-1"},
        machine_spec=MACHINE_SPEC,
        timeout_seconds=timeout_seconds,
    )


def _sha256(relative: str) -> str:
    return hashlib.sha256((REPO / relative).read_bytes()).hexdigest()


def test_libverifyx_so_is_mains_build_and_the_capacity_build_sits_beside_it_in_both_images():
    assert _sha256("neurons/validators/libverifyx.so") == MAIN_LIBRARY_SHA256
    assert _sha256("neurons/executor/libverifyx.so") == MAIN_LIBRARY_SHA256
    assert _sha256("neurons/validators/libverifyx_capacity.so") == CAPACITY_LIBRARY_SHA256
    assert _sha256("neurons/executor/libverifyx_capacity.so") == CAPACITY_LIBRARY_SHA256
    assert LIB_PATH == "/usr/lib/libverifyx.so"
    assert CAPACITY_LIB_PATH == "/usr/lib/libverifyx_capacity.so"
    validator_dockerfile = (REPO / "neurons/validators/Dockerfile").read_text()
    executor_dockerfile = (REPO / "neurons/executor/Dockerfile").read_text()
    assert "RUN mv libverifyx.so /usr/lib/" in validator_dockerfile
    assert "RUN mv libverifyx_capacity.so /usr/lib/" in validator_dockerfile
    assert "mv /root/app/libverifyx.so /usr/lib/" in executor_dockerfile
    assert "mv /root/app/libverifyx_capacity.so /usr/lib/" in executor_dockerfile


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_mode", ["off", "shadow"])
async def test_off_and_shadow_gate_on_todays_library_and_ask_an_executor_for_nothing_new(
    libraries, mode, gate_mode
):
    mode(gate_mode)
    todays, updated = executor_shell(TODAYS_EXECUTOR), executor_shell(UPDATED_EXECUTOR)

    todays_result = await gated_run(todays)
    updated_result = await gated_run(updated)

    for shell, result in ((todays, todays_result), (updated, updated_result)):
        assert result.error is None and result.data["success"] is True
        assert asked_for(shell) == [LIB_PATH]
        [command] = commands(shell)
        assert command.startswith(MAIN_COMMAND_PREFIX) and "--lib" not in command
    assert [record.lib_name for record in libraries.built] == [LIB_PATH, LIB_PATH]


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_mode", ["off", "shadow"])
async def test_off_and_shadow_still_refuse_an_executor_whose_libverifyx_so_is_not_mains(
    libraries, mode, gate_mode
):
    mode(gate_mode)
    # the image this PR replaces on the branch (libverifyx.so = the capacity build) is not main's
    stale = executor_shell({LIB_PATH: CAPACITY_LIBRARY_SHA256})

    result = await gated_run(stale)

    assert result.error == OUTDATED_LIBRARY_ERROR
    assert commands(stale) == []


@pytest.mark.asyncio
async def test_enforce_gates_on_the_capacity_library_and_refuses_todays_executor_before_any_run(
    libraries, mode
):
    mode("enforce")
    todays, updated = executor_shell(TODAYS_EXECUTOR), executor_shell(UPDATED_EXECUTOR)

    refused = await gated_run(todays)
    gated = await gated_run(updated)

    assert refused.error == OUTDATED_LIBRARY_ERROR
    assert asked_for(todays) == [CAPACITY_LIB_PATH] and commands(todays) == []
    assert gated.error is None
    [command] = commands(updated)
    assert command.endswith(f" --lib {CAPACITY_LIB_PATH}")
    assert [record.lib_name for record in libraries.built] == [CAPACITY_LIB_PATH]
    assert gated.data["network"]["download_speed"] == 2400.0


@pytest.mark.asyncio
async def test_the_shadow_run_skips_an_executor_without_the_capacity_library(libraries):
    for digests in (TODAYS_EXECUTOR, {**TODAYS_EXECUTOR, CAPACITY_LIB_PATH: MAIN_LIBRARY_SHA256}):
        shell = executor_shell(digests)
        record = await shadow_run(shell)

        assert record["status"] == "library_missing"
        assert asked_for(shell) == [CAPACITY_LIB_PATH]
        assert commands(shell) == []
    assert libraries.built == []


@pytest.mark.asyncio
async def test_the_shadow_run_reads_the_capacity_library_light_and_under_its_timeout(libraries):
    libraries.answers[CAPACITY_LIB_PATH] = _probe_payload(capacity_mbps=2400.0, upload_mbps=1900.0)
    shell = executor_shell(UPDATED_EXECUTOR)

    record = await shadow_run(shell, timeout_seconds=200.0)

    assert {key: value for key, value in record.items() if key != "seconds"} == {
        "status": "measured",
        "capacity_download_speed": 2400.0,
        "upload_speed": 1900.0,
        "package_download_speed": 180.0,
        "success": True,
        "errors": [],
    }
    [command] = commands(shell)
    assert command.endswith(f" --lib {CAPACITY_LIB_PATH}")
    assert shell.ssh_client.run.await_args.kwargs["timeout"] == 200.0
    [built] = libraries.built
    assert built.lib_name == CAPACITY_LIB_PATH
    assert built.config["memory_max_test_gb"] == settings.verifyx.MEMORY_MIN_TEST_GB
    assert built.config["storage_throughput_test_gb"] == settings.FIRST_PASS_VERIFYX_STORAGE_TEST_GB
    assert built.config["network_timeout_seconds"] == settings.verifyx.NETWORK_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_every_shadow_run_failure_is_a_record(libraries):
    timed_out = executor_shell(UPDATED_EXECUTOR)
    timed_out.ssh_client.run = AsyncMock(side_effect=TimeoutError("240 s"))
    crashed = executor_shell(UPDATED_EXECUTOR)
    crashed.ssh_client.run = AsyncMock(
        return_value=SimpleNamespace(stdout="", stderr="Killed", exit_status=137)
    )
    rejected = executor_shell(UPDATED_EXECUTOR)

    records = [await shadow_run(timed_out), await shadow_run(crashed)]
    with patch.object(
        libraries.validator_cls, "verify_response", side_effect=RuntimeError("cipher rejected")
    ):
        records.append(await shadow_run(rejected))

    assert [record["status"] for record in records] == ["run_failed"] * 3
    assert records[0]["error"] == "SSH transport error (TimeoutError: 240 s)"
    assert records[1]["error"] == "exit status 137"
    assert records[2]["error"] == "RuntimeError: cipher rejected"
