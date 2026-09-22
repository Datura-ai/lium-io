import ctypes
import json
import math
import random
import os
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple, List

from core.config import FeatureFlag, settings
from core.utils import _m, get_extra_info
from core.checksums import sha256_from_executor, sha256_from_path


logger = logging.getLogger(__name__)

GB_TO_BYTES = 1024 * 1024 * 1024

# Minimum length of a valid cipher response from verifyx_executor.py stdout.
# Shorter stdout is treated as empty/truncated (OOM, disk-full, etc.).
MIN_CIPHER_LEN = 64

# Cap on stderr bytes captured per failure. Last 2 KB is kept when longer.
STDERR_TAIL_BYTES = 2048

# Hard cap on the remote verifyx run. Memory/storage/network probes can legitimately take
# minutes, so this is generous — it only cuts true hangs (same silent-hang class as the
# matrix check, DAH-2365) instead of blocking until the outer JOB_TIME_OUT cancellation.
VERIFYX_COMMAND_TIMEOUT_SECONDS = 600

# DAH-2774: both images ship two VerifyX builds. LIB_PATH is the one every executor runs today; off
# and shadow gate on it exactly as before and ask executors for nothing new. CAPACITY_LIB_PATH
# (celium-gpu-verifier#25) reads the Cloudflare capacity: beside the gate under shadow
# (`measure_capacity_shadow`), as the gate under enforce.
LIB_PATH = "/usr/lib/libverifyx.so"
CAPACITY_LIB_PATH = "/usr/lib/libverifyx_capacity.so"


class VerifyXFailureClass(str, Enum):
    SSH_TRANSPORT = "SSH_TRANSPORT"
    EXECUTOR_CRASH = "EXECUTOR_CRASH"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    CIPHER_REJECTED = "CIPHER_REJECTED"
    UNKNOWN = "UNKNOWN"


class SSHCapture(NamedTuple):
    """Result of an SSH command. `transport_error` is None on normal completion (even if exit_status != 0)."""

    stdout: str | None = None
    stderr: str | None = None
    exit_status: int | None = None
    transport_error: str | None = None


def _classify_failure(
    exit_status: int | None,
    stdout: str | None,
    stderr: str | None,
    transport_error: str | None,
) -> VerifyXFailureClass:
    # A non-zero exit is a stronger signal than stdout shape — a crashing process can still
    # flush partial output or a traceback to stdout before dying, and we must not mislabel
    # that as a cipher rejection.
    if transport_error is not None:
        return VerifyXFailureClass.SSH_TRANSPORT
    if exit_status is not None and exit_status != 0:
        return VerifyXFailureClass.EXECUTOR_CRASH
    stdout_stripped = (stdout or "").strip()
    if len(stdout_stripped) >= MIN_CIPHER_LEN:
        return VerifyXFailureClass.CIPHER_REJECTED
    # Short/empty stdout, no crash, no transport error. EMPTY_RESPONSE when we have *some*
    # signal (zero exit code OR non-empty stdout); UNKNOWN only when we have no signal at all.
    if stdout_stripped or exit_status == 0:
        return VerifyXFailureClass.EMPTY_RESPONSE
    return VerifyXFailureClass.UNKNOWN


def _tail_stderr(stderr: str | None) -> str | None:
    if stderr is None:
        return None
    data = stderr.encode("utf-8")
    if len(data) <= STDERR_TAIL_BYTES:
        return stderr
    # errors="ignore" drops any leading partial UTF-8 sequence from the cut.
    return data[-STDERR_TAIL_BYTES:].decode("utf-8", errors="ignore")


class VerifyXValidator:
    def __init__(self, lib_name: str, seed: int):
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self.service = self._create_service()
        self.seed = seed

    def _setup_signatures(self):
        self.lib.service_new.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.generate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.lib.generate.restype = ctypes.c_int
        self.lib.get_cipher_text.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.get_cipher_text.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.verify.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_uint64,
        ]
        self.lib.verify.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.service_del.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.str_del.argtypes = [ctypes.POINTER(ctypes.c_char)]

    def _create_service(self):
        return self.lib.service_new()

    def __del__(self):
        self.lib.service_del(self.service)

    def _decode_string(self, ptr):
        return ctypes.string_at(ptr).decode("utf-8") if ptr else None

    def generate_challenge(self, challenge_input) -> str:
        challenge_input_json = json.dumps(challenge_input).encode("utf-8")
        if self.lib.generate(self.service, challenge_input_json) != 0:
            raise RuntimeError("Failed to generate challenge")
        cipher_ptr = self.lib.get_cipher_text(self.service)
        cipher_hex = self._decode_string(cipher_ptr)
        self.lib.str_del(cipher_ptr)
        return cipher_hex

    def verify_response(self, response_cipher_hex: str) -> Dict[str, Any]:
        verify_ptr = self.lib.verify(self.service, response_cipher_hex.encode("utf-8"), self.seed)
        if not verify_ptr:
            raise RuntimeError("Failed to verify challenge response")
        try:
            verify_result = self._decode_string(verify_ptr)
            verify_data = json.loads(verify_result)
            return verify_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
        finally:
            self.lib.str_del(verify_ptr)


@dataclass
class VerifyXResponse:
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


OUTDATED_LIBRARY_ERROR = (
    "Executor using outdated VerifyX library. Run docker compose restart to update to the latest executor image"
)


@dataclass
class VerifyXChallenge:
    """One prepared VerifyX challenge and the validator object that judges its answer (see
    `VerifyXValidationService.prepare_verifyx_challenge`)."""

    validator: "VerifyXValidator"
    seed: int
    cipher_text: str
    challenge_input: Dict[str, Any]
    log_extra: Dict[str, Any]
    # sha256 of the validator's own libverifyx.so; the executor's must match before its answer counts.
    expected_lib_sha256: str


@dataclass
class NetworkGateTally:
    """DAH-2774: the download-EMA floor (checks/verifyx.py) read against both download readings,
    counted per node since the last summary (the validator logs one per cycle; express-lane runs
    between cycles land in the next one). `newly_fail` is a node the package reading passes and
    the capacity reading fails: what flipping VERIFYX_NETWORK_GATE_MODE to enforce would cost.
    `capacity_library_missing` is a node without the validator's libverifyx_capacity.so: one
    enforce would fail as outdated."""

    package_pass: int = 0
    package_fail: int = 0
    capacity_pass: int = 0
    capacity_fail: int = 0
    newly_fail: int = 0
    newly_pass: int = 0
    unmeasured: int = 0
    capacity_library_missing: int = 0
    probe_failed: int = 0

    def record(
        self,
        package_ema: float | None,
        capacity_ema: float | None,
        floor_mbps: float,
        capacity_library_missing: bool = False,
    ) -> None:
        if capacity_library_missing:
            self.capacity_library_missing += 1
        if package_ema is None or capacity_ema is None:
            self.unmeasured += 1
            return
        package_passes = package_ema >= floor_mbps
        capacity_passes = capacity_ema >= floor_mbps
        self.package_pass += package_passes
        self.package_fail += not package_passes
        self.capacity_pass += capacity_passes
        self.capacity_fail += not capacity_passes
        self.newly_fail += package_passes and not capacity_passes
        self.newly_pass += capacity_passes and not package_passes

    def record_probe_failed(self) -> None:
        self.probe_failed += 1

    def log_and_reset(self, floor_mbps: float, default_extra: dict) -> dict[str, int]:
        counts = {name: getattr(self, name) for name in self.__dataclass_fields__}
        message = (
            "VerifyX network gate summary "
            f"mode={settings.verifyx.NETWORK_GATE_MODE} floor_mbps={floor_mbps:.0f} "
            + " ".join(f"{name}={value}" for name, value in counts.items())
        )
        logger.info(
            _m(
                message,
                extra=get_extra_info(
                    {
                        **default_extra,
                        "network_gate_mode": settings.verifyx.NETWORK_GATE_MODE,
                        "floor_mbps": floor_mbps,
                        **counts,
                    }
                ),
            )
        )
        for name in counts:
            setattr(self, name, 0)
        return counts


# One per process: the validator and the ioc container each build a VerifyXValidationService.
NETWORK_GATE_TALLY = NetworkGateTally()


class VerifyXValidationService:
    def __init__(self):
        self.lib_name = LIB_PATH
        self.capacity_lib_name = CAPACITY_LIB_PATH
        self._lib_sha256s: dict[str, str] = {}

    def gated_lib_name(self) -> str:
        """The library whose answer gates: libverifyx_capacity.so under enforce, libverifyx.so otherwise."""
        if settings.verifyx.NETWORK_GATE_MODE == "enforce":
            return self.capacity_lib_name
        return self.lib_name

    def lib_sha256(self, lib_name: str | None = None) -> str:
        """sha256 of one of the validator's own libraries (the gated one by default), read once per process.

        The files change only with a validator upgrade, which restarts the process, so both readers
        (the SSH path's checksum gate and the challenge's `expected_lib_sha256`) share one digest.
        """
        lib_name = lib_name or self.gated_lib_name()
        if lib_name not in self._lib_sha256s:
            self._lib_sha256s[lib_name] = sha256_from_path(lib_name)
        return self._lib_sha256s[lib_name]

    def prepare_verifyx_challenge(
        self,
        machine_spec: dict,
        default_extra: dict,
        challenge_config_overrides: dict | None = None,
        *,
        lib_name: str | None = None,
    ) -> "VerifyXChallenge":
        """Encrypt one VerifyX challenge; the caller decides how it reaches the executor.

        The SSH path runs `verifyx_executor.py --seed … --cipher_text …` over the shell; the local
        path (liumd phase 1, `POST /verify`) sends the same two arguments in the intent. Either way
        the response comes back to `evaluate_verifyx_capture`, the one place that decides.
        `lib_name` defaults to the gated library.
        """
        # challenge_config_overrides (DAH-3011): keys of the challenge `config` block to replace for
        # this run — a first, unscored verification writes less RAM/disk. None = today's config.
        lib_name = lib_name or self.gated_lib_name()
        gpu_details = machine_spec.get("gpu", {}).get("details", [])
        gpu_count = machine_spec.get("gpu", {}).get("count", 0)
        gpu_uuids = ",".join([detail.get("uuid", "") for detail in gpu_details])
        gpu_model = gpu_details[0].get("name", "") if gpu_details else ""

        gpu_info = {"uuids": gpu_uuids, "gpu_count": gpu_count, "gpu_model": gpu_model}

        seed = random.getrandbits(64)
        verifyx_validator = VerifyXValidator(lib_name, seed)

        challenge_config = {
            "memory_allocation_percentage": settings.verifyx.MEMORY_ALLOCATION_PERCENTAGE,
            "memory_min_test_gb": settings.verifyx.MEMORY_MIN_TEST_GB,
            "memory_max_test_gb": settings.verifyx.MEMORY_MAX_TEST_GB,
            "storage_min_available_gb": settings.verifyx.STORAGE_MIN_AVAILABLE_GB,
            "storage_throughput_test_gb": settings.verifyx.STORAGE_THROUGHPUT_TEST_GB,
            "network_timeout_seconds": settings.verifyx.NETWORK_TIMEOUT_SECONDS,
            "enable_xet_challenge": settings.verifyx.ENABLE_XET_CHALLENGE,
        }
        if challenge_config_overrides:
            challenge_config.update(challenge_config_overrides)
        challenge_input = {
            "seed": seed,
            "machine_info": gpu_info,
            "config": challenge_config,
        }

        cipher_text = verifyx_validator.generate_challenge(challenge_input)
        log_extra = {
            **default_extra,
            "seed": seed,
            "cipher_text": cipher_text,
            "challenge_input": challenge_input,
        }
        return VerifyXChallenge(
            validator=verifyx_validator,
            seed=seed,
            cipher_text=cipher_text,
            challenge_input=challenge_input,
            log_extra=log_extra,
            expected_lib_sha256=self.lib_sha256(lib_name),
        )

    def evaluate_verifyx_capture(
        self,
        challenge: "VerifyXChallenge",
        capture: SSHCapture,
        default_extra: dict,
    ) -> "VerifyXResponse":
        """Judge the executor's `verifyx_executor.py` output — SSH capture or `/verify` step alike."""
        if capture.transport_error is not None:
            return self._failure_response(
                error=f"SSH transport error ({capture.transport_error})",
                ssh_capture=capture,
                default_extra=default_extra,
            )

        challenge_response = (capture.stdout or "").strip()

        logger.info(_m("Challenge response received", extra=get_extra_info({**challenge.log_extra, "challenge_response": challenge_response})))

        # A crashing process may flush partial output before dying; exit_status wins over stdout shape.
        if capture.exit_status is not None and capture.exit_status != 0:
            return self._failure_response(
                error=f"Executor process exited with status {capture.exit_status}",
                ssh_capture=capture,
                default_extra=default_extra,
            )

        if len(challenge_response) < MIN_CIPHER_LEN:
            return self._failure_response(
                error=f"Executor returned empty or truncated response (stdout_len={len(challenge_response)})",
                ssh_capture=capture,
                default_extra=default_extra,
            )

        try:
            payload = challenge.validator.verify_response(challenge_response)
            verification_result = _perform_verification_checks(payload)
            _log_verifyx_network_speeds(verification_result.get("network") or {}, default_extra)
            return VerifyXResponse(data=verification_result)
        except Exception as e:
            return self._failure_response(
                error=f"challenge verification failed ({str(e)})",
                ssh_capture=capture,
                default_extra=default_extra,
            )

    async def validate_verifyx_and_process_job(
        self,
        shell,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
        challenge_config_overrides: dict | None = None,
    ):
        # The SSH transport: library checksum over the shell → prepare_verifyx_challenge → one
        # remote `verifyx_executor.py` run → evaluate_verifyx_capture. The local transport
        # (checks/local_verify.py) calls the same prepare/evaluate around `POST /verify`.
        try:
            # Verify checksum before proceeding with validation
            lib_name = self.gated_lib_name()
            local_checksum = self.lib_sha256(lib_name)
            executor_checksum = await sha256_from_executor(shell, lib_name)

            if local_checksum != executor_checksum:
                return VerifyXResponse(error=OUTDATED_LIBRARY_ERROR)

            challenge = self.prepare_verifyx_challenge(
                machine_spec, default_extra, challenge_config_overrides, lib_name=lib_name
            )

            command = self._verifyx_command(executor_info, challenge, lib_name)

            logger.info(_m("VerifyX Python Script Command", extra=get_extra_info(challenge.log_extra)))

            ssh_capture = await self._run_ssh_command(shell, command)
            return self.evaluate_verifyx_capture(challenge, ssh_capture, default_extra)

        except Exception as e:
            # Pre-SSH failure (checksum fetch, challenge generation, etc.) — emit a structured
            # log line and classify as UNKNOWN so the exception text is not mistaken for SSH transport.
            diagnostics = {
                "failure_class": VerifyXFailureClass.UNKNOWN.value,
                "internal_error": f"{type(e).__name__}: {e}",
            }
            logger.error(_m("VerifyX validation failed", extra=get_extra_info({**default_extra, **diagnostics})))
            return VerifyXResponse(error=f"unexpected error ({e})", diagnostics=diagnostics)

    async def measure_capacity_shadow(
        self,
        shell,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
        timeout_seconds: float,
    ) -> dict:
        """DAH-2774 shadow: one libverifyx_capacity.so run, read as enforce would read its network.

        Recorded beside the gate and never scored, so every outcome is a record (`status`
        measured | library_missing | run_failed), never an exception. Memory and storage run at
        their smallest sizes; the network test is the one enforce would gate on. An executor
        whose libverifyx_capacity.so is absent or another build is not run (`library_missing`).
        """
        started = time.monotonic()
        record = await self._capacity_shadow_run(
            shell, executor_info, default_extra, machine_spec, timeout_seconds
        )
        record["seconds"] = round(time.monotonic() - started, 1)
        logger.info(
            _m(
                "VerifyX capacity shadow run "
                f"status={record['status']} "
                f"cloudflare_download_mbps={_format_mbps(record.get('capacity_download_speed'))} "
                f"exec={default_extra.get('executor_uuid') or 'none'}",
                extra=get_extra_info({**default_extra, "capacity_shadow": record}),
            )
        )
        return record

    async def _capacity_shadow_run(
        self, shell, executor_info, default_extra: dict, machine_spec: dict, timeout_seconds: float
    ) -> dict:
        try:
            lib_name = self.capacity_lib_name
            executor_checksum = await sha256_from_executor(shell, lib_name)
            if executor_checksum != self.lib_sha256(lib_name):
                return {"status": "library_missing", "executor_lib_sha256": executor_checksum or None}
            challenge = self.prepare_verifyx_challenge(
                machine_spec,
                default_extra,
                {
                    "memory_max_test_gb": settings.verifyx.MEMORY_MIN_TEST_GB,
                    "storage_throughput_test_gb": settings.FIRST_PASS_VERIFYX_STORAGE_TEST_GB,
                },
                lib_name=lib_name,
            )
            capture = await self._run_ssh_command(
                shell,
                self._verifyx_command(executor_info, challenge, lib_name),
                timeout=timeout_seconds,
            )
            if capture.transport_error is not None:
                return {"status": "run_failed", "error": f"SSH transport error ({capture.transport_error})"}
            if capture.exit_status is not None and capture.exit_status != 0:
                return {"status": "run_failed", "error": f"exit status {capture.exit_status}"}
            response = (capture.stdout or "").strip()
            if len(response) < MIN_CIPHER_LEN:
                return {"status": "run_failed", "error": f"stdout_len={len(response)}"}
            payload = challenge.validator.verify_response(response)
            stats, errors = _verify_network_capacity_test(
                payload["challenge_data"], payload["response_data"]
            )
        except Exception as e:
            return {"status": "run_failed", "error": f"{type(e).__name__}: {e}"}
        return {
            "status": "measured",
            "capacity_download_speed": stats.get("capacity_download_speed"),
            "upload_speed": stats.get("upload_speed"),
            "package_download_speed": stats.get("package_download_speed"),
            "success": stats.get("success"),
            "errors": errors,
        }

    def _verifyx_command(self, executor_info, challenge: "VerifyXChallenge", lib_name: str) -> str:
        command = f"{executor_info.python_path} {executor_info.root_dir}/src/verifyx_executor.py --seed {challenge.seed} --cipher_text {challenge.cipher_text}"
        if lib_name != self.lib_name:
            command += f" --lib {lib_name}"
        return command

    async def _run_ssh_command(
        self, shell, command: str, timeout: float = VERIFYX_COMMAND_TIMEOUT_SECONDS
    ) -> SSHCapture:
        """Run SSH command; on transport failure populate `transport_error`, else the payload fields."""
        try:
            result = await shell.ssh_client.run(command, timeout=timeout)
        except Exception as e:
            return SSHCapture(transport_error=f"{type(e).__name__}: {e}")

        if result is None:
            return SSHCapture(transport_error="SSH command returned no result")

        try:
            return SSHCapture(
                stdout=result.stdout,
                stderr=_tail_stderr(getattr(result, "stderr", None)),
                exit_status=getattr(result, "exit_status", None),
            )
        except AttributeError:
            return SSHCapture(transport_error="SSH result missing stdout")

    def _failure_response(
        self,
        *,
        error: str,
        ssh_capture: SSHCapture,
        default_extra: dict,
    ) -> "VerifyXResponse":
        """Classify the SSH capture, emit one ERROR log line, and return a populated VerifyXResponse."""
        failure_class = _classify_failure(
            ssh_capture.exit_status,
            ssh_capture.stdout,
            ssh_capture.stderr,
            ssh_capture.transport_error,
        )
        diagnostics = {
            "failure_class": failure_class.value,
            "exit_status": ssh_capture.exit_status,
            "stdout_len": len(ssh_capture.stdout) if ssh_capture.stdout is not None else None,
            "stderr_tail": ssh_capture.stderr,
            "transport_error": ssh_capture.transport_error,
        }
        logger.error(
            _m("VerifyX validation failed", extra=get_extra_info({**default_extra, **diagnostics, "error": error}))
        )
        return VerifyXResponse(error=error, diagnostics=diagnostics)


def _get_memory_stats(memory_execution: dict, success: bool) -> dict:
    return {
        "total": memory_execution["stats"]["total_bytes"] / 1024,
        "used": memory_execution["stats"]["used_bytes"] / 1024,
        "free": memory_execution["stats"]["free_bytes"] / 1024,
        "available": memory_execution["stats"]["available_bytes"] / 1024,
        "utilization": (
            (memory_execution["stats"]["used_bytes"] / memory_execution["stats"]["total_bytes"]) * 100
            if memory_execution["stats"]["total_bytes"] > 0
            else 0
        ),
        "success": success,
        "execution_time_ms": memory_execution["execution_time_ms"],
    }


def _get_storage_stats(storage_execution: dict, success: bool) -> dict:
    return {
        "total": storage_execution["stats"]["total_bytes"] / 1024,
        "used": storage_execution["stats"]["used_bytes"] / 1024,
        "free": storage_execution["stats"]["free_bytes"] / 1024,
        "utilization": storage_execution["stats"]["utilization_percent"],
        "success": success,
        "write_throughput_mb_s": storage_execution["write_throughput_mb_s"],
        "read_throughput_mb_s": storage_execution["read_throughput_mb_s"],
        "execution_time_ms": storage_execution["execution_time_ms"],
    }


def _verify_memory_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
    memory_execution = response_data["memory_execution"]

    if not memory_execution["success"]:
        return _get_memory_stats(memory_execution, False), [memory_execution["error"]]
    errors = []
    success = True

    memory_challenge = challenge_data["memory_challenge"]
    min_size_bytes = memory_challenge["min_test_gb"] * GB_TO_BYTES
    if memory_execution["allocated_bytes"] < min_size_bytes:
        allocated_gb = memory_execution["allocated_bytes"] / GB_TO_BYTES
        required_gb = min_size_bytes / GB_TO_BYTES
        errors.append(f"Insufficient memory: {allocated_gb:.0f} GB allocated, {required_gb:.0f} GB required")
        success = False

    stats = _get_memory_stats(memory_execution, success)

    return stats, errors


def _verify_network_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
    """The gated answer's network. Off and shadow judge libverifyx.so's answer with main's
    function unchanged: the same package reading, the same failure handling (a failed probe
    carries no `download_speed`, so the EMA is fed 0.0) and the same exceptions, plus the package
    reading as `package_download_speed`. The shadow capacity reading comes from its own run
    (`VerifyXValidationService.measure_capacity_shadow`). Enforce judges libverifyx_capacity.so's
    answer on its capacity (`_verify_network_capacity_test`)."""
    if settings.verifyx.NETWORK_GATE_MODE == "enforce":
        return _verify_network_capacity_test(challenge_data, response_data)
    stats, errors = _verify_network_package_test(challenge_data, response_data)
    package_speed = (response_data["network_execution"].get("download") or {}).get("speed_mbps")
    stats["package_download_speed"] = package_speed if _is_speed_reading(package_speed) else None
    return stats, errors


def _verify_network_package_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
    """main's `_verify_network_test`, verbatim: what off and shadow gate and list on."""
    network_execution = response_data["network_execution"]

    if not network_execution["success"]:
        return {"success": False}, [f"Network execution failed: {network_execution.get('error', 'Unknown error')}"]

    errors = []
    success = True

    expected_download = challenge_data["network_challenge"]["download"]
    download_result = network_execution["download"]
    if download_result["pkg"] != expected_download["pkg"]:
        errors.append(f"Resource validation failed: {download_result['pkg']}")
        success = False

    if download_result["size"] != expected_download["size"]:
        errors.append(f"Size validation failed for {download_result['pkg']}")
        success = False

    if download_result["hash"] != expected_download["hash"]:
        errors.append(f"Integrity check failed for {download_result['pkg']}")
        success = False

    download_speed = network_execution["download"]["speed_mbps"]
    upload_speed = network_execution.get("speedtest", {}).get("upload_mbps")

    if download_speed < settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS:
        errors.append(
            f"Network download speed inadequate: {download_speed:.2f} Mbps achieved, {settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS:.0f} Mbps required"
        )
        success = False

    stats = {
        "download_speed": download_speed,
        "upload_speed": upload_speed,
        "success": success,
        "execution_time_ms": network_execution["execution_time_ms"],
    }

    return stats, errors


def _verify_network_capacity_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
    """libverifyx_capacity.so's network (the enforce gate and the shadow record): `download_speed`
    is the Cloudflare capacity; the package download keeps its own floor and is published as
    `package_download_speed`, the capacity as `capacity_download_speed`."""
    network_execution = response_data["network_execution"]

    if not network_execution["success"]:
        # The probe fails as a whole when either direction fails, but the directions are
        # independent (celium-gpu-verifier#25): an upload that could not move its payload still
        # leaves real download readings from the same run. Keep the capacity reading so the fatal
        # download EMA (checks/verifyx.py) is fed the measured value, not a 0; success and
        # upload_speed carry the failure.
        speedtest = network_execution.get("speedtest") or {}
        capacity_speed = speedtest.get("download_mbps")
        package_speed = (network_execution.get("download") or {}).get("speed_mbps")
        upload_speed = speedtest.get("upload_mbps")
        stats = {
            "download_speed": capacity_speed if _is_positive_number(capacity_speed) else None,
            # 0.0 is how the probe reports the failed direction; anything that is not a number is
            # a malformed payload and reads as "no upload measurement" like the download above.
            "upload_speed": upload_speed if _is_speed_reading(upload_speed) else None,
            "package_download_speed": package_speed if _is_positive_number(package_speed) else None,
            "success": False,
            "capacity_download_speed": capacity_speed if _is_positive_number(capacity_speed) else None,
            "execution_time_ms": network_execution.get("execution_time_ms"),
        }
        return stats, [f"Network execution failed: {network_execution.get('error', 'Unknown error')}"]

    errors = []
    success = True

    expected_download = challenge_data["network_challenge"]["download"]
    download_result = network_execution["download"]
    if download_result["pkg"] != expected_download["pkg"]:
        errors.append(f"Resource validation failed: {download_result['pkg']}")
        success = False

    if download_result["size"] != expected_download["size"]:
        errors.append(f"Size validation failed for {download_result['pkg']}")
        success = False

    if download_result["hash"] != expected_download["hash"]:
        errors.append(f"Integrity check failed for {download_result['pkg']}")
        success = False

    speedtest = network_execution.get("speedtest") or {}
    upload_speed = speedtest.get("upload_mbps")
    capacity_speed = speedtest.get("download_mbps")
    package_download_speed = download_result.get("speed_mbps")
    download_speed = capacity_speed
    stats = {
        "download_speed": download_speed,
        "upload_speed": upload_speed,
        "package_download_speed": package_download_speed,
        "capacity_download_speed": capacity_speed,
        "success": success,
        "execution_time_ms": network_execution.get("execution_time_ms"),
    }

    # A probe that could not reach Cloudflare (no `speedtest` block, a null or zero reading) is a
    # failed measurement, never an exception: an exception here is caught upstream as "challenge
    # verification failed" and rejects the machine even while VERIFYX_NETWORK_VALIDATION is off.
    if not all(_is_positive_number(value) for value in (upload_speed, capacity_speed, package_download_speed)):
        errors.append("Network performance data unavailable")
        return {**stats, "success": False}, errors

    if download_speed < settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS:
        errors.append(
            f"Cloudflare download speed inadequate: {download_speed:.2f} Mbps achieved, {settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS:.0f} Mbps required"
        )
        success = False

    if package_download_speed < settings.verifyx.NETWORK_MIN_PACKAGE_DOWNLOAD_SPEED_MBPS:
        errors.append(
            f"Package download speed inadequate: {package_download_speed:.2f} Mbps achieved, {settings.verifyx.NETWORK_MIN_PACKAGE_DOWNLOAD_SPEED_MBPS:.0f} Mbps required"
        )
        success = False

    return {**stats, "success": success}, errors


def _verify_storage_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
    storage_execution = response_data["storage_execution"]

    if storage_execution.get("error"):
        return _get_storage_stats(storage_execution, False), [storage_execution["error"]]

    errors = []
    success = True

    storage_challenge = challenge_data["storage_challenge"]
    expected_sparse_bytes = storage_challenge["minimum_free_storage_gb"] * GB_TO_BYTES
    if storage_execution["allocated_space_bytes"] < expected_sparse_bytes:
        allocated_gb = storage_execution["allocated_space_bytes"] / GB_TO_BYTES
        required_gb = expected_sparse_bytes / GB_TO_BYTES
        errors.append(f"Insufficient storage: {allocated_gb:.0f} GB allocated, {required_gb:.0f} GB required")
        success = False

    stats = _get_storage_stats(storage_execution, success)

    return stats, errors


def _verify_xet_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
    xet_challenge = challenge_data.get("xet_challenge") or {}
    xet_execution = response_data.get("xet_execution") or {}
    expected_download = xet_challenge.get("download") or {}

    if not expected_download.get("url"):
        return {
            "status": "skipped",
            "success": True,
            "bytes_downloaded": 0,
            "speed_mbps": 0.0,
            "elapsed_ms": 0,
            "token_fetch_ms": 0,
            "hash": "",
        }, []

    status = xet_execution.get("status", "failed")
    success = bool(xet_execution.get("success"))
    speed_mbps = xet_execution.get("speed_mbps", 0.0)
    errors: List[str] = []

    if status == "failed":
        errors.append(f"Xet execution failed: {xet_execution.get('error', 'Unknown error')}")
        success = False
    elif speed_mbps < settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS:
        errors.append(
            f"Xet download speed inadequate: {speed_mbps:.2f} Mbps achieved, "
            f"{settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS:.0f} Mbps required"
        )
        success = False

    if xet_execution.get("pkg") != expected_download.get("pkg"):
        errors.append(f"Resource validation failed: {xet_execution.get('pkg', '')}")
        success = False

    if xet_execution.get("bytes_downloaded", 0) != expected_download.get("size", 0):
        errors.append(f"Size validation failed for {xet_execution.get('pkg', '')}")
        success = False

    if xet_execution.get("hash") != expected_download.get("hash"):
        errors.append(f"Integrity check failed for {xet_execution.get('pkg', '')}")
        success = False

    return {
        "status": status,
        "success": success,
        "bytes_downloaded": xet_execution.get("bytes_downloaded", 0),
        "speed_mbps": speed_mbps,
        "elapsed_ms": xet_execution.get("elapsed_ms", 0),
        "token_fetch_ms": xet_execution.get("token_fetch_ms", 0),
        "hash": xet_execution.get("hash", ""),
        "error": xet_execution.get("error"),
    }, errors


def _is_positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_speed_reading(value: object) -> bool:
    """A usable Mbps reading: a finite positive number, or 0 (how a failed direction reads).

    A bool, a string, NaN, ±inf or a negative number is not one — both vendored libraries only
    serializes f64, so any such value is a malformed payload and must never reach EMA arithmetic.
    """
    if _is_positive_number(value):
        return math.isfinite(value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _format_mbps(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "none"


def _log_verifyx_network_speeds(network: dict, default_extra: dict) -> None:
    package_download_mbps = network.get("package_download_speed")
    cloudflare_download_mbps = network.get("capacity_download_speed")
    cloudflare_upload_mbps = network.get("upload_speed")
    exec_id = default_extra.get("executor_uuid") or "none"
    message = (
        "VerifyX network speeds "
        f"package_download_mbps={_format_mbps(package_download_mbps)} "
        f"cloudflare_download_mbps={_format_mbps(cloudflare_download_mbps)} "
        f"cloudflare_upload_mbps={_format_mbps(cloudflare_upload_mbps)} "
        f"success={network.get('success')} "
        f"gate_mode={settings.verifyx.NETWORK_GATE_MODE} "
        f"exec={exec_id}"
    )
    logger.info(
        _m(
            message,
            extra=get_extra_info(
                {
                    **default_extra,
                    "package_download_mbps": package_download_mbps,
                    "cloudflare_download_mbps": cloudflare_download_mbps,
                    "cloudflare_upload_mbps": cloudflare_upload_mbps,
                    "network_success": network.get("success"),
                }
            ),
        )
    )


def _perform_verification_checks(payload: dict) -> Dict[str, Any]:
    challenge_data = payload["challenge_data"]
    response_data = payload["response_data"]

    network_stats, network_errors = _verify_network_test(challenge_data, response_data)
    memory_stats, memory_errors = _verify_memory_test(challenge_data, response_data)
    storage_stats, storage_errors = _verify_storage_test(challenge_data, response_data)
    all_errors = network_errors + memory_errors + storage_errors

    required_checks = [
        memory_stats["success"],
    ]

    if settings.FEATURE_FLAGS.get(FeatureFlag.VERIFYX_NETWORK_VALIDATION, False):
        required_checks.append(network_stats["success"])

    if not settings.debug.SKIP_STORAGE_CHECK:
        required_checks.append(storage_stats["success"])

    result = {
        "success": all(required_checks),
        "network": network_stats,
        "hard_disk": storage_stats,
        "ram": memory_stats,
        "errors": all_errors,
    }

    if "xet_execution" in response_data:
        xet_stats, xet_errors = _verify_xet_test(challenge_data, response_data)
        log_extra = {
            "xet_status": xet_stats.get("status"),
            "xet_success": xet_stats.get("success"),
            "xet_bytes_downloaded": xet_stats.get("bytes_downloaded"),
            "xet_speed_mbps": xet_stats.get("speed_mbps"),
            "xet_elapsed_ms": xet_stats.get("elapsed_ms"),
            "xet_token_fetch_ms": xet_stats.get("token_fetch_ms"),
            "xet_hash": xet_stats.get("hash"),
            "xet_error": xet_stats.get("error"),
            "xet_errors": xet_errors,
        }
        if xet_stats.get("success"):
            logger.info(_m("VerifyX Xet execution passed", extra=get_extra_info(log_extra)))
        else:
            logger.warning(_m("VerifyX Xet execution failed", extra=get_extra_info(log_extra)))
        result["xet"] = xet_stats

    return result
