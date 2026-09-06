import ctypes
import json
import random
import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple, List

from core.config import settings, FeatureFlag
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


class VerifyXValidationService:
    def __init__(self):
        self.lib_name = "/usr/lib/libverifyx.so"

    async def validate_verifyx_and_process_job(
        self,
        shell,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
    ):
        try:
            # Verify checksum before proceeding with validation
            local_checksum = sha256_from_path(self.lib_name)
            executor_checksum = await sha256_from_executor(shell, self.lib_name)

            if local_checksum != executor_checksum:
                return VerifyXResponse(error="Executor using outdated VerifyX library. Run docker compose restart to update to the latest executor image")

            gpu_details = machine_spec.get("gpu", {}).get("details", [])
            gpu_count = machine_spec.get("gpu", {}).get("count", 0)
            gpu_uuids = ",".join([detail.get("uuid", "") for detail in gpu_details])
            gpu_model = gpu_details[0].get("name", "") if gpu_details else ""

            gpu_info = {"uuids": gpu_uuids, "gpu_count": gpu_count, "gpu_model": gpu_model}

            seed = random.getrandbits(64)
            verifyx_validator = VerifyXValidator(self.lib_name, seed)

            challenge_input = {
                "seed": seed,
                "machine_info": gpu_info,
                "config": {
                    "memory_allocation_percentage": settings.verifyx.MEMORY_ALLOCATION_PERCENTAGE,
                    "memory_min_test_gb": settings.verifyx.MEMORY_MIN_TEST_GB,
                    "memory_max_test_gb": settings.verifyx.MEMORY_MAX_TEST_GB,
                    "storage_min_available_gb": settings.verifyx.STORAGE_MIN_AVAILABLE_GB,
                    "storage_throughput_test_gb": settings.verifyx.STORAGE_THROUGHPUT_TEST_GB,
                    "network_timeout_seconds": settings.verifyx.NETWORK_TIMEOUT_SECONDS,
                    "enable_xet_challenge": settings.verifyx.ENABLE_XET_CHALLENGE,
                },
            }

            cipher_text = verifyx_validator.generate_challenge(challenge_input)

            command = f"{executor_info.python_path} {executor_info.root_dir}/src/verifyx_executor.py --seed {seed} --cipher_text {cipher_text}"

            log_extra = {
                **default_extra,
                "seed": seed,
                "cipher_text": cipher_text,
                "challenge_input": challenge_input,
            }

            logger.info(_m("VerifyX Python Script Command", extra=get_extra_info(log_extra)))

            ssh_capture = await self._run_ssh_command(shell, command)

            if ssh_capture.transport_error is not None:
                return self._failure_response(
                    error=f"SSH transport error ({ssh_capture.transport_error})",
                    ssh_capture=ssh_capture,
                    default_extra=default_extra,
                )

            challenge_response = (ssh_capture.stdout or "").strip()

            logger.info(_m("Challenge response received", extra=get_extra_info({**log_extra, "challenge_response": challenge_response})))

            # A crashing process may flush partial output before dying; exit_status wins over stdout shape.
            if ssh_capture.exit_status is not None and ssh_capture.exit_status != 0:
                return self._failure_response(
                    error=f"Executor process exited with status {ssh_capture.exit_status}",
                    ssh_capture=ssh_capture,
                    default_extra=default_extra,
                )

            if len(challenge_response) < MIN_CIPHER_LEN:
                return self._failure_response(
                    error=f"Executor returned empty or truncated response (stdout_len={len(challenge_response)})",
                    ssh_capture=ssh_capture,
                    default_extra=default_extra,
                )

            try:
                payload = verifyx_validator.verify_response(challenge_response)
                verification_result = _perform_verification_checks(payload)
                return VerifyXResponse(data=verification_result)
            except Exception as e:
                return self._failure_response(
                    error=f"challenge verification failed ({str(e)})",
                    ssh_capture=ssh_capture,
                    default_extra=default_extra,
                )

        except Exception as e:
            # Pre-SSH failure (checksum fetch, challenge generation, etc.) — emit a structured
            # log line and classify as UNKNOWN so the exception text is not mistaken for SSH transport.
            diagnostics = {
                "failure_class": VerifyXFailureClass.UNKNOWN.value,
                "internal_error": f"{type(e).__name__}: {e}",
            }
            logger.error(_m("VerifyX validation failed", extra=get_extra_info({**default_extra, **diagnostics})))
            return VerifyXResponse(error=f"unexpected error ({e})", diagnostics=diagnostics)

    async def _run_ssh_command(self, shell, command: str) -> SSHCapture:
        """Run SSH command; on transport failure populate `transport_error`, else the payload fields."""
        try:
            result = await shell.ssh_client.run(command, timeout=VERIFYX_COMMAND_TIMEOUT_SECONDS)
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
