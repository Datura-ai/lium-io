import ctypes
import hashlib
import json
import random
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
from core.utils import _m, get_extra_info

from services.const import (
    MEMORY_ALLOCATION_PERCENTAGE,
    MEMORY_MIN_TEST_GB,
    MEMORY_MAX_TEST_GB,
    STORAGE_MIN_AVAILABLE_GB,
    STORAGE_THROUGHPUT_TEST_GB,
    NETWORK_TIMEOUT_SECONDS,
    NETWORK_MIN_DOWNLOAD_SPEED_MBPS,
)


logger = logging.getLogger(__name__)

GB_TO_BYTES = 1024 * 1024 * 1024


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


class VerifyXResponse:
    def __init__(self, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.data = data
        self.error = error


class VerifyXValidationService:
    def __init__(self):
        self.lib_name = "/usr/lib/libverifyx.so"

    def _calculate_lib_checksum(self, lib_path: str) -> str:
        """Calculate SHA256 checksum of the VerifyX shared library."""
        with open(lib_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    async def _get_executor_checksum(self, shell) -> str:
        """Get the VerifyX library checksum from executor using SCP."""
        try:
            checksums = await shell.get_checksums_over_scp(self.lib_name)
            # Format is "md5:sha256", we need sha256
            sha256_checksum = checksums.split(":")[1]
            return sha256_checksum
        except Exception:
            return ""

    async def validate_verifyx_and_process_job(
        self,
        shell,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
    ):
        try:
            # Verify checksum before proceeding with validation
            local_checksum = self._calculate_lib_checksum(self.lib_name)
            executor_checksum = await self._get_executor_checksum(shell)

            if local_checksum != executor_checksum:
                return VerifyXResponse(error="executor not using latest VerifyX library (checksum verification failed)")

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
                    "memory_allocation_percentage": MEMORY_ALLOCATION_PERCENTAGE,
                    "memory_min_test_gb": MEMORY_MIN_TEST_GB,
                    "memory_max_test_gb": MEMORY_MAX_TEST_GB,
                    "storage_min_available_gb": STORAGE_MIN_AVAILABLE_GB,
                    "storage_throughput_test_gb": STORAGE_THROUGHPUT_TEST_GB,
                    "network_timeout_seconds": NETWORK_TIMEOUT_SECONDS,
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

            try:
                result = await shell.ssh_client.run(command)
            except Exception:
                return VerifyXResponse(error="SSH command execution failed")

            if result is None:
                return VerifyXResponse(error="SSH command returned no result")

            try:
                stdout = result.stdout.strip()
            except AttributeError:
                return VerifyXResponse(error="SSH result missing stdout")

            logger.info(_m("Challenge response received", extra=get_extra_info(log_extra)))

            try:
                payload = verifyx_validator.verify_response(stdout)
                verification_result = _perform_verification_checks(payload)
                return VerifyXResponse(data=verification_result)
            except Exception as e:
                return VerifyXResponse(error=f"challenge verification failed ({str(e)})")

        except Exception as e:
            return VerifyXResponse(error=f"unexpected error ({str(e)})")


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

    speedtest = network_execution["speedtest"]
    if speedtest["download_mbps"] <= 0.0 or speedtest["upload_mbps"] <= 0.0:
        errors.append("Network performance data unavailable")
        success = False

    download_speeds = [speedtest["download_mbps"], network_execution["download"]["speed_mbps"]]
    average_download_speed = sum(download_speeds) / len(download_speeds)

    if average_download_speed < NETWORK_MIN_DOWNLOAD_SPEED_MBPS:
        errors.append(
            f"Network download speed inadequate: {average_download_speed:.2f} Mbps achieved, {NETWORK_MIN_DOWNLOAD_SPEED_MBPS:.0f} Mbps required"
        )
        success = False

    stats = {
        "upload_speed": speedtest["upload_mbps"],
        "download_speed": average_download_speed,
        "success": success,
        "execution_time_ms": network_execution["execution_time_ms"],
    }

    return stats, errors


def _verify_speed_test(challenge_data: dict, response_data: dict) -> Tuple[dict, List[str]]:
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


def _perform_verification_checks(payload: dict) -> Dict[str, Any]:
    challenge_data = payload["challenge_data"]
    response_data = payload["response_data"]

    network_stats, network_errors = _verify_network_test(challenge_data, response_data)
    memory_stats, memory_errors = _verify_memory_test(challenge_data, response_data)
    storage_stats, storage_errors = _verify_speed_test(challenge_data, response_data)
    all_errors = network_errors + memory_errors + storage_errors

    success = network_stats["success"] and memory_stats["success"] and storage_stats["success"]

    return {
        "success": success,
        "network": network_stats,
        "hard_disk": storage_stats,
        "ram": memory_stats,
        "errors": all_errors,
    }
