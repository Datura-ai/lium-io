import ctypes
import hashlib
import json
import random
import os
import logging
from typing import Dict, Any, Optional
from core.utils import _m, get_extra_info


logger = logging.getLogger(__name__)


class VerifyXValidator:
    def __init__(self, lib_name: str, seed: int):
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self.service = self._create_service()
        self.seed = seed

    def _setup_signatures(self):
        self.lib.service_new.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.generate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_char_p]
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

    def generate_challenge(self, info=None) -> str:
        if self.lib.generate(self.service, self.seed, info) != 0:
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

    async def _get_executor_checksum(self, ssh_client, executor_info) -> str:
        """Get the VerifyX library checksum from executor."""
        command = f"{executor_info.python_path} {executor_info.root_dir}/src/checksum_executor.py"

        try:
            result = await ssh_client.run(command)
            return result.stdout.strip() if result else ""
        except Exception:
            return ""

    async def validate_verifyx_and_process_job(
        self,
        ssh_client,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
    ):
        try:
            # Verify checksum before proceeding with validation
            local_checksum = self._calculate_lib_checksum(self.lib_name)
            executor_checksum = await self._get_executor_checksum(ssh_client, executor_info)

            logger.info(f"local_checksum: {local_checksum}, executor_checksum: {executor_checksum}")

            if local_checksum != executor_checksum:
                return VerifyXResponse(error="executor not using latest VerifyX library (checksum verification failed)")

            gpu_details = machine_spec.get("gpu", {}).get("details", [])
            gpu_count = machine_spec.get("gpu", {}).get("count", 0)
            gpu_uuids = ",".join([detail.get("uuid", "") for detail in gpu_details])
            gpu_model = gpu_details[0].get("name", "") if gpu_details else ""

            gpu_info = {"uuids": gpu_uuids, "gpu_count": gpu_count, "gpu_model": gpu_model}
            machine_info = json.dumps(gpu_info, sort_keys=True)

            seed = random.getrandbits(64)
            verifyx_validator = VerifyXValidator(self.lib_name, seed)
            cipher_text = verifyx_validator.generate_challenge(machine_info.encode("utf-8"))

            command = f"{executor_info.python_path} {executor_info.root_dir}/src/verifyx_executor.py --seed {seed} --cipher_text {cipher_text}"

            log_extra = {
                **default_extra,
                "seed": seed,
                "cipher_text": cipher_text,
                "machine_info": machine_info,
            }

            logger.info(_m("VerifyX Python Script Command", extra=get_extra_info(log_extra)))

            try:
                result = await ssh_client.run(command)
            except Exception:
                return VerifyXResponse(error="SSH command execution failed")

            if result is None:
                return VerifyXResponse(error="SSH command returned no result")

            try:
                stdout = result.stdout.strip()
            except AttributeError:
                return VerifyXResponse(error="SSH result missing stdout")

            logger.info(_m("Challenge response received", extra=get_extra_info({**log_extra, "stdout": stdout})))
            # validate the response
            try:
                response = verifyx_validator.verify_response(stdout)
                return VerifyXResponse(data=response)
            except Exception as e:
                return VerifyXResponse(error=f"challenge verification failed ({str(e)})")

        except Exception as e:
            return VerifyXResponse(error=f"unexpected error ({str(e)})")
