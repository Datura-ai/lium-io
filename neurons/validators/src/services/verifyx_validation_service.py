import ctypes
import json
import random
import os
import logging
from typing import Dict, Any
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)


class VerifyXValidator:
    def __init__(self, lib_name: str, seed: int):
        self._initialized = False
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
        self._initialized = True

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


class VerifyXValidationService:
    def __init__(self):
        self.lib_name = "/usr/lib/libverifyx.so"

    async def validate_verifyx_and_process_job(
        self,
        ssh_client,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
    ):
        try:
            script_path = f"{executor_info.root_dir}/src/verifyx_executor.py"

            gpu_details = machine_spec.get("gpu", {}).get("details", [])
            gpu_count = machine_spec.get("gpu", {}).get("count", 0)
            gpu_uuids = ",".join([detail.get("uuid", "") for detail in gpu_details])
            gpu_model = gpu_details[0].get("name", "") if gpu_details else ""

            gpu_info = {"uuids": gpu_uuids, "gpu_count": gpu_count, "gpu_model": gpu_model}
            machine_info = json.dumps(gpu_info, sort_keys=True)

            seed = random.getrandbits(64)
            verifyx_validator = VerifyXValidator(self.lib_name, seed)
            cipher_text = verifyx_validator.generate_challenge(machine_info.encode("utf-8"))

            command = f"{executor_info.python_path} {script_path} --seed {seed} --cipher_text {cipher_text}"

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
                logger.error(_m("Failed to execute SSH command", extra=get_extra_info(log_extra)))
                return None

            if result is None:
                logger.warning(_m("VerifyX validation job failed", extra=get_extra_info(log_extra)))
                return None

            try:
                stdout = result.stdout.strip()
            except AttributeError:
                logger.error(_m("Result object missing stdout attribute", extra=get_extra_info(log_extra)))
                return None

            # validate the response
            try:
                response = verifyx_validator.verify_response(stdout)
                logger.info(_m("VerifyX Verification Succeed", extra=get_extra_info({**log_extra, "result": response})))
                return response
            except Exception:
                logger.error(_m("Error during VerifyX verification", extra=get_extra_info(log_extra)))
                return None

        except Exception:
            logger.error(_m("Unexpected error in validate_verifyx_and_process_job", extra=default_extra))
            return None
