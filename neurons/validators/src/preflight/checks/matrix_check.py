"""Matrix multiplication GPU compute validation check."""

import json
import logging
import random
import time
import uuid
from ctypes import CDLL, c_longlong, POINTER, c_void_p, c_char_p
from typing import Optional, Tuple

from preflight.base import PreflightCheck, CheckResult, CheckStatus
from preflight.utils import suppress_library_output, get_gpu_info

logger = logging.getLogger(__name__)

# --- Domain errors -----------------------------------------------------------

class MatrixValidationError(Exception): ...
class NoGpuError(MatrixValidationError): ...
class LibraryLoadError(MatrixValidationError): ...
class CipherGenError(MatrixValidationError): ...
class ExecuteError(MatrixValidationError): ...
class UuidMismatchError(MatrixValidationError): ...
class DimensionError(MatrixValidationError): ...


# --- Wrapper -----------------------------------------------------------------

class DMCompVerifyWrapper:
    """
    Wrapper for libdmcompverify.so that handles both encryption and decryption.
    """

    def __init__(self, lib_path: str):
        """Initialize the wrapper with the library path."""
        self._lib = CDLL(lib_path)
        self._setup_lib_functions()

    def _setup_lib_functions(self):
        """Set up function signatures for the library."""
        # Create new verifier
        self._lib.DMCompVerify_new.argtypes = [c_longlong, c_longlong]
        self._lib.DMCompVerify_new.restype = POINTER(c_void_p)

        # Set dimensions
        self._lib.setDimension.argtypes = [POINTER(c_void_p), c_longlong, c_longlong]
        self._lib.setDimension.restype = POINTER(c_void_p)

        # Generate challenge (encrypt)
        self._lib.generateChallenge.argtypes = [POINTER(c_void_p), c_longlong, c_char_p, c_char_p]
        self._lib.generateChallenge.restype = None

        # Get cipher text
        self._lib.getCipherText.argtypes = [c_void_p]
        self._lib.getCipherText.restype = c_char_p

        # Process challenge result (decrypt - does GPU matrix multiplication)
        self._lib.processChallengeResult.argtypes = [POINTER(c_void_p), c_longlong, c_char_p]
        self._lib.processChallengeResult.restype = c_char_p

        # Get UUID after processing
        self._lib.getUUID.argtypes = [c_void_p]
        self._lib.getUUID.restype = c_char_p

        # Free memory
        self._lib.free.argtypes = [c_void_p]
        self._lib.free.restype = None

    def create_verifier(self, dim_n: int, dim_k: int) -> POINTER(c_void_p):
        """Create a new DMCompVerify object."""
        return self._lib.DMCompVerify_new(dim_n, dim_k)

    def set_dimension(self, verifier_ptr: POINTER(c_void_p), dim_n: int, dim_k: int):
        """Set dimensions for the verifier."""
        self._lib.setDimension(verifier_ptr, dim_n, dim_k)

    def generate_challenge(self, verifier_ptr: POINTER(c_void_p), seed: int, machine_info: str, challenge_uuid: str):
        """Generate a challenge (encrypt)."""
        machine_info_bytes = machine_info.encode('utf-8')
        uuid_bytes = challenge_uuid.encode('utf-8')
        self._lib.generateChallenge(verifier_ptr, seed, machine_info_bytes, uuid_bytes)

    def get_cipher_text(self, verifier_ptr: POINTER(c_void_p)) -> Optional[str]:
        """Get the generated cipher text."""
        cipher_text_ptr = self._lib.getCipherText(verifier_ptr)
        if cipher_text_ptr:
            cipher_text = c_char_p(cipher_text_ptr).value
            return cipher_text.decode('utf-8')
        return None

    def process_challenge_result(self, verifier_ptr: POINTER(c_void_p), seed: int, cipher_text: str):
        """Process challenge result (decrypt - runs GPU matrix multiplication)."""
        cipher_text_bytes = cipher_text.encode('utf-8')
        self._lib.processChallengeResult(verifier_ptr, seed, cipher_text_bytes)

    def get_uuid(self, verifier_ptr: POINTER(c_void_p)) -> Optional[str]:
        """Get the UUID after processing."""
        uuid_ptr = self._lib.getUUID(verifier_ptr)
        if uuid_ptr:
            uuid_value = c_char_p(uuid_ptr).value
            return uuid_value.decode('utf-8')
        return None

    def free(self, verifier_ptr: POINTER(c_void_p)):
        """Free memory."""
        self._lib.free(verifier_ptr)


# --- Main check --------------------------------------------------------------

class MatrixValidationCheck(PreflightCheck):
    """
    Validates GPU compute capability using matrix multiplication.

    This check proves the machine has actual GPU compute capability,
    not just GPU presence. It:
    1) Generates a challenge (cipher text) based on GPU specs
    2) Runs matrix multiplication on the GPU to decrypt
    3) Verifies the result matches expected output
    """

    def __init__(self, lib_path: str = "/usr/lib/libdmcompverify.so"):
        """
        Initialize the matrix validation check.

        Args:
            lib_path: Path to libdmcompverify.so
        """
        self.lib_path = lib_path

    @property
    def name(self) -> str:
        return "GPU Matrix Multiplication"

    async def run(self) -> CheckResult:
        """Run the matrix multiplication validation in one clear, linear flow."""
        try:
            # 1) Detect GPU
            gpu_info = get_gpu_info(include_memory=True)
            if not gpu_info:
                raise NoGpuError("No GPUs detected for matrix multiplication test")

            # 2) Load native wrapper
            wrapper = self._load_wrapper(self.lib_path)

            # 3) Choose dimensions/seed/uuid
            dim_n, seed, challenge_uuid, dim_k = self._choose_params(gpu_info["gpu_memory_mb"])

            # 4) Generate cipher (encrypt side)
            cipher_text = self._generate_cipher_text(
                wrapper=wrapper,
                gpu_info=gpu_info,
                dim_n=dim_n,
                dim_k=dim_k,
                seed=seed,
                challenge_uuid=challenge_uuid,
            )
            if not cipher_text:
                raise CipherGenError(f"Failed to generate cipher text using {self.lib_path}")

            # 5) Execute (decrypt side) and fetch returned UUID
            returned_uuid = self._execute_and_get_uuid(
                wrapper=wrapper,
                cipher_text=cipher_text,
                seed=seed,
                dim_n=dim_n,
                dim_k=dim_k,
            )
            if not returned_uuid:
                raise ExecuteError("GPU matrix computation completed but failed to extract UUID")

            # 6) Verify
            if returned_uuid != challenge_uuid:
                raise UuidMismatchError(
                    f"GPU computation produced incorrect result - UUID mismatch "
                    f"(expected: {challenge_uuid[:8]}..., got: {returned_uuid[:8] if returned_uuid else 'None'}...)"
                )

            # ✅ Success
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASSED,
                message="GPU compute capability verified via matrix multiplication",
            )

        except MatrixValidationError as e:
            logger.error("Matrix validation failed: %s", e, exc_info=True)
            return CheckResult(
                name=self.name,
                status=CheckStatus.FAILED,
                message=str(e),
            )
        except Exception as e:
            logger.error("Unexpected matrix validation error: %s", e, exc_info=True)
            return CheckResult(
                name=self.name,
                status=CheckStatus.FAILED,
                message=f"Matrix validation encountered unexpected error: {e}",
            )

    # --- Pure helpers (no CheckResult returns) --------------------------------

    def _load_wrapper(self, lib_path: str):
        logger.debug("Loading matrix validation library from %s", lib_path)
        try:
            wrapper = DMCompVerifyWrapper(lib_path)
            logger.debug("Matrix validation library loaded successfully")
            return wrapper
        except Exception as e:
            raise LibraryLoadError(f"Cannot load matrix validation library at {lib_path}: {e}") from e

    def _choose_params(self, gpu_memory_mb: int) -> Tuple[int, int, str, int]:
        """
        Generate matrix dimensions and identifiers.
        Returns: (dim_n, seed, challenge_uuid, dim_k)
        """
        dim_n = random.randint(1900, 2000)
        seed = int(time.time())
        challenge_uuid = str(uuid.uuid4())
        dim_k = self._calculate_max_dim_k(gpu_memory_mb, dim_n)

        if dim_k <= 0:
            raise DimensionError(
                f"Calculated dim_k={dim_k} is not positive for dim_n={dim_n} and gpu_memory_mb={gpu_memory_mb}"
            )
        logger.debug("Chosen params: dim_n=%d dim_k=%d seed=%d uuid=%s", dim_n, dim_k, seed, challenge_uuid)
        return dim_n, seed, challenge_uuid, dim_k

    def _generate_cipher_text(
        self,
        wrapper,
        gpu_info: dict,
        dim_n: int,
        dim_k: int,
        seed: int,
        challenge_uuid: str,
    ) -> str:
        """Generate cipher text (encryption side)."""
        machine_info = json.dumps(
            {
                "uuids": gpu_info["gpu_uuids"],
                "gpu_count": gpu_info["gpu_count"],
                "gpu_model": gpu_info["gpu_model"],
            },
            sort_keys=True,
        )

        with suppress_library_output():
            encrypt_verifier = wrapper.create_verifier(10, 10)
            try:
                wrapper.set_dimension(encrypt_verifier, dim_n, dim_k)
                wrapper.generate_challenge(encrypt_verifier, seed, machine_info, challenge_uuid)
                cipher_text = wrapper.get_cipher_text(encrypt_verifier)
            finally:
                # Ensure native resources are released even on exceptions
                wrapper.free(encrypt_verifier)

        if not cipher_text:
            raise CipherGenError(f"Failed to generate cipher text using {self.lib_path}")

        logger.debug("Cipher text generated (preview): %s", cipher_text[:50])
        return cipher_text

    def _execute_and_get_uuid(
        self,
        wrapper,
        cipher_text: str,
        seed: int,
        dim_n: int,
        dim_k: int,
    ) -> str:
        """Execute the challenge (GPU matrix multiplication) and return the UUID."""
        with suppress_library_output():
            decrypt_verifier = wrapper.create_verifier(dim_n, dim_k)
            try:
                wrapper.process_challenge_result(decrypt_verifier, seed, cipher_text)
                returned_uuid = wrapper.get_uuid(decrypt_verifier)
            except Exception as e:
                raise ExecuteError(f"GPU matrix computation failed during processing: {e}") from e
            finally:
                wrapper.free(decrypt_verifier)

        logger.debug("Returned UUID: %s", returned_uuid)
        return returned_uuid

    def _calculate_max_dim_k(self, gpu_memory_mb: int, dim_n: int) -> int:
        """
        Calculate maximum dimension k based on GPU memory.
        Reserve ~2GB for safety and compute a conservative k.
        """
        # Reserve 2GB
        gpu_memory_adjusted = max(0, gpu_memory_mb - 2 * 1024)
        max_memory = gpu_memory_adjusted * (1024.0 ** 2)
        element_size = 8  # 8 bytes for double precision
        if max_memory <= 0:
            return 0

        max_elements = max_memory // element_size
        # Memory for two matrices of size (dim_n x (dim_n + dim_k)) -> solve for dim_k
        dim_k = int(max_elements // (2 * dim_n) - dim_n)

        # Guardrail: ensure dim_k doesn't explode or go negative
        dim_k = max(1, min(dim_k, 8192))
        return dim_k
