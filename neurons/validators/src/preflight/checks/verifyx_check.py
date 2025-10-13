"""VerifyX validation check for RAM, storage, and network."""

import json
import logging
import random
import ctypes
from dataclasses import dataclass
from ctypes import CDLL, POINTER, c_void_p, c_char_p, c_char, c_uint64
from typing import Optional

from preflight.base import PreflightCheck, CheckResult, CheckStatus
from preflight.utils import suppress_library_output, get_gpu_info
from preflight.constants import (
    MEMORY_ALLOCATION_PERCENTAGE,
    MEMORY_MIN_TEST_GB,
    MEMORY_MAX_TEST_GB,
    STORAGE_MIN_AVAILABLE_GB,
    STORAGE_THROUGHPUT_TEST_GB,
    NETWORK_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# --- Domain errors -----------------------------------------------------------

class VerifyXError(Exception): ...
class NoGpuError(VerifyXError): ...
class LibraryLoadError(VerifyXError): ...
class ChallengeGenError(VerifyXError): ...
class CipherFetchError(VerifyXError): ...
class ExecuteError(VerifyXError): ...
class VerificationError(VerifyXError): ...

# --- Config ------------------------------------------------------------------

@dataclass(frozen=True)
class VerifyXConfig:
    memory_allocation_percentage: int
    memory_min_test_gb: int
    memory_max_test_gb: int
    storage_min_available_gb: int
    storage_throughput_test_gb: int
    network_timeout_seconds: int

# --- Wrapper ------------------------------------------------------------------

class VerifyXWrapper:
    """
    Wrapper for libverifyx.so that handles challenge generation, execution, and verification.
    """

    def __init__(self, lib_path: str):
        """Initialize the wrapper with the library path."""
        self.lib = CDLL(lib_path)
        self._setup_signatures()
        self.service = self._create_service()

    def _setup_signatures(self):
        """Set up function signatures for the library."""
        self.lib.service_new.restype = POINTER(c_void_p)

        # Generate challenge
        self.lib.generate.argtypes = [POINTER(c_void_p), c_char_p]
        self.lib.generate.restype = int

        # Get cipher text
        self.lib.get_cipher_text.argtypes = [POINTER(c_void_p)]
        self.lib.get_cipher_text.restype = POINTER(c_char)

        # Execute (does the actual RAM/storage/network tests)
        self.lib.execute.argtypes = [POINTER(c_void_p), c_char_p, c_uint64]
        self.lib.execute.restype = POINTER(c_char)

        # Verify result
        self.lib.verify.argtypes = [POINTER(c_void_p), c_char_p, c_uint64]
        self.lib.verify.restype = POINTER(c_char)

        # Cleanup
        self.lib.service_del.argtypes = [POINTER(c_void_p)]
        self.lib.str_del.argtypes = [POINTER(c_char)]

    def _create_service(self):
        """Create a new VerifyX service."""
        return self.lib.service_new()

    def __del__(self):
        """Cleanup service."""
        self.lib.service_del(self.service)

    def _decode_string(self, ptr):
        """Decode a C string pointer."""
        return ctypes.string_at(ptr).decode("utf-8") if ptr else None

    def generate_challenge(self, challenge_input: dict) -> bool:
        """
        Generate a challenge from input configuration.

        Returns:
            True if successful, False otherwise
        """
        challenge_input_json = json.dumps(challenge_input).encode("utf-8")
        result = self.lib.generate(self.service, challenge_input_json)
        return result == 0

    def get_cipher_text(self) -> Optional[str]:
        """Get the generated cipher text."""
        cipher_ptr = self.lib.get_cipher_text(self.service)
        cipher_hex = self._decode_string(cipher_ptr)
        self.lib.str_del(cipher_ptr)
        return cipher_hex

    def execute(self, cipher_hex: str, seed: int) -> Optional[str]:
        """
        Execute the challenge (runs RAM/storage/network tests).

        Returns:
            Result cipher text or None if failed
        """
        result_ptr = self.lib.execute(self.service, cipher_hex.encode("utf-8"), seed)
        result_cipher_hex = self._decode_string(result_ptr)
        self.lib.str_del(result_ptr)
        return result_cipher_hex

    def verify(self, response_cipher_hex: str, seed: int) -> Optional[dict]:
        """
        Verify the response.

        Returns:
            Verification result dict or None if failed
        """
        verify_ptr = self.lib.verify(self.service, response_cipher_hex.encode("utf-8"), seed)
        if not verify_ptr:
            return None

        verify_result = self._decode_string(verify_ptr)
        self.lib.str_del(verify_ptr)

        if not verify_result:
            return None

        try:
            return json.loads(verify_result)
        except json.JSONDecodeError:
            return None

# --- Main class --------------------------------------------------------------

class VerifyXCheck(PreflightCheck):
    """
    Validates RAM, storage, and network capabilities.

    This check:
    1. Tests RAM allocation and availability
    2. Tests storage space and throughput
    3. Tests network download speed and integrity
    """

    def __init__(
        self,
        lib_path: str = "/usr/lib/libverifyx.so",
        memory_allocation_percentage: int = MEMORY_ALLOCATION_PERCENTAGE,
        memory_min_test_gb: int = MEMORY_MIN_TEST_GB,
        memory_max_test_gb: int = MEMORY_MAX_TEST_GB,
        storage_min_available_gb: int = STORAGE_MIN_AVAILABLE_GB,
        storage_throughput_test_gb: int = STORAGE_THROUGHPUT_TEST_GB,
        network_timeout_seconds: int = NETWORK_TIMEOUT_SECONDS,
    ):
        """
        Initialize the VerifyX check.

        Args:
            lib_path: Path to libverifyx.so
            memory_allocation_percentage: Percentage of RAM to allocate for testing
            memory_min_test_gb: Minimum RAM required (GB)
            memory_max_test_gb: Maximum RAM to test (GB)
            storage_min_available_gb: Minimum storage space required (GB)
            storage_throughput_test_gb: Size of data to test storage throughput (GB)
            network_timeout_seconds: Timeout for network tests
        """
        self.lib_path = lib_path
        self.memory_allocation_percentage = memory_allocation_percentage
        self.memory_min_test_gb = memory_min_test_gb
        self.memory_max_test_gb = memory_max_test_gb
        self.storage_min_available_gb = storage_min_available_gb
        self.storage_throughput_test_gb = storage_throughput_test_gb
        self.network_timeout_seconds = network_timeout_seconds

    @property
    def name(self) -> str:
        return "VerifyX (RAM/Storage/Network)"

    async def run(self) -> CheckResult:
        """Run VerifyX validation in one clear, linear flow."""
        try:
            # 1) Detect GPU
            gpu_info = get_gpu_info(include_memory=True)
            if not gpu_info:
                raise NoGpuError("No GPUs detected for verifyx test")

            # 2) Load VerifyX wrapper
            wrapper = self._load_wrapper(self.lib_path)

            # 3) Build challenge input
            seed = random.getrandbits(64)
            logger.debug("Generated seed: %s", seed)
            challenge_input = {
                "seed": seed,
                "machine_info": gpu_info,
                "config": {
                    "memory_allocation_percentage": self.memory_allocation_percentage,
                    "memory_min_test_gb": self.memory_min_test_gb,
                    "memory_max_test_gb": self.memory_max_test_gb,
                    "storage_min_available_gb": self.storage_min_available_gb,
                    "storage_throughput_test_gb": self.storage_throughput_test_gb,
                    "network_timeout_seconds": self.network_timeout_seconds,
                },
            }

            # 4) Generate challenge
            logger.debug("generate_challenge input:\n%s", json.dumps(challenge_input, indent=2))
            if not wrapper.generate_challenge(challenge_input):
                raise ChallengeGenError("Library returned non-zero from generate_challenge")

            # 5) Get cipher text
            with suppress_library_output():
                cipher_text = wrapper.get_cipher_text()
            if not cipher_text:
                raise CipherFetchError("Failed to get cipher text from VerifyX challenge")

            # 6) Execute + verify
            with suppress_library_output():
                logger.debug("Executing VerifyX tests with cipher preview: %s", cipher_text[:50])
                result_cipher = wrapper.execute(cipher_text, seed)
                if not result_cipher:
                    raise ExecuteError("VerifyX tests failed to execute")

                verification = wrapper.verify(result_cipher, seed)
            if not verification:
                raise VerificationError("VerifyX verify() returned no data")

            logger.debug("Verification result:\n%s", json.dumps(verification, indent=2))

            # 7) Analyze results
            errors = self._collect_errors(verification)
            if errors:
                logger.debug("VerifyX validation failed: %s", errors)
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message="VerifyX validation failed: " + "; ".join(errors),
                )

            # ✅ Success
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASSED,
                message="RAM, storage, and network validation passed",
            )

        except VerifyXError as e:
            logger.error("VerifyX failed: %s", e, exc_info=True)
            return CheckResult(
                name=self.name,
                status=CheckStatus.FAILED,
                message=str(e),
            )
        except Exception as e:
            logger.error("Unexpected VerifyX error: %s", e, exc_info=True)
            return CheckResult(
                name=self.name,
                status=CheckStatus.FAILED,
                message=f"Unexpected error: {e}",
            )

    # --- Helpers --------------------------------------------------------------

    def _load_wrapper(self, lib_path: str):
        """Load the VerifyX shared library wrapper."""
        logger.debug("Loading VerifyX library from %s", lib_path)
        try:
            wrapper = VerifyXWrapper(lib_path)
            logger.debug("VerifyX library loaded successfully")
            return wrapper
        except Exception as e:
            raise LibraryLoadError(f"Cannot load VerifyX library at {lib_path}: {e}") from e

    def _collect_errors(self, verification: dict) -> list[str]:
        """Return human-readable list of errors (empty = pass)."""
        response_data = verification.get("response_data", {})
        errors = []

        net = response_data.get("network_execution", {})
        if not net.get("success", False):
            errors.append(f"Network: {net.get('error', 'Network test failed')}")

        mem = response_data.get("memory_execution", {})
        if not mem.get("success", False):
            errors.append(f"RAM: {mem.get('error', 'Memory test failed')}")

        stg = response_data.get("storage_execution", {})
        if stg.get("error"):
            errors.append(f"Storage: {stg['error']}")

        return errors
