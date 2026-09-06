import os
import json
import argparse
from ctypes import CDLL, c_longlong, POINTER, c_void_p, c_char_p

# Opt-in to the TFLOPS benchmark for THIS process only. The shared library reads
# TFLOPS_ENABLE at solve time and runs the throughput benchmark after the challenge
# is solved. Validator-side in-process callers of the same .so (e.g. the preflight
# matrix_check) never set this, so they pay zero benchmark cost.
os.environ.setdefault("TFLOPS_ENABLE", "1")


class DMCompVerifyWrapper:
    def __init__(self, lib_name: str):
        """
        Constructor, differentiate miner vs validator libs.
        """
        self._initialized = False
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        self._lib = CDLL(lib_path)
        self._setup_lib_functions()

    def _setup_lib_functions(self):
        """
        Set up function signatures for the library.
        """
        # Set up function signatures for the library.
        self._lib.DMCompVerify_new.argtypes = [c_longlong, c_longlong]  # Parameters (long m_dim_n, long m_dim_k)
        self._lib.DMCompVerify_new.restype = POINTER(c_void_p)  # Return type is a pointer to a structure.

        self._lib.generateChallenge.argtypes = [POINTER(c_void_p), c_longlong, c_char_p, c_char_p]
        self._lib.generateChallenge.restype = None

        self._lib.processChallengeResult.argtypes = [POINTER(c_void_p), c_longlong, c_char_p]
        self._lib.processChallengeResult.restype = c_char_p

        self._lib.getUUID.argtypes = [c_void_p]
        self._lib.getUUID.restype = c_char_p

        # getMetrics is optional: an older .so without the symbol must still load.
        self._has_metrics = hasattr(self._lib, "getMetrics")
        if self._has_metrics:
            self._lib.getMetrics.argtypes = [c_void_p]
            self._lib.getMetrics.restype = c_char_p

        # getSealedResult is optional too: an older .so without it must still load.
        # When present, it returns an authenticated-encrypted {uuid, metrics} blob the
        # validator decrypts itself, so this (miner-controlled) wrapper cannot forge or
        # inflate the result by swapping the printed line.
        self._has_sealed = hasattr(self._lib, "getSealedResult")
        if self._has_sealed:
            self._lib.getSealedResult.argtypes = [c_void_p]
            self._lib.getSealedResult.restype = c_char_p

        self._lib.free.argtypes = [c_void_p]
        self._lib.free.restype = None

        self._initialized = True

    def DMCompVerify_new(self, m_dim_n: int, m_dim_k: int):
        """
        Wrap the C++ function DMCompVerify_new.
        Creates a new DMCompVerify object in C++.
        """
        return self._lib.DMCompVerify_new(m_dim_n, m_dim_k)

    def generateChallenge(self, verifier_ptr: POINTER(c_void_p), seed: int, machine_info: str, uuid: str):
        """
        Wrap the C++ function generateChallenge.
        Generates a challenge using the provided DMCompVerify pointer.
        """
        machine_info_bytes = machine_info.encode('utf-8')
        uuid_bytes = uuid.encode('utf-8')
        self._lib.generateChallenge(verifier_ptr, seed, machine_info_bytes, uuid_bytes)

    def processChallengeResult(self, verifier_ptr: POINTER(c_void_p), seed: int, cipher_text: str) -> int:
        """
        Wrap the C++ function processChallengeResult.
        Processes the challenge result using the provided DMCompVerify pointer.
        """
        self._lib.processChallengeResult(verifier_ptr, seed, cipher_text)

    def getUUID(self, verifier_ptr: POINTER(c_void_p)) -> str:
        """
        Wrap the C++ function getUUID.
        Retrieves the UUID as a string.
        """
         # Extract the pointer returned by the C++ function, and convert it to a C string (char*) using c_char_p
        uuid_ptr = self._lib.getUUID(verifier_ptr)

        if uuid_ptr:
            uuid = c_char_p(uuid_ptr).value  # Decode the C string
            return uuid.decode('utf-8')
        else:
            return None

    def getMetrics(self, verifier_ptr: POINTER(c_void_p)) -> str:
        """
        Wrap the C++ function getMetrics. Returns the TFLOPS metrics JSON string
        (or None if the symbol is absent / the benchmark did not run).
        """
        if not getattr(self, "_has_metrics", False):
            return None
        metrics_ptr = self._lib.getMetrics(verifier_ptr)
        if metrics_ptr:
            metrics = c_char_p(metrics_ptr).value
            return metrics.decode('utf-8')
        return None

    def getSealedResult(self, verifier_ptr: POINTER(c_void_p)) -> str:
        """
        Wrap the C++ function getSealedResult. Returns the authenticated-encrypted
        {uuid, metrics} blob (hex) that the validator decrypts, or None if the symbol
        is absent / no UUID was recovered.
        """
        if not getattr(self, "_has_sealed", False):
            return None
        sealed_ptr = self._lib.getSealedResult(verifier_ptr)
        if sealed_ptr:
            sealed = c_char_p(sealed_ptr).value
            sealed = sealed.decode('utf-8') if sealed else ""
            return sealed or None
        return None

    def free(self, ptr: c_void_p):
        """
        Frees memory allocated for the given pointer.
        """
        self._lib.free(ptr)

def decrypt_challenge():
    parser = argparse.ArgumentParser(description="DMCompVerify Python Wrapper")
    parser.add_argument("--lib", type=str, default="/usr/lib/libdmcompverify.so", help="Path to the shared library")
    parser.add_argument("--dim_n", type=int, default=1981, help="Matrix dimension n")
    parser.add_argument("--dim_k", type=int, default=1555929, help="Matrix dimension k")
    parser.add_argument("--seed", type=int, default=1743502434, help="Random seed")
    parser.add_argument("--cipher_text", type=str, default="e28702c2f187f34d56744d64a4399e00cbecbde2d3f6ca53a8abec5cbc40481d42a1a505", help="Cipher Text")

    args = parser.parse_args()
    
    # Example of usage:
    wrapper = DMCompVerifyWrapper(args.lib)

    # Create a new DMCompVerify object
    verifier_ptr = wrapper.DMCompVerify_new(args.dim_n, args.dim_k)

    # Example of processing challenge result
    wrapper.processChallengeResult(verifier_ptr, args.seed, args.cipher_text.encode('utf-8'))

    # Example to get the UUID
    uuid = wrapper.getUUID(verifier_ptr)
    # Legacy line — kept byte-for-byte so older validators (which scan for a line
    # starting with "UUID:") keep working during independent deploys.
    print("UUID: ", uuid)

    # Combined result: the decrypted UUID plus best-effort TFLOPS metrics. Assembled
    # in Python (guaranteed valid one-line JSON) behind a stable marker the validator
    # greps for. Metrics is null when the benchmark was disabled or failed.
    metrics_raw = wrapper.getMetrics(verifier_ptr)
    try:
        metrics = json.loads(metrics_raw) if metrics_raw else None
    except (ValueError, TypeError):
        metrics = None

    # The sealed blob is the AUTHORITATIVE result: the validator decrypts it and reads
    # uuid+metrics from inside, so this wrapper cannot forge or inflate them. The legacy
    # uuid/metrics fields are kept ONLY for validators that predate sealing; a
    # sealing-aware validator ignores them and trusts only the sealed blob.
    sealed = wrapper.getSealedResult(verifier_ptr)
    print("RESULT_JSON: " + json.dumps({"uuid": uuid, "metrics": metrics, "sealed": sealed}))

    # Free resources
    wrapper.free(verifier_ptr)

    return uuid

if __name__ == "__main__":
    decrypt_challenge()