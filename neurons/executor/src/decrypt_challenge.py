import os
import base64
import argparse
from ctypes import CDLL, c_longlong, POINTER, c_int, c_void_p, c_char_p

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
        self._lib.processChallengeResult.restype = c_int

        self._lib.getUUID.argtypes = [c_void_p]
        self._lib.getUUID.restype = c_int

        self._lib.getCipherText.argtypes = [c_void_p]
        self._lib.getCipherText.restype = c_int

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
        result = self._lib.processChallengeResult(verifier_ptr, seed, cipher_text)
        return result

    def getCipherText(self, verifier_ptr: POINTER(c_void_p)) -> str:
        """
        Wrap the C++ function getCipherText.
        Retrieves the cipher text as a string.
        """
        # Fetch the cipher text from the C++ function (assumes it's returned as a char*).
        cipher_text_ptr = self._lib.getCipherText(verifier_ptr)

        if cipher_text_ptr:
            cipher_text = c_char_p(cipher_text_ptr).value  # Decode the C string
            return cipher_text
        else:
            return None

    def getUUID(self, verifier_ptr: POINTER(c_void_p)) -> str:
        """
        Wrap the C++ function getUUID.
        Retrieves the UUID as a string.
        """
         # Extract the pointer returned by the C++ function, and convert it to a C string (char*) using c_char_p
        cipher_text_ptr = self._lib.getUUID(verifier_ptr)

        if cipher_text_ptr:
            cipher_text = c_char_p(cipher_text_ptr).value  # Decode the C string
            return cipher_text.decode('utf-8')
        else:
            return None

    def free(self, ptr: c_void_p):
        """
        Frees memory allocated for the given pointer.
        """
        self._lib.free(ptr)

def decrypt_challenge():
    parser = argparse.ArgumentParser(description="DMCompVerify Python Wrapper")
    parser.add_argument("--lib", type=str, default="libdmcompverify.so", help="Path to the shared library")
    parser.add_argument("--m_dim_n", type=int, default=1024, help="Matrix dimension n")
    parser.add_argument("--m_dim_k", type=int, default=2043345, help="Matrix dimension k")
    parser.add_argument("--seed", type=int, default=1234567890, help="Random seed")
    parser.add_argument("--cipher_text", type=str, default="IjBR7kbPvwFfxImr+M/f0lsKvgNoYb3RenOe4l12nzI79R9z", help="Cipher Text")

    args = parser.parse_args()
    
    # Example of usage:
    wrapper = DMCompVerifyWrapper(args.lib)

    # Create a new DMCompVerify object
    verifier_ptr = wrapper.DMCompVerify_new(args.m_dim_n, args.m_dim_k)
    
    decoded_binary = base64.b64decode(args.cipher_text)

    # Example of processing challenge result
    wrapper.processChallengeResult(verifier_ptr, args.seed, decoded_binary)

    # Example to get the UUID
    uuid = wrapper.getUUID(verifier_ptr)
    print(uuid)

    # Free resources
    wrapper.free(verifier_ptr)

    return uuid

if __name__ == "__main__":
    decrypt_challenge()