import argparse
import ctypes
import os


class VerifyXExecutor:
    def __init__(self, lib_name: str):
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        self.service = self._create_service()

    def _setup_signatures(self):
        self.lib.service_new.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.execute.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_uint64]
        self.lib.execute.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.service_del.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.str_del.argtypes = [ctypes.POINTER(ctypes.c_char)]

    def _create_service(self):
        return self.lib.service_new()

    def __del__(self):
        self.lib.service_del(self.service)

    def _decode_string(self, ptr):
        return ctypes.string_at(ptr).decode("utf-8") if ptr else None

    def execute(self, cipher_hex: str, seed: int) -> str:
        result_ptr = self.lib.execute(self.service, cipher_hex.encode("utf-8"), seed)
        result_cipher_hex = self._decode_string(result_ptr)
        self.lib.str_del(result_ptr)
        return result_cipher_hex


def main():
    parser = argparse.ArgumentParser(description="VerifyXExecutor")
    parser.add_argument("--lib", type=str, default="/usr/lib/libverifyx.so", help="Path to the shared library")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--cipher_text", type=str, required=True, help="Cipher text")

    args = parser.parse_args()

    executor = VerifyXExecutor(args.lib)
    result = executor.execute(args.cipher_text, args.seed)
    print(result)


if __name__ == "__main__":
    main()
