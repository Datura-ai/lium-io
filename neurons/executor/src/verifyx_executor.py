import argparse
import ctypes
import os
import signal
import sys
from types import FrameType


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

    def close(self) -> None:
        # DAH-2427: a context left behind becomes an orphaned kernel that pins the card at
        # 100% with no process, so release it on every exit we can still run code on — a clean
        # finish, an exception, or a signal. A hang inside the native execute() is NOT one of
        # them (handlers only run between bytecodes); GpuUsageCheck is the backstop there.
        # Idempotent so finally + signal + __del__ can all call it; getattr guards __del__
        # when __init__ raised early.
        if getattr(self, "service", None) is not None:
            self.lib.service_del(self.service)
            self.service = None

    def __del__(self):
        self.close()

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

    def _release_and_exit(signum: int, frame: FrameType | None) -> None:
        executor.close()
        sys.exit(128 + signum)

    # SIGHUP too: a validator-side timeout closes the SSH channel, and OpenSSH signals the
    # remote command with SIGHUP whose default action would kill us before cleanup runs.
    signal.signal(signal.SIGTERM, _release_and_exit)
    signal.signal(signal.SIGINT, _release_and_exit)
    signal.signal(signal.SIGHUP, _release_and_exit)

    try:
        result = executor.execute(args.cipher_text, args.seed)
        print(result)
    finally:
        executor.close()


if __name__ == "__main__":
    main()
