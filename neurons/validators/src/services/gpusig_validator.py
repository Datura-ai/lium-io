"""Thin ctypes wrapper around libgpusig.so — validator side.

Canonical source; published into Datura-ai/lium-io alongside the compiled
libgpusig.so. It exposes the verifier's C ABI and nothing about how the seal is
computed: all crypto lives in the compiled, obfuscated libgpusig.so.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any

# Where the validator image installs the verifier (its Dockerfile moves the
# published file to /usr/lib, next to libverifyx.so / libdmcompverify.so).
DEFAULT_LIB_PATH = "/usr/lib/libgpusig.so"

# The C ABI's output contract (GPUSIG_OUT_MAX in gpusig_seal.h): the verifier
# returns a NUL-terminated string shorter than this many bytes. The wrapper never
# reads past it, so a corrupted or unterminated native buffer cannot make
# ctypes walk the heap.
OUT_MAX = 1024


class GpuSigVerifier:
    def __init__(self, lib_path: str | None = None):
        self.lib = ctypes.CDLL(lib_path or DEFAULT_LIB_PATH)
        self._setup_signatures()

    def _setup_signatures(self) -> None:
        self.lib.gpusig_verify_seal.argtypes = [
            ctypes.c_char_p,  # master_key
            ctypes.c_char_p,  # expected_nonce
            ctypes.c_char_p,  # msg (the signed record, opaque to the caller)
            ctypes.c_char_p,  # sig (hex)
        ]
        self.lib.gpusig_verify_seal.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.gpusig_str_free.argtypes = [ctypes.POINTER(ctypes.c_char)]

    def _take(self, ptr) -> str:
        if not ptr:
            raise RuntimeError("libgpusig call returned null")
        try:
            # Bounded read, one byte at a time: stop at the first NUL and never
            # touch a byte past OUT_MAX (no unsized string_at, no block read
            # beyond the allocation).
            out = bytearray()
            for i in range(OUT_MAX):
                b = ptr[i]
                if b == b"\0":
                    return out.decode("utf-8")
                out += b
            raise ValueError("libgpusig returned an unterminated buffer")
        finally:
            self.lib.gpusig_str_free(ptr)

    def verify_seal(
        self, master_key: str, expected_nonce: str, msg: str, sig: str
    ) -> dict[str, Any]:
        """Authenticate one sealed per-card response.

        Returns the verifier's verdict dict. ``sealed`` is True only when the
        seal recomputes, the scheme tag matches, the signed nonce equals
        ``expected_nonce`` and the record carries a kernel identity; then
        ``tflops``/``gbps``/``kernel_uuid``/``pci``/``device``/``vram_mb`` are
        the AUTHENTICATED values parsed from the signed record. Otherwise
        ``sealed`` is False with a ``reason``.
        """
        ptr = self.lib.gpusig_verify_seal(
            master_key.encode("utf-8"),
            expected_nonce.encode("utf-8"),
            msg.encode("utf-8"),
            sig.encode("utf-8"),
        )
        body = self._take(ptr)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"libgpusig returned non-JSON: {exc}") from exc
