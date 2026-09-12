"""Thin ctypes wrapper around libgpusig.so — validator side.

Published from the private celium-gpu-verifier repo. It exposes the verifier's
C ABI and nothing about how the seal is computed: all crypto lives in the
compiled, obfuscated libgpusig.so shipped alongside it. Keep this file in sync
with celium-gpu-verifier/gpusig/gpusig_validator.py.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any


class GpuSigVerifier:
    def __init__(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = Path(__file__).parent / "libgpusig.so"
        self.lib = ctypes.CDLL(str(lib_path))
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
            return ctypes.string_at(ptr).decode("utf-8")
        finally:
            self.lib.gpusig_str_free(ptr)

    def verify_seal(
        self, master_key: str, expected_nonce: str, msg: str, sig: str
    ) -> dict[str, Any]:
        """Authenticate one sealed per-card response.

        Returns the verifier's verdict dict. ``sealed`` is True only when the
        seal recomputes, the scheme tag matches and the signed nonce equals
        ``expected_nonce``; then ``tflops``/``gbps``/``kernel_uuid``/``pci``/
        ``device``/``vram_mb`` are the AUTHENTICATED values parsed from the
        signed record. Otherwise ``sealed`` is False with a ``reason``.
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
