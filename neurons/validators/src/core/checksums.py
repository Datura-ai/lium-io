from __future__ import annotations

import asyncio
import hashlib
import os


def sha256_from_path(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


async def sha256_from_executor(shell, file_path: str, *, max_retries: int = 2) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            checksum = await shell.get_sha256_checksum_by_path(file_path)
            return checksum.strip()
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(1.0)
    return ""


ARM_VERIFIER_NOT_AVAILABLE = "ARM_VERIFIER_NOT_AVAILABLE"
ARM_VERIFIER_NOT_AVAILABLE_ERROR = (
    f"{ARM_VERIFIER_NOT_AVAILABLE}: executor CPU is aarch64; the ARM verifier library is not available yet"
)
EXECUTOR_ARCH_UNKNOWN = "EXECUTOR_ARCH_UNKNOWN"
EXECUTOR_ARCH_UNKNOWN_ERROR = (
    f"{EXECUTOR_ARCH_UNKNOWN}: could not read the executor CPU architecture; its library was not replaced"
)
_ARCHES = {"aarch64": "aarch64", "arm64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}


async def executor_arch(shell) -> str | None:
    """"aarch64" or "x86_64" from `uname -m`, read once per shell. None when the read fails or names another CPU:
    never guessed, so a failed read cannot put the wrong build on a host; only a supported answer is cached."""
    arch = getattr(shell, "executor_arch", None)
    if arch in ("aarch64", "x86_64"):
        return arch
    try:
        result = await shell.ssh_client.run("uname -m", timeout=15)
        arch = _ARCHES.get(str(getattr(result, "stdout", "") or "").strip())
    except Exception:
        arch = None
    if arch is not None:
        shell.executor_arch = arch
    return arch


def aarch64_lib_path(lib_path: str) -> str | None:
    """The validator's aarch64 build of `lib_path` (`<lib>-aarch64.so` next to it), for ARM executors
    (GB300 Grace hosts). None while that build is not on this validator."""
    path = lib_path.removesuffix(".so") + "-aarch64.so"
    return path if os.path.exists(path) else None
