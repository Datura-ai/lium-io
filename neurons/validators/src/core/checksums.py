from __future__ import annotations

import asyncio
import hashlib


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
