"""Install the latest published liumd binary on an executor when it is missing or stale."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from core.config import settings

ARCH_KEYS = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
TMP_BINARY = "/tmp/liumd.binary.fetch"


@dataclass(frozen=True)
class LiumdRefresh:
    ok: bool
    build_sha: str | None
    detail: str = ""
    changed: bool = False


def _version_build_sha(stdout: str) -> str | None:
    try:
        doc = json.loads(stdout.strip())
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    value = doc.get("build_sha")
    return value if isinstance(value, str) and value else None


async def _run(ctx: Any, command: str, timeout: int) -> Any:
    return await ctx.runner.run(command, timeout=timeout, retryable=False)


async def _read_version(ctx: Any) -> str | None:
    result = await _run(ctx, f"{shlex.quote(settings.LIUMD_BINARY_PATH)} version", 15)
    if not result.success:
        return None
    return _version_build_sha(result.stdout or "")


async def _read_manifest(ctx: Any) -> tuple[dict[str, Any] | None, str | None]:
    url = settings.LIUMD_BINARY_MANIFEST_URL
    result = await _run(ctx, f"curl -fsSL --max-time 60 {shlex.quote(url)}", 90)
    if not result.success:
        return None, (result.stderr or result.stdout or "manifest fetch failed")[-300:]
    try:
        doc = json.loads(result.stdout or "")
    except ValueError as exc:
        return None, f"manifest is not JSON: {exc}"
    return doc if isinstance(doc, dict) else None, None


async def _download(ctx: Any, url: str) -> tuple[str | None, str | None]:
    command = (
        f"curl -fsSL --max-time 180 -o {shlex.quote(TMP_BINARY)} {shlex.quote(url)}; "
        f"rc=$?; echo RC:$rc; "
        f"if [ -f {shlex.quote(TMP_BINARY)} ]; then sha256sum {shlex.quote(TMP_BINARY)}; fi"
    )
    result = await _run(ctx, command, 210)
    stdout = result.stdout or ""
    rc = None
    digest = None
    for line in stdout.splitlines():
        if line.startswith("RC:"):
            try:
                rc = int(line[3:].strip())
            except ValueError:
                rc = None
        parts = line.split()
        if parts and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            digest = parts[0]
    if rc != 0:
        return None, (result.stderr or stdout or f"curl exit {rc}")[-300:]
    if digest is None:
        return None, "downloaded binary has no sha256"
    return digest, None


async def ensure_liumd_binary(ctx: Any) -> LiumdRefresh | None:
    """Repair liumd from the manifest when enabled. None means the feature is off."""
    if not settings.LIUMD_BINARY_REFRESH_ENABLED:
        return None
    try:
        arch_result = await _run(ctx, "uname -m", 15)
        arch = ARCH_KEYS.get((arch_result.stdout or "").strip())
        if not arch_result.success or arch is None:
            return LiumdRefresh(False, await _read_version(ctx), "unsupported host architecture")

        manifest, error = await _read_manifest(ctx)
        if manifest is None:
            return LiumdRefresh(False, await _read_version(ctx), error or "manifest unavailable")
        expected_build = manifest.get("build_sha")
        binaries = manifest.get("binaries")
        entry = binaries.get(arch) if isinstance(binaries, dict) else None
        if not isinstance(expected_build, str) or not isinstance(entry, dict):
            return LiumdRefresh(False, await _read_version(ctx), "manifest has no build_sha/binary for this arch")
        expected_sha = entry.get("sha256")
        filename = entry.get("file")
        if not isinstance(expected_sha, str) or not isinstance(filename, str):
            return LiumdRefresh(False, await _read_version(ctx), "manifest binary entry is incomplete")

        local_build = await _read_version(ctx)
        if local_build == expected_build:
            return LiumdRefresh(True, local_build)

        url = urljoin(settings.LIUMD_BINARY_MANIFEST_URL, filename)
        fetched_sha, error = await _download(ctx, url)
        if fetched_sha is None:
            return LiumdRefresh(False, local_build, error or "download failed")
        if fetched_sha != expected_sha:
            return LiumdRefresh(False, local_build, "downloaded sha256 does not match manifest")

        install = await _run(
            ctx,
            f"install -m 0755 {shlex.quote(TMP_BINARY)} {shlex.quote(settings.LIUMD_BINARY_PATH)}",
            30,
        )
        if not install.success:
            return LiumdRefresh(False, local_build, (install.stderr or install.stdout or "install failed")[-300:])
        installed_build = await _read_version(ctx)
        if installed_build != expected_build:
            return LiumdRefresh(False, installed_build, "installed binary reports the wrong build_sha")
        return LiumdRefresh(True, installed_build, changed=True)
    except Exception as exc:  # noqa: BLE001 - a repair failure is reported to the caller
        return LiumdRefresh(False, None, f"{type(exc).__name__}: {exc}"[:300])
