"""Install the latest published liumd binary on an executor when it is missing or stale."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import aiohttp

from core.config import settings

ARCH_KEYS = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
TARGET_ARCH = {"x86_64": "x86_64", "aarch64": "aarch64"}
FILE_BY_ARCH = {"x86_64": "liumd-amd64", "aarch64": "liumd-arm64"}
LIUMD_FILE_RX = re.compile(r"^liumd-(amd64|arm64)$")
SHA256_RX = re.compile(r"^[0-9a-f]{64}$")
TMP_BINARY = "/tmp/liumd.binary.fetch"

MANIFEST_TIMEOUT_SECONDS = 10
MAX_MANIFEST_BYTES = 64 * 1024
# The manifest is the same for every executor in a cycle; the batch id keeps one fetch per cycle.
MANIFEST_CACHE_TTL_SECONDS = 15 * 60
MANIFEST_FALLBACK_CACHE_SECONDS = 60
DEFAULT_REFRESH_BUDGET_SECONDS = 20
MIN_DOWNLOAD_BUDGET_SECONDS = 5

_MANIFEST_CACHE: dict[tuple[str, str], tuple[dict[str, Any] | None, str | None, float]] = {}
_MANIFEST_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class LiumdRefresh:
    ok: bool
    build_sha: str | None
    detail: str = ""
    changed: bool = False
    skipped: bool = False


@dataclass(frozen=True)
class LiumdVersion:
    build_sha: str | None
    target: str | None
    checked: bool = True


def _version_doc(stdout: str) -> LiumdVersion:
    try:
        doc = json.loads(stdout.strip())
    except ValueError:
        return LiumdVersion(None, None)
    if not isinstance(doc, dict):
        return LiumdVersion(None, None)
    build_sha = doc.get("build_sha")
    target = doc.get("target")
    return LiumdVersion(
        build_sha if isinstance(build_sha, str) and build_sha else None,
        target if isinstance(target, str) and target else None,
    )


async def _run(ctx: Any, command: str, timeout: int) -> Any:
    return await ctx.runner.run(command, timeout=timeout, retryable=False)


async def _read_version(ctx: Any) -> LiumdVersion:
    result = await _run(ctx, f"{shlex.quote(settings.LIUMD_BINARY_PATH)} version", 15)
    if not result.success:
        if getattr(result, "error_type", None) is not None:
            return LiumdVersion(None, None, checked=False)
        return LiumdVersion(None, None)
    return _version_doc(result.stdout or "")


async def _fetch_manifest(url: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    return None, f"manifest fetch status {response.status}"
                body = await response.content.read(MAX_MANIFEST_BYTES + 1)
    except (aiohttp.ClientError, TimeoutError) as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]
    if len(body) > MAX_MANIFEST_BYTES:
        return None, "manifest response is too large"
    try:
        doc = json.loads(body)
    except ValueError as exc:
        return None, f"manifest is not JSON: {exc}"
    return (doc, None) if isinstance(doc, dict) else (None, "manifest is not an object")


async def _manifest(ctx: Any, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    """The manifest, fetched at most once per validator cycle and shared across executors."""
    url = settings.LIUMD_BINARY_MANIFEST_URL
    batch = getattr(getattr(ctx, "config", None), "job_batch_id", None) or ""
    key = (batch, url)
    ttl = MANIFEST_CACHE_TTL_SECONDS if batch else MANIFEST_FALLBACK_CACHE_SECONDS
    cached = _MANIFEST_CACHE.get(key)
    if cached and time.monotonic() - cached[2] < ttl:
        return cached[0], cached[1]
    async with _MANIFEST_LOCK:
        cached = _MANIFEST_CACHE.get(key)
        if cached and time.monotonic() - cached[2] < ttl:
            return cached[0], cached[1]
        doc, error = await _fetch_manifest(url, timeout)
        _MANIFEST_CACHE[key] = (doc, error, time.monotonic())
        now = time.monotonic()
        for old_key in list(_MANIFEST_CACHE):
            if old_key != key and now - _MANIFEST_CACHE[old_key][2] > MANIFEST_CACHE_TTL_SECONDS:
                _MANIFEST_CACHE.pop(old_key, None)
        return doc, error


def _target_arch(target: str | None) -> str | None:
    return TARGET_ARCH.get((target or "").split("-", 1)[0])


async def _host_arch(ctx: Any) -> str | None:
    result = await _run(ctx, "uname -m", 15)
    if not result.success:
        return None
    return ARCH_KEYS.get((result.stdout or "").strip())


def _manifest_entry(manifest: dict[str, Any], arch: str) -> tuple[str, str] | None:
    binaries = manifest.get("binaries")
    entry = binaries.get(arch) if isinstance(binaries, dict) else None
    if not isinstance(entry, dict):
        return None
    filename = entry.get("file")
    sha256 = entry.get("sha256")
    if (
        not isinstance(filename, str)
        or not LIUMD_FILE_RX.fullmatch(filename)
        or filename != FILE_BY_ARCH.get(arch)
        or not isinstance(sha256, str)
        or not SHA256_RX.fullmatch(sha256)
    ):
        return None
    return filename, sha256


async def _download(ctx: Any, url: str, max_time: int) -> tuple[str | None, str | None]:
    command = (
        f"curl -fsSL --max-time {max_time} -o {shlex.quote(TMP_BINARY)} {shlex.quote(url)}; "
        f"rc=$?; echo RC:$rc; "
        f"if [ -f {shlex.quote(TMP_BINARY)} ]; then sha256sum {shlex.quote(TMP_BINARY)}; fi"
    )
    result = await _run(ctx, command, max_time + 10)
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


async def ensure_liumd_binary(
    ctx: Any, *, budget_seconds: float | None = None
) -> LiumdRefresh | None:
    """Repair liumd from the main manifest when enabled. None means the feature is off.

    The manifest comes from the validator, once per cycle. The whole attempt is bounded by the
    executor task time left after the scrape's own budget; a matching local build never downloads.
    """
    if not settings.LIUMD_BINARY_REFRESH_ENABLED:
        return None
    budget = (
        DEFAULT_REFRESH_BUDGET_SECONDS
        if budget_seconds is None
        else min(DEFAULT_REFRESH_BUDGET_SECONDS, budget_seconds)
    )
    if budget <= 0:
        return LiumdRefresh(False, None, "no task budget for the liumd refresh", skipped=True)
    started = time.monotonic()
    try:
        async with asyncio.timeout(budget):
            return await _ensure(ctx, budget, started)
    except TimeoutError:
        return LiumdRefresh(False, None, "liumd refresh budget spent", skipped=True)


async def _ensure(ctx: Any, budget: float, started: float) -> LiumdRefresh:
    try:
        version = await _read_version(ctx)
        if not version.checked:
            return LiumdRefresh(False, None, "local liumd version could not be read")

        manifest, error = await _manifest(ctx, min(MANIFEST_TIMEOUT_SECONDS, budget))
        if manifest is None:
            if version.build_sha:
                return LiumdRefresh(
                    True,
                    version.build_sha,
                    f"manifest unavailable, using local liumd: {error or 'unknown error'}",
                )
            return LiumdRefresh(False, None, error or "manifest unavailable")
        expected_build = manifest.get("build_sha")
        if not isinstance(expected_build, str) or not expected_build:
            return LiumdRefresh(False, None, "manifest has no build_sha")

        if version.build_sha == expected_build:
            return LiumdRefresh(True, expected_build)

        arch = _target_arch(version.target) or await _host_arch(ctx)
        if arch is None:
            return LiumdRefresh(False, version.build_sha, "unsupported host architecture")
        entry = _manifest_entry(manifest, arch)
        if entry is None:
            return LiumdRefresh(False, version.build_sha, "manifest has no binary for this arch")
        filename, expected_sha = entry

        remaining = budget - (time.monotonic() - started)
        if remaining < MIN_DOWNLOAD_BUDGET_SECONDS:
            return LiumdRefresh(
                False, version.build_sha, "no task budget for the liumd download", skipped=True
            )
        url = urljoin(settings.LIUMD_BINARY_MANIFEST_URL, filename)
        fetched_sha, error = await _download(ctx, url, max(1, int(remaining)))
        if fetched_sha is None:
            return LiumdRefresh(
                False, version.build_sha, error or "download failed", skipped=bool(version.build_sha)
            )
        if fetched_sha != expected_sha:
            return LiumdRefresh(
                False,
                version.build_sha,
                "downloaded sha256 does not match manifest",
                skipped=bool(version.build_sha),
            )

        install = await _run(
            ctx,
            f"install -m 0755 {shlex.quote(TMP_BINARY)} {shlex.quote(settings.LIUMD_BINARY_PATH)}",
            30,
        )
        if not install.success:
            return LiumdRefresh(
                False,
                version.build_sha,
                (install.stderr or install.stdout or "install failed")[-300:],
                skipped=bool(version.build_sha),
            )
        installed_build = await _read_version(ctx)
        if installed_build.build_sha != expected_build:
            return LiumdRefresh(
                False, installed_build.build_sha, "installed binary reports the wrong build_sha"
            )
        return LiumdRefresh(True, installed_build.build_sha, changed=True)
    except Exception as exc:  # noqa: BLE001 - a repair failure is reported to the caller
        return LiumdRefresh(False, None, f"{type(exc).__name__}: {exc}"[:300])


async def refresh_liumd_from_answer(
    ctx: Any,
    build_sha: str | None,
    target: str | None,
    *,
    budget_seconds: float | None = None,
) -> LiumdRefresh | None:
    """Refresh from the build the run answered with, without a separate version command."""
    if not settings.LIUMD_BINARY_REFRESH_ENABLED:
        return None
    budget = (
        DEFAULT_REFRESH_BUDGET_SECONDS
        if budget_seconds is None
        else min(DEFAULT_REFRESH_BUDGET_SECONDS, budget_seconds)
    )
    if budget <= 0:
        return LiumdRefresh(False, build_sha, "no task budget for the liumd refresh", skipped=True)
    started = time.monotonic()
    try:
        async with asyncio.timeout(budget):
            return await _refresh_from_answer(
                ctx, build_sha or None, target or None, budget, started
            )
    except TimeoutError:
        return LiumdRefresh(False, build_sha, "liumd refresh budget spent", skipped=True)


async def _refresh_from_answer(
    ctx: Any, build_sha: str | None, target: str | None, budget: float, started: float
) -> LiumdRefresh:
    try:
        manifest, error = await _manifest(ctx, min(MANIFEST_TIMEOUT_SECONDS, budget))
        if manifest is None:
            if build_sha:
                return LiumdRefresh(
                    True,
                    build_sha,
                    f"manifest unavailable, using local liumd: {error or 'unknown error'}",
                )
            return LiumdRefresh(False, None, error or "manifest unavailable")
        expected_build = manifest.get("build_sha")
        if not isinstance(expected_build, str) or not expected_build:
            return LiumdRefresh(False, build_sha, "manifest has no build_sha")
        if build_sha == expected_build:
            return LiumdRefresh(True, expected_build)

        arch = _target_arch(target) or await _host_arch(ctx)
        if arch is None:
            return LiumdRefresh(False, build_sha, "unsupported host architecture")
        entry = _manifest_entry(manifest, arch)
        if entry is None:
            return LiumdRefresh(False, build_sha, "manifest has no binary for this arch")
        filename, expected_sha = entry

        remaining = budget - (time.monotonic() - started)
        if remaining < MIN_DOWNLOAD_BUDGET_SECONDS:
            return LiumdRefresh(
                False, build_sha, "no task budget for the liumd download", skipped=True
            )
        url = urljoin(settings.LIUMD_BINARY_MANIFEST_URL, filename)
        fetched_sha, error = await _download(ctx, url, max(1, int(remaining)))
        if fetched_sha is None:
            return LiumdRefresh(
                False, build_sha, error or "download failed", skipped=bool(build_sha)
            )
        if fetched_sha != expected_sha:
            return LiumdRefresh(
                False,
                build_sha,
                "downloaded sha256 does not match manifest",
                skipped=bool(build_sha),
            )

        install = await _run(
            ctx,
            f"install -m 0755 {shlex.quote(TMP_BINARY)} {shlex.quote(settings.LIUMD_BINARY_PATH)}"
            f" && sha256sum {shlex.quote(settings.LIUMD_BINARY_PATH)}",
            30,
        )
        if not install.success:
            return LiumdRefresh(
                False,
                build_sha,
                (install.stderr or install.stdout or "install failed")[-300:],
                skipped=bool(build_sha),
            )
        installed_sha = None
        for line in (install.stdout or "").splitlines():
            parts = line.split()
            if (
                parts
                and len(parts[0]) == 64
                and all(c in "0123456789abcdef" for c in parts[0])
            ):
                installed_sha = parts[0]
        if installed_sha != expected_sha:
            return LiumdRefresh(
                False,
                build_sha,
                "installed binary sha256 does not match manifest",
                skipped=bool(build_sha),
            )
        return LiumdRefresh(True, expected_build, changed=True)
    except Exception as exc:  # noqa: BLE001 - a repair failure is reported to the caller
        return LiumdRefresh(False, build_sha, f"{type(exc).__name__}: {exc}"[:300])
