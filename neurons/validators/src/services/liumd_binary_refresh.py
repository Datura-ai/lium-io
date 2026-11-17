"""Repair a missing or stale liumd binary on an executor from the validator's pinned build."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from core.config import settings

ARCH_KEYS = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
TARGET_ARCH = {"x86_64": "x86_64", "aarch64": "aarch64"}
FILE_BY_ARCH = {"x86_64": "liumd-amd64", "aarch64": "liumd-arm64"}
LIUMD_FILE_RX = re.compile(r"^liumd-(amd64|arm64)$")
SHA256_RX = re.compile(r"^[0-9a-f]{64}$")
# A verified build is trusted for this long; a reimaged host at the same endpoint re-checks after it.
VERIFIED_BUILD_TTL_SECONDS = 30 * 60
# The whole refresh is capped from the executor task's remaining time, keeping this much for the scrape.
REFRESH_BUDGET_SECONDS = 60
REFRESH_TASK_RESERVE_SECONDS = 60
MIN_REFRESH_BUDGET_SECONDS = 20

# The liumd build this validator release expects. test_liumd_binary_manifest.py keeps this equal
# to neurons/executor/liumd/manifest.json.
PINNED_MANIFEST = {
    "build_sha": "25f8f05508ac",
    "binaries": {
        "x86_64": {
            "file": "liumd-amd64",
            "sha256": "1b88754fd33a515cd98e68c94a21b0aaa833f1a0e06d56e5176da3876b6d1555",
        },
        "aarch64": {
            "file": "liumd-arm64",
            "sha256": "cc12eda4fb8042a6aa60d957e52a8ed674d67d9bda29f26ec6d4a765ccba4625",
        },
    },
}

_VERIFIED_BUILDS: dict[tuple[str, str, str, int], tuple[str, float]] = {}


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


def _target_arch(target: str | None) -> str | None:
    return TARGET_ARCH.get((target or "").split("-", 1)[0])


async def _host_arch(ctx: Any) -> str | None:
    result = await _run(ctx, "uname -m", 15)
    if not result.success:
        return None
    return ARCH_KEYS.get((result.stdout or "").strip())


def _pinned_entry(arch: str) -> tuple[str, str] | None:
    binaries = PINNED_MANIFEST.get("binaries")
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


def _install_command(url: str, expected_sha256: str, path: str) -> str:
    return (
        "tmp=$(mktemp /tmp/liumd.XXXXXX) || exit 20; "
        "trap 'rm -f \"$tmp\"' EXIT; "
        f"curl -fsSL --max-time 180 -o \"$tmp\" {shlex.quote(url)}; curl_rc=$?; "
        "echo \"CURL_RC:$curl_rc\"; "
        "[ \"$curl_rc\" -eq 0 ] || exit 10; "
        f"printf '%s  %s\\n' {shlex.quote(expected_sha256)} \"$tmp\" | sha256sum -c -; sha_rc=$?; "
        "echo \"SHA_RC:$sha_rc\"; "
        "[ \"$sha_rc\" -eq 0 ] || exit 11; "
        f"install -m 0755 \"$tmp\" {shlex.quote(path)}; install_rc=$?; "
        "echo \"INSTALL_RC:$install_rc\"; "
        "exit \"$install_rc\""
    )


def _marker(stdout: str, name: str) -> int | None:
    for line in stdout.splitlines():
        if line.startswith(f"{name}:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


async def _install(ctx: Any, url: str, expected_sha256: str, path: str) -> tuple[bool, str]:
    result = await _run(ctx, _install_command(url, expected_sha256, path), 240)
    if result.success:
        return True, ""
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    curl_rc = _marker(stdout, "CURL_RC")
    if curl_rc not in (None, 0):
        return False, f"download failed (curl exit {curl_rc}): {stderr or stdout}"[:300]
    sha_rc = _marker(stdout, "SHA_RC")
    if sha_rc not in (None, 0):
        return False, f"downloaded sha256 does not match the pinned build: {stdout[-200:]}"
    install_rc = _marker(stdout, "INSTALL_RC")
    if install_rc not in (None, 0):
        return False, f"install failed (exit {install_rc}): {stderr or stdout}"[:300]
    detail = stderr or stdout or getattr(result, "error_type", None) or "download or install failed"
    return False, str(detail)[-300:]


def _cache_key(ctx: Any) -> tuple[str, str, str, int]:
    executor = ctx.executor
    return ctx.miner_hotkey, executor.uuid, executor.address, executor.ssh_port


def _cached_build(ctx: Any, expected_build: str) -> bool:
    cached = _VERIFIED_BUILDS.get(_cache_key(ctx))
    return bool(
        cached
        and cached[0] == expected_build
        and time.monotonic() - cached[1] < VERIFIED_BUILD_TTL_SECONDS
    )


def _remember_build(ctx: Any, expected_build: str) -> None:
    now = time.monotonic()
    for key, (_, stamp) in list(_VERIFIED_BUILDS.items()):
        if now - stamp >= VERIFIED_BUILD_TTL_SECONDS:
            _VERIFIED_BUILDS.pop(key, None)
    _VERIFIED_BUILDS[_cache_key(ctx)] = (expected_build, now)


def _forget_build(ctx: Any) -> None:
    _VERIFIED_BUILDS.pop(_cache_key(ctx), None)


def _refresh_budget(seconds_left: float | None) -> float:
    if seconds_left is None:
        return float(REFRESH_BUDGET_SECONDS)
    return min(float(REFRESH_BUDGET_SECONDS), seconds_left - REFRESH_TASK_RESERVE_SECONDS)


async def ensure_liumd_binary(
    ctx: Any, *, seconds_left: float | None = None
) -> LiumdRefresh | None:
    """Repair liumd from the pinned build when enabled. None means the feature is off.

    The whole attempt is capped from the executor task's remaining time; a node whose local build
    already matches is checked with one short SSH command and never fetches.
    """
    if not settings.LIUMD_BINARY_REFRESH_ENABLED:
        return None
    expected_build = PINNED_MANIFEST.get("build_sha")
    if not isinstance(expected_build, str) or not expected_build:
        return LiumdRefresh(False, None, "pinned manifest has no build_sha")
    if _cached_build(ctx, expected_build):
        return LiumdRefresh(True, expected_build)
    budget = _refresh_budget(seconds_left)
    if budget < MIN_REFRESH_BUDGET_SECONDS:
        return LiumdRefresh(
            False, None, f"refresh budget {budget:.0f}s is too small", skipped=True
        )
    try:
        async with asyncio.timeout(budget):
            return await _ensure(ctx, expected_build)
    except TimeoutError:
        _forget_build(ctx)
        return LiumdRefresh(False, None, f"refresh budget {budget:.0f}s spent", skipped=True)


async def _ensure(ctx: Any, expected_build: str) -> LiumdRefresh:
    try:
        version = await _read_version(ctx)
        if not version.checked:
            _forget_build(ctx)
            return LiumdRefresh(False, None, "local liumd version could not be read", skipped=True)
        if version.build_sha == expected_build:
            _remember_build(ctx, expected_build)
            return LiumdRefresh(True, expected_build)

        arch = _target_arch(version.target) or await _host_arch(ctx)
        if arch is None:
            _forget_build(ctx)
            return LiumdRefresh(False, version.build_sha, "unsupported host architecture")
        entry = _pinned_entry(arch)
        if entry is None:
            _forget_build(ctx)
            return LiumdRefresh(False, version.build_sha, "pinned manifest has no binary for this arch")
        filename, expected_sha256 = entry

        base = settings.LIUMD_BINARY_FETCH_BASE_URL
        url = urljoin(base if base.endswith("/") else f"{base}/", filename)
        installed, error = await _install(ctx, url, expected_sha256, settings.LIUMD_BINARY_PATH)
        if not installed:
            _forget_build(ctx)
            return LiumdRefresh(False, version.build_sha, error)

        installed_version = await _read_version(ctx)
        if installed_version.build_sha != expected_build:
            _forget_build(ctx)
            return LiumdRefresh(
                False, installed_version.build_sha, "installed binary reports the wrong build_sha"
            )
        _remember_build(ctx, expected_build)
        return LiumdRefresh(True, expected_build, changed=True)
    except Exception as exc:  # noqa: BLE001 - a repair failure is reported to the caller
        _forget_build(ctx)
        return LiumdRefresh(False, None, f"{type(exc).__name__}: {exc}"[:300])
