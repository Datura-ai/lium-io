"""Install the latest published liumd binary on an executor when it is missing or stale.

The manifest names the build and each binary's sha256. It is trusted only when manifest.json.sig, an ed25519
signature over the manifest's exact bytes, verifies against the release key pinned below. A missing, malformed or
foreign signature refuses the refresh: nothing is installed and the local binary stays.

The signed manifest also names its `release` (the liumd-v X.Y.Z tag). The validator keeps the highest release it
accepted, for each node and overall, and refuses a lower one, so an older signed manifest (with its binaries) cannot be
put back on main to roll nodes back. The floor is recorded before anything is installed; if it cannot be recorded,
nothing is installed. MIN_RELEASE, the release of the signed manifest vendored with this code, is a floor no Redis
loss can lower, so every release vendored into the image raises it. A refusal is reported as `refused`, not as an
unavailable manifest.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiohttp
import redis.exceptions
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from core.config import settings

ARCH_KEYS = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
TARGET_ARCH = {"x86_64": "x86_64", "aarch64": "aarch64"}
FILE_BY_ARCH = {"x86_64": "liumd-amd64", "aarch64": "liumd-arm64"}
LIUMD_FILE_RX = re.compile(r"^liumd-(amd64|arm64)$")
SHA256_RX = re.compile(r"^[0-9a-f]{64}$")
TMP_BINARY = "/tmp/liumd.binary.fetch"

# X.Y.Z, each part at most 6 digits with no leading zero (sign_manifest.py signs only this form)
RELEASE_RX = re.compile(r"(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})")
# the signed manifest vendored with this code: neurons/executor/liumd in the repo, /app/liumd-vendored in the image
VENDORED_MANIFEST_DIRS = (
    Path(__file__).resolve().parents[2] / "liumd-vendored",
    Path(__file__).resolve().parents[3] / "executor" / "liumd",
)
# highest release accepted overall, and per node under ":<miner hotkey>:<executor uuid>"; stored zero-padded, so the
# strings compare in release order
RELEASE_FLOOR_KEY = "liumd_release_floor"

MANIFEST_TIMEOUT_SECONDS = 10
MAX_MANIFEST_BYTES = 64 * 1024
MAX_SIGNATURE_BYTES = 1024
# The liumd release key (ed25519:04d614eaf59dbbb8). Its seed is held only by the release signing job, which signs
# manifest.json on a liumd-v* tag; a manifest changed without that key does not verify.
MANIFEST_PUBLIC_KEY_HEX = "7b187df472a586d61423cf24d72bbccdb98bfd671df5b4b6b9e713676990a41b"
# The manifest is the same for every executor in a cycle; the batch id keeps one fetch per cycle.
MANIFEST_CACHE_TTL_SECONDS = 15 * 60
MANIFEST_FALLBACK_CACHE_SECONDS = 60
DEFAULT_REFRESH_BUDGET_SECONDS = 20
MIN_DOWNLOAD_BUDGET_SECONDS = 5

_MANIFEST_CACHE: dict[tuple[str, str], tuple[dict[str, Any] | None, str | None, float]] = {}
_MANIFEST_LOCK = asyncio.Lock()


class _Refused(str):
    """An error that refuses the manifest itself (signature, form or release), not a network miss."""


@dataclass(frozen=True)
class LiumdRefresh:
    ok: bool
    build_sha: str | None
    detail: str = ""
    changed: bool = False
    skipped: bool = False
    # the manifest was refused (bad or missing signature, older release): nothing installed, the local binary stays
    refused: bool = False


def _release(text: Any) -> tuple[int, int, int] | None:
    m = RELEASE_RX.fullmatch(text) if isinstance(text, str) else None
    return tuple(int(g) for g in m.groups()) if m else None  # type: ignore[return-value]


def _padded(release: tuple[int, int, int]) -> str:
    return ".".join(f"{part:06d}" for part in release)


def _floor_keys(ctx: Any) -> list[str]:
    return [RELEASE_FLOOR_KEY, f"{RELEASE_FLOOR_KEY}:{ctx.miner_hotkey}:{ctx.executor.uuid}"]


async def _release_floor(ctx: Any) -> tuple[int, int, int]:
    """The highest of MIN_RELEASE and the releases this validator accepted, overall and for this node."""
    floor = MIN_RELEASE
    for stored in await ctx.services.redis.redis.mget(_floor_keys(ctx)):
        if stored is not None:
            text = stored.decode() if isinstance(stored, bytes) else stored
            floor = max(floor, tuple(int(part) for part in text.split(".")))
    return floor


async def _raise_floor(ctx: Any, release: tuple[int, int, int]) -> str:
    """Records an accepted release before anything is installed; "" or why it could not be recorded."""
    keys, value = _floor_keys(ctx), _padded(release)
    try:
        # WATCHed: a concurrent raise reruns the compare, so a lower release never overwrites a higher one
        async with ctx.services.redis.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(*keys)
                    current = [c.decode() if isinstance(c, bytes) else c for c in await pipe.mget(keys)]
                    pipe.multi()
                    for key, stored in zip(keys, current):
                        if stored is None or value > stored:
                            pipe.set(key, value)
                    await pipe.execute()
                    break
                except redis.exceptions.WatchError:
                    continue
    except Exception as exc:  # noqa: BLE001 - the caller installs nothing on an unrecorded floor
        return f"release floor not recorded: {type(exc).__name__}: {exc}"[:300]
    return ""


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


async def _fetch_bytes(url: str, timeout: float, limit: int) -> tuple[bytes | None, str | None]:
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    return None, f"fetch status {response.status}"
                body = await response.content.read(limit + 1)
    except (aiohttp.ClientError, TimeoutError) as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]
    if len(body) > limit:
        return None, "response is too large"
    return body, None


def _signature_error(manifest: bytes, signature: bytes) -> str | None:
    """None when `signature` (64 bytes, hex) is the pinned release key's ed25519 signature over `manifest`."""
    try:
        sig = bytes.fromhex(signature.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        return "manifest signature is not hex"
    if len(sig) != 64:
        return "manifest signature is not 64 bytes"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(MANIFEST_PUBLIC_KEY_HEX)).verify(sig, manifest)
    except InvalidSignature:
        return "manifest signature does not verify against the pinned release key"
    return None


def _vendored_release(dirs: tuple[Path, ...] = VENDORED_MANIFEST_DIRS) -> tuple[int, int, int]:
    """The release of the first vendored manifest.json found, if its .sig verifies; 1.0.0 while none is signed."""
    for directory in dirs:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            try:
                body, signature = manifest.read_bytes(), (directory / "manifest.json.sig").read_bytes()
                doc = json.loads(body) if _signature_error(body, signature) is None else None
            except (OSError, ValueError):
                doc = None
            release = _release(doc.get("release")) if isinstance(doc, dict) else None
            return release or (1, 0, 0)
    return (1, 0, 0)


MIN_RELEASE = _vendored_release()


async def _fetch_manifest(url: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    """The manifest, only once its signature verifies over the exact bytes fetched; nothing in it is read before."""
    body, error = await _fetch_bytes(url, timeout, MAX_MANIFEST_BYTES)
    if body is None:
        return None, f"manifest {error}"
    signature, error = await _fetch_bytes(url + ".sig", timeout, MAX_SIGNATURE_BYTES)
    if signature is None:
        # the manifest is there without its signature: refused; any other miss is the network's
        message = f"manifest signature {error}"
        return None, _Refused(message) if error == "fetch status 404" else message
    if error := _signature_error(body, signature):
        return None, _Refused(error)
    try:
        doc = json.loads(body)
    except ValueError as exc:
        return None, _Refused(f"manifest is not JSON: {exc}")
    return (doc, None) if isinstance(doc, dict) else (None, _Refused("manifest is not an object"))


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


async def _release_gate(ctx: Any, manifest: dict[str, Any], local_build: str | None) -> LiumdRefresh | None:
    """None when the manifest's release may be installed (and is now the floor); else the refusal or the reason nothing
    is installed. Both refresh paths go through it, so neither can install a signed but older release."""
    release = _release(manifest.get("release"))
    if release is None:
        return LiumdRefresh(False, local_build, "manifest refused: no X.Y.Z release", refused=True)
    try:
        floor = await _release_floor(ctx)
    except Exception as exc:  # noqa: BLE001 - unknown floor: install nothing, keep the local binary
        detail = f"release floor unavailable: {type(exc).__name__}: {exc}"[:300]
        return LiumdRefresh(bool(local_build), local_build, detail)
    if release < floor:
        return LiumdRefresh(
            False,
            local_build,
            f"manifest refused: release {'.'.join(map(str, release))} is below "
            f"{'.'.join(map(str, floor))}, the highest accepted",
            refused=True,
        )
    # recorded first: an install the floor does not remember could be rolled back by an older signed manifest
    if error := await _raise_floor(ctx, release):
        return LiumdRefresh(bool(local_build), local_build, f"nothing installed: {error}")
    return None


async def _ensure(ctx: Any, budget: float, started: float) -> LiumdRefresh:
    try:
        version = await _read_version(ctx)
        if not version.checked:
            return LiumdRefresh(False, None, "local liumd version could not be read")

        manifest, error = await _manifest(ctx, min(MANIFEST_TIMEOUT_SECONDS, budget))
        if isinstance(error, _Refused):
            return LiumdRefresh(False, version.build_sha, f"manifest refused: {error}", refused=True)
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
        if (gate := await _release_gate(ctx, manifest, version.build_sha)) is not None:
            return gate

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
        if isinstance(error, _Refused):
            return LiumdRefresh(False, build_sha, f"manifest refused: {error}", refused=True)
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
        if (gate := await _release_gate(ctx, manifest, build_sha)) is not None:
            return gate
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
