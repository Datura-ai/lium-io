"""The build context of a custom-dockerfile pod: a gzipped tar the renter uploaded, fetched by the
validator, checked, and unpacked into the build container's context directory next to the Dockerfile."""

from __future__ import annotations

import hashlib
import io
import posixpath
import tarfile
from urllib.parse import urlsplit

import aiohttp

_FETCH_TIMEOUT_SECONDS = 120


class BuildContextError(Exception):
    """A build context that is refused; the message is renter-facing."""


def allowed_hosts(raw: str) -> set[str]:
    return {host.strip().lower() for host in (raw or "").split(",") if host.strip()}


def check_url(url: str, hosts: set[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.hostname.lower() not in hosts:
        raise BuildContextError("build context URL is not on an allowed host")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise BuildContextError("build context URL is not on an allowed host")


def check_archive(data: bytes, max_unpacked_bytes: int) -> None:
    """A gzipped tar of plain files, directories and links, every path inside the context."""
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                name = posixpath.normpath(member.name)
                if name.startswith(("/", "../")) or name == "..":
                    raise BuildContextError(
                        f"build context entry leaves the context: {member.name}"
                    )
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise BuildContextError(
                        f"build context entry is not a file, directory or link: {member.name}"
                    )
                if member.issym() or member.islnk():
                    # a symlink resolves from its own directory, a hard link from the archive root
                    base = posixpath.dirname(name) if member.issym() else ""
                    target = posixpath.normpath(posixpath.join(base, member.linkname))
                    if target.startswith(("/", "../")) or target == "..":
                        raise BuildContextError(
                            f"build context link leaves the context: {member.name}"
                        )
                total += member.size
                if total > max_unpacked_bytes:
                    raise BuildContextError(
                        f"build context unpacks to more than {max_unpacked_bytes} bytes"
                    )
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise BuildContextError("build context is not a gzipped tar archive") from exc


async def fetch(
    url: str,
    sha256: str,
    *,
    hosts: set[str],
    max_bytes: int,
    max_unpacked_bytes: int,
) -> bytes:
    """The archive at `url`, once its size, digest and entries pass; BuildContextError otherwise."""
    check_url(url, hosts)
    expected = (sha256 or "").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise BuildContextError("build context sha256 is missing or malformed")

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # no redirects: the host check above is the one the bytes come from
            async with session.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    raise BuildContextError(
                        f"build context download failed (HTTP {response.status})"
                    )
                if (response.content_length or 0) > max_bytes:
                    raise BuildContextError(f"build context is larger than {max_bytes} bytes")
                async for chunk in response.content.iter_chunked(1 << 16):
                    size += len(chunk)
                    if size > max_bytes:
                        raise BuildContextError(f"build context is larger than {max_bytes} bytes")
                    digest.update(chunk)
                    chunks.append(chunk)
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise BuildContextError("build context download failed") from exc

    if digest.hexdigest() != expected:
        raise BuildContextError("build context sha256 does not match")
    data = b"".join(chunks)
    check_archive(data, max_unpacked_bytes)
    return data
