"""The build context of a custom-dockerfile pod: a gzipped tar the renter uploaded, fetched by the
validator, checked, and unpacked into the build container's context directory next to the Dockerfile."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import tarfile
import zlib
from urllib.parse import urlsplit

import aiohttp

_FETCH_TIMEOUT_SECONDS = 120
# tarfile keeps every member it has read, so cap the count, not only the bytes
MAX_ARCHIVE_ENTRIES = 10_000
# only plain ustar headers: every GNU long-name/long-link, PAX, sparse or other extension header is refused before
# tarfile parses it, which closes tarfile's memory tricks as a class instead of one by one
_ALLOWED_TYPES = (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE)


class BuildContextError(Exception):
    """A build context that is refused; the message is renter-facing."""


class _CappedReader:
    """Counts the decompressed bytes tarfile reads, headers and long names included, and stops at the cap."""

    def __init__(self, raw: gzip.GzipFile, cap: int, max_unpacked_bytes: int) -> None:
        self._raw, self._cap, self._max, self._seen = raw, cap, max_unpacked_bytes, 0

    def read(self, size: int = -1) -> bytes:
        # never ask for more than the cap allows, so one huge header cannot be allocated first
        left = self._cap - self._seen + 1
        chunk = self._raw.read(left if size < 0 else min(size, left))
        self._seen += len(chunk)
        if self._seen > self._cap:
            raise BuildContextError(f"build context unpacks to more than {self._max} bytes")
        return chunk


class _BoundedTarInfo(tarfile.TarInfo):
    """Refuses any header but a plain file, directory or symlink before tarfile reads what follows it."""

    def _proc_member(self, tarfile_):
        if self.type not in _ALLOWED_TYPES:
            raise BuildContextError(
                f"build context entry is not a plain file, directory or symlink: {self.name} "
                "(create the archive with `tar --format=ustar`, names up to 100 characters, no extended attributes)"
            )
        return super()._proc_member(tarfile_)


def allowed_hosts(raw: str) -> set[str]:
    return {host.strip().lower() for host in (raw or "").split(",") if host.strip()}


def check_url(url: str, hosts: set[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.hostname.lower() not in hosts:
        raise BuildContextError("build context URL is not on an allowed host")
    try:
        port = parts.port
    except ValueError:  # a port that is not a number
        raise BuildContextError("build context URL is not on an allowed host") from None
    if parts.username or parts.password or port not in (None, 443):
        raise BuildContextError("build context URL is not on an allowed host")


def check_archive(data: bytes, max_unpacked_bytes: int) -> None:
    """A gzipped ustar of plain files, directories and symlinks, every path inside the context."""
    total = 0
    # tar headers and padding take at most 1 KiB per entry on top of the file bytes, plus the end records
    cap = max_unpacked_bytes + (MAX_ARCHIVE_ENTRIES + 20) * 1024
    reader = _CappedReader(gzip.GzipFile(fileobj=io.BytesIO(data)), cap, max_unpacked_bytes)
    try:
        # stream mode: tarfile reads the decompressed bytes only through the capped reader
        with tarfile.open(fileobj=reader, mode="r|", tarinfo=_BoundedTarInfo) as archive:
            for count, member in enumerate(archive, start=1):
                archive.members = []  # stream mode keeps every member read; only the count is needed
                if count > MAX_ARCHIVE_ENTRIES:
                    raise BuildContextError(
                        f"build context has more than {MAX_ARCHIVE_ENTRIES} entries"
                    )
                if member.name.startswith("/") or ".." in member.name.split("/"):
                    raise BuildContextError(
                        f"build context entry leaves the context: {member.name}"
                    )
                # no `..` part at all: a chain of symlinks through `.` can otherwise walk out of the context
                if member.issym() and (
                    member.linkname.startswith("/") or ".." in member.linkname.split("/")
                ):
                    raise BuildContextError(f"build context link leaves the context: {member.name}")
                total += member.size
                if total > max_unpacked_bytes:
                    raise BuildContextError(
                        f"build context unpacks to more than {max_unpacked_bytes} bytes"
                    )
    except (tarfile.TarError, OSError, EOFError, zlib.error, RecursionError) as exc:
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
    # the parse is CPU-bound; off the event loop so a crafted archive cannot stall the validator
    await asyncio.to_thread(check_archive, data, max_unpacked_bytes)
    return data
