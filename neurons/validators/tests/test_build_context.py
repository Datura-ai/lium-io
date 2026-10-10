"""The build context of a custom-dockerfile pod: fetched, checked and unpacked before the build."""

from __future__ import annotations

import hashlib
import io
import tarfile
from unittest.mock import AsyncMock

import pytest
from services import build_context
from services.build_context import BuildContextError

from tests.test_custom_dockerfile_build import _base_payload, _make_dind_ssh, _make_esl, deps, svc  # noqa: F401

HOSTS = {"contexts.example.com"}
URL = "https://contexts.example.com/ctx.tar.gz?sig=x"


def _archive(
    entries: dict[str, bytes] | None = None,
    *,
    links: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (entries or {"app.py": b"print('hi')\n"}).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
        for name, target in (hardlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE
            info.linkname = target
            tar.addfile(info)
    return buf.getvalue()


class _FakeContent:
    def __init__(self, data: bytes):
        self._data = data

    async def iter_chunked(self, size: int):
        for i in range(0, len(self._data), size):
            yield self._data[i : i + size]


class _FakeResponse:
    def __init__(self, data: bytes, status: int = 200, content_length: int | None = None):
        self.status = status
        self.content_length = content_length
        self.content = _FakeContent(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse, calls: list):
        self._response = response
        self._calls = calls

    def get(self, url, **kwargs):
        self._calls.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _serve(monkeypatch, data: bytes, **response_kwargs) -> list:
    calls: list = []
    monkeypatch.setattr(
        build_context.aiohttp,
        "ClientSession",
        lambda **kwargs: _FakeSession(_FakeResponse(data, **response_kwargs), calls),
    )
    return calls


async def _fetch(url: str = URL, sha256: str | None = None, data: bytes = b"", **limits) -> bytes:
    return await build_context.fetch(
        url,
        sha256 if sha256 is not None else hashlib.sha256(data).hexdigest(),
        hosts=HOSTS,
        max_bytes=limits.get("max_bytes", 1 << 20),
        max_unpacked_bytes=limits.get("max_unpacked_bytes", 1 << 20),
    )


@pytest.mark.asyncio
async def test_fetch_returns_the_archive_when_digest_and_entries_pass(monkeypatch):
    # Arrange
    data = _archive(
        {"app.py": b"print('hi')\n", "src/lib.py": b"x = 1\n"}, links={"latest": "src/lib.py"}
    )
    calls = _serve(monkeypatch, data)

    # Act
    result = await _fetch(data=data)

    # Assert
    assert result == data
    assert calls[0][1]["allow_redirects"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://contexts.example.com/ctx.tar.gz",
        "https://other.example.com/ctx.tar.gz",
        "https://contexts.example.com.evil.test/ctx.tar.gz",
        "https://user@contexts.example.com/ctx.tar.gz",
        "https://contexts.example.com:8443/ctx.tar.gz",
        "file:///etc/passwd",
    ],
)
async def test_fetch_refuses_a_url_off_the_allowed_hosts_without_a_request(monkeypatch, url):
    # Arrange
    calls = _serve(monkeypatch, _archive())

    # Act / Assert
    with pytest.raises(BuildContextError, match="allowed host"):
        await _fetch(url=url, data=_archive())
    assert calls == []


@pytest.mark.asyncio
async def test_fetch_refuses_a_digest_mismatch(monkeypatch):
    # Arrange
    data = _archive()
    _serve(monkeypatch, data)

    # Act / Assert
    with pytest.raises(BuildContextError, match="sha256 does not match"):
        await _fetch(sha256="0" * 64, data=data)


@pytest.mark.asyncio
async def test_fetch_refuses_a_malformed_digest_without_a_request(monkeypatch):
    # Arrange
    calls = _serve(monkeypatch, _archive())

    # Act / Assert
    with pytest.raises(BuildContextError, match="malformed"):
        await _fetch(sha256="abc", data=_archive())
    assert calls == []


@pytest.mark.asyncio
async def test_fetch_stops_reading_past_the_size_cap(monkeypatch):
    # Arrange: no Content-Length, so only the streamed count can catch it
    data = _archive({"big.bin": bytes(range(256)) * 4096})
    _serve(monkeypatch, data)

    # Act / Assert
    with pytest.raises(BuildContextError, match="larger than 100 bytes"):
        await _fetch(data=data, max_bytes=100)


@pytest.mark.asyncio
async def test_fetch_refuses_a_declared_length_past_the_cap(monkeypatch):
    # Arrange
    data = _archive()
    _serve(monkeypatch, data, content_length=10_000)

    # Act / Assert
    with pytest.raises(BuildContextError, match="larger than"):
        await _fetch(data=data, max_bytes=5_000)


@pytest.mark.asyncio
async def test_fetch_refuses_a_non_200(monkeypatch):
    # Arrange
    _serve(monkeypatch, b"", status=403)

    # Act / Assert
    with pytest.raises(BuildContextError, match="HTTP 403"):
        await _fetch(data=b"")


@pytest.mark.parametrize(
    "archive",
    [
        _archive({"../escape": b"x"}),
        _archive({"/etc/motd": b"x"}),
        _archive(links={"etc": "/etc"}),
        _archive(links={"up": "../../x"}),
        _archive(hardlinks={"h": "../x"}),
    ],
)
def test_check_archive_refuses_entries_that_leave_the_context(archive):
    # Act / Assert
    with pytest.raises(BuildContextError, match="leaves the context"):
        build_context.check_archive(archive, 1 << 20)


def test_check_archive_refuses_device_entries():
    # Arrange
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("dev")
        info.type = tarfile.CHRTYPE
        tar.addfile(info)

    # Act / Assert
    with pytest.raises(BuildContextError, match="not a file"):
        build_context.check_archive(buf.getvalue(), 1 << 20)


def test_check_archive_refuses_more_unpacked_bytes_than_the_cap():
    # Arrange: compresses to far less than it unpacks to
    archive = _archive({"zeros.bin": bytes(1 << 20)})

    # Act / Assert
    with pytest.raises(BuildContextError, match="unpacks to more than"):
        build_context.check_archive(archive, 1 << 16)


def test_check_archive_refuses_bytes_that_are_not_a_gzipped_tar():
    # Act / Assert
    with pytest.raises(BuildContextError, match="not a gzipped tar"):
        build_context.check_archive(b"not an archive", 1 << 20)


@pytest.mark.asyncio
async def test_custom_build_unpacks_the_context_before_the_dockerfile(svc, monkeypatch):  # noqa: F811
    # Arrange
    from core.config import settings

    data = _archive()
    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_CONTEXT_ENABLED", True)
    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_CONTEXT_HOSTS", "contexts.example.com")
    _serve(monkeypatch, data)
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(svc, "execute_and_stream_logs", _make_esl())
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nCOPY app.py /app.py\n")
    payload.build_context_url = URL
    payload.build_context_sha256 = hashlib.sha256(data).hexdigest()

    # Act
    ok, step, _tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )

    # Assert
    assert ok is True and step is None
    untar = next(i for i, c in enumerate(ssh_client.calls) if "tar -xzf -" in c)
    write = next(i for i, c in enumerate(ssh_client.calls) if "cat > /build/Dockerfile" in c)
    assert untar < write
    assert ssh_client.call_kwargs[untar]["input"] == data
    assert ssh_client.call_kwargs[untar]["encoding"] is None


@pytest.mark.asyncio
async def test_custom_build_with_a_context_fails_before_the_executor_when_the_setting_is_off(
    svc, monkeypatch  # noqa: F811
):
    # Arrange
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_CONTEXT_ENABLED", False)
    calls = _serve(monkeypatch, _archive())
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nCOPY app.py /app.py\n")
    payload.build_context_url = URL
    payload.build_context_sha256 = "0" * 64

    # Act
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )

    # Assert
    assert ok is False and step == "build_context"
    assert "build contexts" in tail
    assert ssh_client.calls == [] and calls == []


@pytest.mark.asyncio
async def test_custom_build_with_a_refused_context_never_starts_the_build_container(
    svc, monkeypatch  # noqa: F811
):
    # Arrange
    from core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_CONTEXT_ENABLED", True)
    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_CONTEXT_HOSTS", "contexts.example.com")
    _serve(monkeypatch, _archive())
    ssh_client = _make_dind_ssh()
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\nCOPY app.py /app.py\n")
    payload.build_context_url = URL
    payload.build_context_sha256 = "0" * 64

    # Act
    ok, step, tail = await svc._custom_build_image(
        ssh_client=ssh_client,
        payload=payload,
        log_tag="t",
        default_extra={"pod_id": payload.pod_id},
    )

    # Assert
    assert ok is False and step == "build_context"
    assert tail == "build context sha256 does not match"
    assert ssh_client.calls == []
