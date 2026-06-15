"""DAH-2211 — BuildLogShipper unit tests.

Pins the contract the backend's `POST /pods/{pod_id}/build-log-chunk`
endpoint expects (auth shape, URL, payload), and the best-effort guarantees:

- Backend down / 5xx / signing error → logs once, never raises, never blocks.
- Lines are batched and posted under back-pressure.
- `stop()` drains the queue and flushes the buffer.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


def _fake_keypair(ss58: str = "5Hvalidator0000000000000000000000000000000000000000"):
    kp = MagicMock()
    kp.ss58_address = ss58
    kp.sign = MagicMock(return_value=b"\xde\xad\xbe\xef")
    return kp


class _MockResp:
    def __init__(self, status: int = 200):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _MockSession:
    """Captures posts; never actually opens a connection."""

    def __init__(self, status: int = 200):
        self._status = status
        self.posts: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _MockResp(self._status)


@pytest.mark.asyncio
async def test_shipper_posts_chunk_with_validator_headers():
    """Shipper hits /pods/{pod_id}/build-log-chunk with hotkey/timestamp/signature
    headers and a payload matching BuildLogChunkRequest (chunk, seq, phase).
    """
    from services.build_log_shipper import BuildLogShipper

    pod_id = uuid4()
    kp = _fake_keypair()
    session = _MockSession()

    with patch("aiohttp.ClientSession", return_value=session):
        shipper = BuildLogShipper(pod_id=pod_id, keypair=kp, backend_url="http://backend.test")
        await shipper.start()
        await shipper.append("Step 1/2 : FROM alpine", phase="build")
        await shipper.append("Step 2/2 : RUN echo hi", phase="build")
        await shipper.stop()

    assert len(session.posts) == 1, f"expected 1 batched POST; got {len(session.posts)}"
    posted = session.posts[0]
    assert posted["url"] == f"http://backend.test/pods/{pod_id}/build-log-chunk"
    # Auth header shape (matches utils.auth `_verify_validator_headers`).
    assert posted["headers"]["hotkey"] == kp.ss58_address
    assert posted["headers"]["timestamp"].isdigit()
    assert posted["headers"]["signature"] == "0xdeadbeef"
    body = posted["json"]
    assert body["seq"] == 1
    assert body["phase"] == "build"
    assert "Step 1/2 : FROM alpine" in body["chunk"]
    assert "Step 2/2 : RUN echo hi" in body["chunk"]
    # Signing input is the timestamp string (not the payload), per existing scheme.
    kp.sign.assert_called_with(posted["headers"]["timestamp"].encode())


@pytest.mark.asyncio
async def test_shipper_backend_down_does_not_raise():
    """A backend that 5xxs or refuses connection must not raise out of append/stop."""
    from services.build_log_shipper import BuildLogShipper

    pod_id = uuid4()
    kp = _fake_keypair()

    # aiohttp raises on connection refused — simulate by patching the session
    # itself to raise inside __aenter__.
    class _Boom:
        async def __aenter__(self):
            raise ConnectionError("backend down")

        async def __aexit__(self, *a):
            return False

    with patch("aiohttp.ClientSession", return_value=_Boom()):
        shipper = BuildLogShipper(pod_id=pod_id, keypair=kp, backend_url="http://backend.test")
        await shipper.start()
        await shipper.append("line that will fail to ship")
        # stop() must not raise — best-effort guarantee.
        await shipper.stop()


@pytest.mark.asyncio
async def test_shipper_no_backend_url_is_silent_noop():
    """Without a backend URL configured (e.g. dev / local validator) the shipper
    must accept appends and stop cleanly without issuing any HTTP.
    """
    from services.build_log_shipper import BuildLogShipper

    pod_id = uuid4()
    kp = _fake_keypair()
    session = _MockSession()
    # Pass an explicit empty string to force the "no backend URL" branch — bypasses
    # the settings-fallback default the production caller relies on.
    with patch("aiohttp.ClientSession", return_value=session):
        shipper = BuildLogShipper(pod_id=pod_id, keypair=kp, backend_url="")
        await shipper.start()
        await shipper.append("ignored")
        await shipper.stop()
    assert session.posts == []


@pytest.mark.asyncio
async def test_shipper_flushes_remaining_buffer_on_stop():
    """`stop()` must flush whatever's buffered, even below the batch size."""
    from services.build_log_shipper import BuildLogShipper

    pod_id = uuid4()
    kp = _fake_keypair()
    session = _MockSession()

    with patch("aiohttp.ClientSession", return_value=session):
        shipper = BuildLogShipper(pod_id=pod_id, keypair=kp, backend_url="http://backend.test")
        await shipper.start()
        await shipper.append("one")
        await shipper.append("two")
        await shipper.stop()

    assert len(session.posts) == 1
    assert "one" in session.posts[0]["json"]["chunk"]
    assert "two" in session.posts[0]["json"]["chunk"]


@pytest.mark.asyncio
async def test_shipper_phase_shift_mid_buffer_flushes_first_phase():
    """Switching phase mid-buffer should flush the prior-phase batch before
    starting the new one — keeps the backend's phase tag accurate.
    """
    from services.build_log_shipper import BuildLogShipper

    pod_id = uuid4()
    kp = _fake_keypair()
    session = _MockSession()

    with patch("aiohttp.ClientSession", return_value=session):
        shipper = BuildLogShipper(pod_id=pod_id, keypair=kp, backend_url="http://backend.test")
        await shipper.start()
        await shipper.append("stdout 1", phase="build")
        await shipper.append("stderr 1", phase="build_err")
        await shipper.stop()

    # Two POSTs — one per phase batch.
    assert len(session.posts) == 2
    phases = [p["json"]["phase"] for p in session.posts]
    assert phases == ["build", "build_err"]
