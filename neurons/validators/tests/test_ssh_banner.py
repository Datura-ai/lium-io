"""ssh_banner.ssh_banner_error against real loopback sockets (DAH-2255).

The rented-state check and the rental probe both decide from this one read whether a renter's ssh client
would get past the TCP accept. Each case below is a server behaviour seen on a provider host.
"""
from __future__ import annotations

import asyncio
import socket

import pytest
from neurons.validators.src.services.task.checks.ssh_banner import (
    BANNER_NO_BANNER,
    BANNER_REFUSED,
    BANNER_TIMEOUT,
    ssh_banner_error,
)

BANNER = b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n"


async def serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def read(port: int, *, banner_timeout: float = 0.3):
    return await ssh_banner_error("127.0.0.1", port, connect_timeout=1.0, banner_timeout=banner_timeout)


@pytest.mark.asyncio
async def test_sshd_banner_passes():
    async def sshd(reader, writer):
        writer.write(BANNER)
        await writer.drain()
        writer.close()

    server, port = await serve(sshd)
    async with server:
        assert await read(port) is None


@pytest.mark.asyncio
async def test_nothing_listening_is_refused():
    """The prod case of 17 Sep 2026: `docker ps` running, `ssh -p <port>` → Connection refused."""
    error = await read(free_port())
    assert error is not None
    assert error[0] == BANNER_REFUSED


@pytest.mark.asyncio
async def test_accept_then_close_without_bytes_is_no_banner():
    """docker-proxy in front of a container whose sshd is not running: the TCP accept succeeds, the
    connection closes with nothing on it. A port check would call this reachable."""

    async def proxy_with_dead_backend(reader, writer):
        writer.close()

    server, port = await serve(proxy_with_dead_backend)
    async with server:
        error = await read(port)
    assert error is not None
    assert error[0] == BANNER_NO_BANNER


@pytest.mark.asyncio
async def test_accept_then_silence_is_a_timeout():
    async def silent(reader, writer):
        await asyncio.sleep(2)
        writer.close()

    server, port = await serve(silent)
    async with server:
        error = await read(port, banner_timeout=0.2)
    assert error is not None
    assert error[0] == BANNER_TIMEOUT
    assert "sent no banner" in error[1]


@pytest.mark.asyncio
async def test_a_server_that_is_not_ssh_is_no_banner():
    async def http(reader, writer):
        writer.write(b"HTTP/1.1 400 Bad Request\r\n")
        await writer.drain()
        writer.close()

    server, port = await serve(http)
    async with server:
        error = await read(port)
    assert error is not None
    assert error[0] == BANNER_NO_BANNER
    assert "HTTP/1.1" in error[1]
