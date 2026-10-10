import socket

import pytest

from neurons.validators.src.services import node_host
from neurons.validators.src.services.node_host import NodeHostError, require_public_host


def _resolves_to(monkeypatch, *addresses):
    async def fake_getaddrinfo(self, host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)


@pytest.mark.asyncio
async def test_ip_literal_is_not_resolved(monkeypatch):
    _resolves_to(monkeypatch)  # would raise if consulted
    await require_public_host("10.0.0.5")


@pytest.mark.asyncio
async def test_public_name_passes(monkeypatch):
    _resolves_to(monkeypatch, "8.8.8.8")
    await require_public_host("gpu-node1.example.net")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.2"])
async def test_name_with_a_non_public_address_is_refused(monkeypatch, bad):
    _resolves_to(monkeypatch, "8.8.8.8", bad)
    with pytest.raises(NodeHostError, match="non-public"):
        await require_public_host("gpu-node1.example.net")


@pytest.mark.asyncio
async def test_unresolvable_name_is_refused(monkeypatch):
    async def fail(self, host, port, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fail)
    with pytest.raises(NodeHostError, match="does not resolve"):
        await require_public_host("gone.example.net")


def test_is_ip_literal():
    assert node_host.is_ip_literal("203.0.113.9") and node_host.is_ip_literal("[2001:db8::1]")
    assert not node_host.is_ip_literal("gpu-node1.example.net")
