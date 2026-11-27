"""The miner's requests to a node never reach its own network (services/node_address.py)."""

import ipaddress
import socket

import aiohttp
import pytest
from aiohttp import web

from services import node_address
from services.node_address import NonPublicNodeAddress, executor_session, is_public_address


def _answers(*hosts, port=0):
    return [
        {"hostname": "node.example.com", "host": host, "port": port, "family": socket.AF_INET, "proto": 0, "flags": 0}
        for host in hosts
    ]


@pytest.fixture
async def local_node():
    """An HTTP server on 127.0.0.1 that records the requests it gets; the stand-in for a node."""
    seen = []

    async def handle(request):
        seen.append(request.path)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield port, seen
    await runner.cleanup()


@pytest.mark.parametrize(
    "address",
    # private, CGNAT and reserved IPv4 ranges are the stdlib's is_global; IPv4 literals here are the scan-allowed ones
    ["127.0.0.1", "169.254.169.254", "0.0.0.0", "192.0.2.1", "::1", "fe80::1%eth0", "fc00::1", "ff02::1", "ff0e::1",
     "::ffff:127.0.0.1", "::ffff:169.254.169.254",
     # Python 3.11-3.12 call these global: IPv6 site-local, and reserved ranges incl. NAT64 of 127.0.0.1
     "fec0::1", "fec0::1%eth0", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "100::1"],
)
def test_internal_addresses_are_not_public(address):
    assert not is_public_address(address)


def test_a_dev_stack_admits_only_its_own_private_network(monkeypatch):
    assert not is_public_address("10.20.30.40")  # secret-scan: allow
    monkeypatch.setattr(node_address, "ALLOWED_PRIVATE_NETWORKS", (ipaddress.ip_network("10.20.0.0/16"),))  # secret-scan: allow
    assert is_public_address("10.20.30.40")  # secret-scan: allow
    assert not is_public_address("10.21.0.1") and not is_public_address("192.168.7.7")  # secret-scan: allow
    assert not is_public_address("127.0.0.1") and not is_public_address("169.254.169.254")


@pytest.mark.parametrize("address", ["::ffff:10.20.30.40", "64:ff9b::a14:1e28"])  # secret-scan: allow
def test_a_dev_stack_admits_the_mapped_and_nat64_forms_of_its_network(monkeypatch, address):
    monkeypatch.setattr(node_address, "ALLOWED_PRIVATE_NETWORKS", (ipaddress.ip_network("10.20.0.0/16"),))  # secret-scan: allow
    assert is_public_address(address)


def test_public_addresses_are_public():
    assert is_public_address("1.2.3.4") and is_public_address("2001:4860:4860::8888")
    # lium-io#2369 review 5981962325: an IPv6-only host's DNS64 answer for a public A record stays reachable
    assert is_public_address("64:ff9b::808:808")


@pytest.mark.asyncio
@pytest.mark.parametrize("answers", [["127.0.0.1"], ["fc00::1"], ["169.254.169.254"], ["::1"], ["1.2.3.4", "127.0.0.1"]])
async def test_a_name_resolving_inside_is_refused_without_a_connection(monkeypatch, local_node, answers):
    port, seen = local_node

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return _answers(*answers)

    monkeypatch.setattr(node_address.DefaultResolver, "resolve", resolve)
    async with executor_session() as session:
        with pytest.raises(NonPublicNodeAddress):
            await session.post(f"http://node.example.com:{port}/upload_ssh_key", json={}, allow_redirects=False)
    assert seen == []


@pytest.mark.asyncio
async def test_the_connection_goes_to_the_first_lookups_answer_not_a_second_one(monkeypatch, local_node):
    """A rebinding name answers a public address, then a private one. The miner dials what the checked lookup
    returned and never looks the name up again for that request. (127.0.0.1 stands in for the public answer here,
    so the request can land on the local server; 169.254.169.254 is the rebound answer.)"""
    port, seen = local_node
    lookups = []

    async def resolve(self, host, port=0, family=socket.AF_INET):
        lookups.append(host)
        return _answers("127.0.0.1" if len(lookups) == 1 else "169.254.169.254", port=port)

    monkeypatch.setattr(node_address.DefaultResolver, "resolve", resolve)
    monkeypatch.setattr(node_address, "is_public_address", lambda address: address == "127.0.0.1")
    async with executor_session() as session:
        async with session.post(f"http://node.example.com:{port}/upload_ssh_key", json={}, allow_redirects=False) as response:
            assert response.status == 200
    assert seen == ["/upload_ssh_key"] and lookups == ["node.example.com"]


@pytest.mark.asyncio
async def test_a_redirect_from_the_node_is_not_followed(monkeypatch):
    """A node answering 307 to http://127.0.0.1/... would otherwise send the miner's POST inside."""
    inside = []

    async def target(request):
        inside.append(request.path)
        return web.json_response({})

    async def redirect(request):
        raise web.HTTPTemporaryRedirect(f"http://127.0.0.1:{target_port}/admin")

    runners = []
    ports = []
    for handler in (target, redirect):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        ports.append(site._server.sockets[0].getsockname()[1])
    target_port, node_port = ports
    try:
        async with executor_session() as session:
            # an IP literal is dialled as given (the node registry only takes public ones)
            async with session.post(f"http://127.0.0.1:{node_port}/upload_ssh_key", json={}, allow_redirects=False) as response:
                assert response.status == 307
            # a request that does not pass allow_redirects=False still ends at the redirect
            with pytest.raises(aiohttp.ClientConnectionError, match="redirect"):
                await session.post(f"http://127.0.0.1:{node_port}/upload_ssh_key", json={})
        assert inside == []
    finally:
        for runner in runners:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_a_public_name_is_dialled(monkeypatch):
    """A name whose every answer is public reaches the connector unchanged."""

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return _answers("1.2.3.4", "5.6.7.8")

    monkeypatch.setattr(node_address.DefaultResolver, "resolve", resolve)
    resolver = node_address.PublicOnlyResolver()
    assert [answer["host"] for answer in await resolver.resolve("node.example.com", 8080)] == ["1.2.3.4", "5.6.7.8"]
    await resolver.close()


@pytest.mark.asyncio
async def test_every_request_to_a_node_uses_the_checked_session(monkeypatch):
    """Each executor_service call to a node builds its session through executor_session()."""
    import inspect

    from services import executor_service

    source = inspect.getsource(executor_service)
    assert "aiohttp.ClientSession(" not in source
    assert source.count("allow_redirects=False") == source.count("session.post(") + source.count("session.get(")


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254", "[::1]", "[fec0::1]", "[64:ff9b::7f00:1]", "[::ffff:127.0.0.1]"])
async def test_a_central_miner_never_dials_a_stored_non_public_ip_literal(monkeypatch, local_node, host):
    """sergej-bot 18:57Z: aiohttp dials an IP literal without the resolver, and the central miner's addresses come
    from the portal snapshot. Every request's host is checked before it is sent; nothing reaches the node."""
    port, seen = local_node
    monkeypatch.setattr(node_address.settings, "CENTRAL_MODE", True)
    async with executor_session() as session:
        for method in (session.get, session.post):
            with pytest.raises(NonPublicNodeAddress):
                await method(f"http://{host}:{port}/upload_ssh_key", allow_redirects=False)
    assert seen == []


@pytest.mark.asyncio
async def test_a_self_hosted_miner_still_dials_its_operators_private_executor(monkeypatch, local_node):
    """A self-hosted miner's executors are its own operator's; one on a private network keeps working."""
    port, seen = local_node
    monkeypatch.setattr(node_address.settings, "CENTRAL_MODE", False)
    async with executor_session() as session:
        async with session.get(f"http://127.0.0.1:{port}/health", allow_redirects=False) as response:
            assert response.status == 200
    assert seen == ["/health"]
