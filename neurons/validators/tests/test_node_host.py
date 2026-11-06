import asyncio
import logging
import socket
import sys
import threading
import time

import aiohttp
import pytest
from aiohttp import web

from neurons.validators.src.services import node_host
from neurons.validators.src.services.node_host import (
    NodeHostError,
    PublicHostResolver,
    dial_validated,
    for_payload_miner,
    node_owner,
    open_public_connection,
    public_host_connector,
    public_host_sock,
)


def _resolves_to(monkeypatch, *addresses, port=None):
    def fake_getaddrinfo(host, requested_port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port or requested_port)) for a in addresses
        ]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def listener():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    yield server
    server.close()


@pytest.mark.asyncio
async def test_ip_literal_is_not_resolved(monkeypatch):
    # Arrange
    _resolves_to(monkeypatch)  # would raise if consulted

    # Act
    extra = await public_host_sock("10.0.0.5")

    # Assert
    assert extra == {}


@pytest.mark.asyncio
async def test_public_name_returns_a_socket_connected_to_the_validated_address(monkeypatch, listener):
    # Arrange: the name resolves to the local listener, treated as public for this test
    port = listener.getsockname()[1]
    _resolves_to(monkeypatch, "127.0.0.1", port=port)
    monkeypatch.setattr("ipaddress.IPv4Address.is_global", property(lambda self: True))

    # Act
    extra = await public_host_sock("gpu-node1.example.net", port, connect_timeout=5)

    # Assert
    sock = extra["sock"]
    try:
        assert sock.getpeername() == ("127.0.0.1", port)
    finally:
        sock.close()


@pytest.mark.asyncio
async def test_open_public_connection_reaches_the_validated_address(monkeypatch, listener):
    # Arrange: the name resolves to the local listener, treated as public for this test
    port = listener.getsockname()[1]
    _resolves_to(monkeypatch, "127.0.0.1", port=port)
    monkeypatch.setattr("ipaddress.IPv4Address.is_global", property(lambda self: True))

    # Act
    _, writer = await open_public_connection("gpu-node1.example.net", port)

    # Assert
    try:
        assert writer.get_extra_info("peername") == ("127.0.0.1", port)
    finally:
        writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.2"])
async def test_name_with_a_non_public_address_is_refused(monkeypatch, bad):
    # Arrange
    _resolves_to(monkeypatch, "8.8.8.8", bad)

    # Act / Assert
    with pytest.raises(NodeHostError, match="non-public"):
        await public_host_sock("gpu-node1.example.net")


@pytest.mark.asyncio
async def test_unresolvable_name_is_refused(monkeypatch):
    # Arrange
    def fail(host, port, *args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr("socket.getaddrinfo", fail)

    # Act / Assert
    with pytest.raises(NodeHostError, match="does not resolve"):
        await public_host_sock("gone.example.net")


def test_is_ip_literal():
    assert node_host.is_ip_literal("203.0.113.9") and node_host.is_ip_literal("[2001:db8::1]")
    assert not node_host.is_ip_literal("gpu-node1.example.net")


@pytest.mark.asyncio
async def test_resolver_returns_only_the_checked_public_addresses(monkeypatch):
    # Arrange
    _resolves_to(monkeypatch, "8.8.8.8", "1.1.1.1")

    # Act
    results = await PublicHostResolver().resolve("gpu-node1.example.net", 8080, family=socket.AF_UNSPEC)

    # Assert
    assert sorted(r["host"] for r in results) == ["1.1.1.1", "8.8.8.8"]
    assert {(r["hostname"], r["port"]) for r in results} == {("gpu-node1.example.net", 8080)}


@pytest.mark.asyncio
async def test_resolver_refuses_a_name_with_a_metadata_address(monkeypatch):
    # Arrange
    _resolves_to(monkeypatch, "8.8.8.8", "169.254.169.254")

    # Act / Assert
    with pytest.raises(NodeHostError, match="non-public"):
        await PublicHostResolver().resolve("gpu-node1.example.net", 80)


@pytest.mark.asyncio
async def test_http_request_to_a_name_goes_to_the_checked_address_with_the_name_as_host(monkeypatch):
    # Arrange: a local server stands in for the node; the name resolves to it, treated as public
    seen_hosts = []

    async def handler(request):
        seen_hosts.append(request.host)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    lookups = []

    def fake_getaddrinfo(host, requested_port, *args, **kwargs):
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", requested_port))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("ipaddress.IPv4Address.is_global", property(lambda self: True))

    # Act
    try:
        async with aiohttp.ClientSession(connector=public_host_connector()) as session:
            async with session.get(f"http://gpu-node1.example.net:{port}/") as response:
                body = await response.text()
    finally:
        await runner.cleanup()

    # Assert
    assert body == "ok"
    assert seen_hosts == [f"gpu-node1.example.net:{port}"]
    assert lookups == ["gpu-node1.example.net"]


@pytest.mark.asyncio
async def test_http_request_to_a_name_with_a_private_address_is_never_sent(monkeypatch):
    # Arrange
    _resolves_to(monkeypatch, "10.0.0.5")

    # Act / Assert
    async with aiohttp.ClientSession(connector=public_host_connector()) as session:
        with pytest.raises(aiohttp.ClientConnectorError):
            await session.get("http://gpu-node1.example.net:8080/")


def _logged(caplog, level):
    return [(r.getMessage(), r.msg.extra) for r in caplog.records if r.levelno == level]


@pytest.mark.asyncio
async def test_pinned_dial_logs_the_address_dialed_for_the_name(monkeypatch, listener, caplog):
    # Arrange: the name resolves to the local listener, treated as public for this test
    port = listener.getsockname()[1]
    _resolves_to(monkeypatch, "127.0.0.1", port=port)
    monkeypatch.setattr("ipaddress.IPv4Address.is_global", property(lambda self: True))

    # Act
    with caplog.at_level(logging.INFO, logger=node_host.logger.name):
        extra = await public_host_sock("gpu-node1.example.net", port, connect_timeout=5)
    extra["sock"].close()

    # Assert
    [(message, fields)] = _logged(caplog, logging.INFO)
    assert message == f"Dialed hostname node gpu-node1.example.net:{port} at 127.0.0.1"
    assert (fields["host"], fields["port"], fields["address"]) == ("gpu-node1.example.net", port, "127.0.0.1")


@pytest.mark.asyncio
async def test_blocking_dial_logs_the_address_dialed_for_the_name(monkeypatch, listener, caplog):
    # Arrange: the name resolves to the local listener, treated as public for this test
    port = listener.getsockname()[1]
    _resolves_to(monkeypatch, "127.0.0.1", port=port)
    monkeypatch.setattr("ipaddress.IPv4Address.is_global", property(lambda self: True))
    infos = await node_host.resolve_public_host("gpu-node1.example.net", port)

    # Act
    with caplog.at_level(logging.INFO, logger=node_host.logger.name):
        dial_validated(infos, 5).close()

    # Assert
    [(message, fields)] = _logged(caplog, logging.INFO)
    assert message == f"Dialed hostname node gpu-node1.example.net:{port} at 127.0.0.1"
    assert (fields["host"], fields["port"], fields["address"]) == ("gpu-node1.example.net", port, "127.0.0.1")


@pytest.mark.asyncio
async def test_resolver_logs_the_checked_addresses_for_the_name(monkeypatch, caplog):
    # Arrange
    _resolves_to(monkeypatch, "8.8.8.8", "1.1.1.1")

    # Act
    with caplog.at_level(logging.INFO, logger=node_host.logger.name):
        await PublicHostResolver().resolve("gpu-node1.example.net", 8080, family=socket.AF_UNSPEC)

    # Assert
    [(message, fields)] = _logged(caplog, logging.INFO)
    assert message == "Resolved hostname node gpu-node1.example.net:8080 to 1.1.1.1, 8.8.8.8"
    assert (fields["host"], fields["port"], fields["addresses"]) == (
        "gpu-node1.example.net",
        8080,
        ["1.1.1.1", "8.8.8.8"],
    )


@pytest.mark.asyncio
async def test_refused_name_logs_a_warning_with_the_offending_address(monkeypatch, caplog):
    # Arrange
    _resolves_to(monkeypatch, "8.8.8.8", "10.0.0.5")

    # Act
    with caplog.at_level(logging.INFO, logger=node_host.logger.name), pytest.raises(NodeHostError):
        await public_host_sock("gpu-node1.example.net", 22)

    # Assert
    [(message, fields)] = _logged(caplog, logging.WARNING)
    assert message == "Refused hostname node gpu-node1.example.net:22: resolves to a non-public address: 10.0.0.5"
    assert (fields["host"], fields["addresses"]) == ("gpu-node1.example.net", ["10.0.0.5"])
    assert _logged(caplog, logging.INFO) == []


@pytest.mark.asyncio
async def test_unresolvable_name_logs_a_warning_with_the_error(monkeypatch, caplog):
    # Arrange
    def fail(host, port, *args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr("socket.getaddrinfo", fail)

    # Act
    with caplog.at_level(logging.WARNING, logger=node_host.logger.name), pytest.raises(NodeHostError):
        await public_host_sock("gone.example.net", 22)

    # Assert
    [(message, fields)] = _logged(caplog, logging.WARNING)
    assert message == "Refused hostname node gone.example.net:22: does not resolve: [Errno -2] Name or service not known"
    assert fields["host"] == "gone.example.net"


@pytest.mark.asyncio
async def test_ip_literal_logs_nothing(caplog):
    # Arrange
    resolver = PublicHostResolver()

    # Act
    with caplog.at_level(logging.INFO, logger=node_host.logger.name):
        await public_host_sock("203.0.113.9")
        await resolver.resolve("203.0.113.9", 8080)

    # Assert
    assert caplog.records == []


@pytest.fixture
def stalled_dns(monkeypatch):
    """socket.getaddrinfo that never answers until the test ends; records the thread each lookup ran in."""
    release, threads = threading.Event(), []

    def stalled(*args, **kwargs):
        threads.append(threading.current_thread().name)
        release.wait(5)
        raise socket.gaierror(-3, "Temporary failure in name resolution")

    monkeypatch.setattr("socket.getaddrinfo", stalled)
    yield threads
    release.set()


@pytest.mark.asyncio
async def test_a_stalled_lookup_is_bounded_by_the_connect_timeout(stalled_dns):
    with pytest.raises(TimeoutError):
        await public_host_sock("gpu-node1.example.net", 22, connect_timeout=0.05)


@pytest.mark.asyncio
async def test_a_stalled_lookup_without_a_connect_timeout_is_refused_at_the_dns_deadline(stalled_dns, monkeypatch):
    # most callers (miner_service, docker_service, interactive_shell) pass no connect_timeout: before, they waited as long as the miner's DNS did
    monkeypatch.setattr(node_host, "NODE_DNS_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(NodeHostError, match="did not resolve within 0.05 s"):
        await public_host_sock("gpu-node1.example.net", 22)


@pytest.mark.asyncio
async def test_a_stalled_resolver_lookup_is_refused_at_the_dns_deadline(stalled_dns, monkeypatch):
    # the aiohttp resolver (port checks, local verify) had no deadline at all
    monkeypatch.setattr(node_host, "NODE_DNS_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(NodeHostError, match="did not resolve"):
        await PublicHostResolver().resolve("gpu-node1.example.net", 8080)


@pytest.mark.asyncio
async def test_stalled_node_lookups_never_occupy_the_default_executor(stalled_dns, monkeypatch):
    # a miner zone that never answers must not fill the pool the validator's own DNS (Redis, WebSocket) resolves in
    monkeypatch.setattr(node_host, "NODE_DNS_TIMEOUT_SECONDS", 0.05)
    loop = asyncio.get_running_loop()
    default_pool_used = []
    real_run_in_executor = loop.run_in_executor

    def watch(executor, *args):
        if executor is None:
            default_pool_used.append(args)
        return real_run_in_executor(executor, *args)

    monkeypatch.setattr(loop, "run_in_executor", watch)

    for _ in range(3):
        with pytest.raises(NodeHostError):
            await public_host_sock("gpu-node1.example.net", 22)
    with pytest.raises(NodeHostError):
        await PublicHostResolver().resolve("gpu-node1.example.net", 8080)

    assert default_pool_used == []
    assert stalled_dns and all(name.startswith("node-dns") for name in stalled_dns)


def _stall_names_containing(monkeypatch, marker):
    """socket.getaddrinfo that holds every name containing ``marker`` until released; others resolve to 1.1.1.1."""
    release = threading.Event()

    def lookup(host, port, *args, **kwargs):
        if marker in host:
            release.wait(5)
            raise socket.gaierror(-3, "Temporary failure in name resolution")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]

    monkeypatch.setattr("socket.getaddrinfo", lookup)
    return release


async def _resolve_as(miner, host):
    with node_owner(miner):
        return await PublicHostResolver().resolve(host, 8080)


@pytest.mark.asyncio
async def test_a_miner_whose_dns_stalls_cannot_stop_another_miners_name_resolving(monkeypatch):
    # review of #1609: hostile hostname executors behind a deliberately slow zone filled the shared pool, and
    # every other miner's hostname node then hit the deadline and was hidden. Spreading the names over many
    # domains must not help: the quota belongs to the authenticated miner, not to anything in the name.
    monkeypatch.setattr(node_host, "NODE_DNS_TIMEOUT_SECONDS", 0.3)
    release = _stall_names_containing(monkeypatch, "evil")
    try:
        hostile = [
            asyncio.create_task(_resolve_as("hostile-hotkey", f"gpu.evil{i}.example"))
            for i in range(node_host.NODE_DNS_MAX_WORKERS + 8)
        ]
        await asyncio.sleep(0.05)

        [honest] = await _resolve_as("honest-hotkey", "gpu-node1.good.example")

        assert honest["host"] == "1.1.1.1"
        results = await asyncio.gather(*hostile, return_exceptions=True)
        assert all(isinstance(r, NodeHostError) for r in results)
        assert node_host._dns_busy.get("miner:hostile-hotkey", 0) <= node_host.NODE_DNS_PER_MINER
    finally:
        release.set()


@pytest.mark.asyncio
async def test_names_under_a_shared_public_suffix_do_not_share_a_quota(monkeypatch):
    # review of #1609: a key of the last two labels put every *.co.uk (and *.duckdns.org) name in one bucket,
    # so two stalled evil.co.uk lookups blocked gpu.good.co.uk
    monkeypatch.setattr(node_host, "NODE_DNS_TIMEOUT_SECONDS", 0.3)
    release = _stall_names_containing(monkeypatch, "evil")
    try:
        hostile = [asyncio.create_task(_resolve_as("hostile-hotkey", f"gpu{i}.evil.co.uk")) for i in range(8)]
        anonymous = [asyncio.create_task(_resolve_as(None, f"gpu{i}.evil.co.uk")) for i in range(8)]
        await asyncio.sleep(0.05)

        [honest] = await _resolve_as("honest-hotkey", "gpu.good.co.uk")
        [no_miner] = await _resolve_as(None, "gpu2.good.co.uk")

        assert honest["host"] == no_miner["host"] == "1.1.1.1"
        await asyncio.gather(*hostile, *anonymous, return_exceptions=True)
    finally:
        release.set()


def test_the_quota_key_is_the_miner_and_otherwise_the_whole_name():
    assert node_host.dns_quota_key("gpu.Evil.co.uk.") == "name:gpu.evil.co.uk"
    with node_owner("5Hotkey"):
        assert node_host.dns_quota_key("gpu.evil.co.uk") == "miner:5Hotkey"
    assert node_host.dns_quota_key("gpu.evil.co.uk") == "name:gpu.evil.co.uk"


@pytest.mark.asyncio
async def test_per_miner_handlers_attribute_their_tasks_lookups_to_the_payload_miner():
    class Payload:
        miner_hotkey = "5Hotkey"

    class Service:
        @for_payload_miner
        async def handle(self, payload, host):
            # the validation cycle runs each node in its own task: those lookups count against the miner too
            return await asyncio.create_task(asyncio.sleep(0, node_host.dns_quota_key(host)))

    assert await Service().handle(Payload(), "gpu.example") == "miner:5Hotkey"
    assert node_host.dns_quota_key("gpu.example") == "name:gpu.example"


@pytest.mark.asyncio
async def test_a_port_check_of_one_host_resolves_it_once_and_dials_each_port(monkeypatch):
    # review of #1609: the port check dials up to 300 ports of a node at once, each a lookup through 4 slots
    calls = []

    def lookup(host, port, *args, **kwargs):
        calls.append(host)
        time.sleep(0.05)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]

    monkeypatch.setattr("socket.getaddrinfo", lookup)
    ports = range(20000, 20300)

    results = await asyncio.gather(*(_resolve_port("miner-hotkey", "gpu.node.example", p) for p in ports))

    assert calls == ["gpu.node.example"]
    assert [result["port"] for [result] in results] == list(ports)


async def _resolve_port(miner, host, port):
    with node_owner(miner):
        return await PublicHostResolver().resolve(host, port)


@pytest.mark.asyncio
async def test_a_lookup_waiting_for_its_miners_slot_sleeps_until_one_is_given_back(monkeypatch):
    # review of #1609: waiters polled the loop every 20 ms for as long as the miner's lookups stalled
    release = _stall_names_containing(monkeypatch, "evil")
    polls = []
    real_sleep = asyncio.sleep

    async def sleep(delay, *args, **kwargs):
        if sys._getframe(1).f_globals.get("__name__", "").endswith("node_host"):
            polls.append(delay)
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    try:
        stalled = [
            asyncio.create_task(_resolve_as("miner-hotkey", f"gpu{i}.evil.example"))
            for i in range(node_host.NODE_DNS_PER_MINER)
        ]
        await real_sleep(0.05)
        waiting = asyncio.create_task(_resolve_as("miner-hotkey", "gpu.good.example"))
        await real_sleep(0.2)
        assert not waiting.done() and polls == []

        release.set()
        [resolved] = await asyncio.wait_for(waiting, 1)

        assert resolved["host"] == "1.1.1.1"
        await asyncio.gather(*stalled, return_exceptions=True)
    finally:
        release.set()
