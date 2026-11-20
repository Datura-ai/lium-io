"""DAH-6782: MinerService._make_rest_request sends only to a public address and never follows a redirect.

Outside test_node_host.py the conftest lets loopback stand in for a public node, so the local servers here are "public";
0.0.0.0 and a unique-local IPv6 address stay non-public."""

import aiohttp
import pytest
from aiohttp import web
from services.miner_service import MinerService
from services.node_host import NodeHostError

from core.config import settings


async def _serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def _request(url):
    return await MinerService._make_rest_request(
        MinerService.__new__(MinerService), "POST", url, {"a": 1}, {}, 5, {}, "test"
    )


@pytest.mark.asyncio
async def test_a_public_axon_gets_the_request():
    seen = []

    async def handler(request):
        seen.append(await request.json())
        return web.json_response({"ok": True})

    runner, port = await _serve(handler)
    try:
        assert await _request(f"http://127.0.0.1:{port}/executors") == (200, {"ok": True})
    finally:
        await runner.cleanup()
    assert seen == [{"a": 1}]


@pytest.mark.asyncio
async def test_a_redirect_is_not_followed():
    reached = []

    async def target(request):
        reached.append(request.path)
        return web.json_response({})

    target_runner, target_port = await _serve(target)

    async def redirecting(request):
        return web.json_response({}, status=307, headers={"Location": f"http://127.0.0.1:{target_port}/x"})

    runner, port = await _serve(redirecting)
    try:
        status, _ = await _request(f"http://127.0.0.1:{port}/executors")
    finally:
        await runner.cleanup()
        await target_runner.cleanup()
    assert status == 307
    assert reached == []


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["0.0.0.0", "[fd00::5]"])
async def test_a_non_public_axon_is_refused_with_nothing_sent(monkeypatch, host):
    def no_session(*args, **kwargs):
        raise AssertionError("a session was opened")

    monkeypatch.setattr(aiohttp, "ClientSession", no_session)
    with pytest.raises(NodeHostError):
        await _request(f"http://{host}:8080/executors")


def _debug_local_miner(monkeypatch, host, port):
    monkeypatch.setattr(settings.debug, "USE_LOCAL_MINER", True)
    monkeypatch.setattr(settings.debug, "MINER_ADDRESS", host)
    monkeypatch.setattr(settings.debug, "MINER_PORT", port)


def _configured_origin(monkeypatch, host, port):
    monkeypatch.setattr(settings, "CONFIGURED_MINER_ORIGINS", f"miner.example.net:9000, {host}:{port}")


@pytest.mark.asyncio
@pytest.mark.parametrize("configure", [_debug_local_miner, _configured_origin], ids=["debug-local-miner", "origin"])
async def test_a_miner_the_operator_configured_is_reached_at_its_private_address(monkeypatch, configure):
    """A miner the validator's operator set up (a stand's local miner, a dev copy's in-cluster central miner) is
    dialled at its private address; any other private address, even on the same host, is still refused."""

    async def handler(request):
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(NodeHostError):
            await _request(f"http://0.0.0.0:{port}/executors")
        configure(monkeypatch, "0.0.0.0", port)
        assert await _request(f"http://0.0.0.0:{port}/executors") == (200, {"ok": True})
        with pytest.raises(NodeHostError):
            await _request(f"http://0.0.0.0:{port + 1}/executors")
    finally:
        await runner.cleanup()
