"""DAH-6782: MinerService._make_rest_request sends only to a public address and never follows a redirect.

Outside test_node_host.py the conftest lets loopback stand in for a public node, so the local servers here are "public";
0.0.0.0 and a unique-local IPv6 address stay non-public."""

import aiohttp
import pytest
from aiohttp import web
from services.miner_service import MinerService
from services.node_host import NodeHostError


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
