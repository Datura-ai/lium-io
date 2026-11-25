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


@pytest.mark.asyncio
async def test_a_refused_literal_axon_logs_no_error(monkeypatch):
    logged = []
    monkeypatch.setattr("services.miner_service.logger.error", lambda *a, **k: logged.append(a))
    with pytest.raises(NodeHostError):
        await _request("http://0.0.0.0:8080/executors")
    assert logged == []


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


@pytest.mark.asyncio
async def test_a_malformed_configured_origin_breaks_no_other_request(monkeypatch):
    """A typo in CONFIGURED_MINER_ORIGINS (a port that is not a number) is skipped: the well-formed entry still counts
    and a request to any public miner still goes out (stand check 2381-03)."""
    monkeypatch.setattr(
        settings,
        "CONFIGURED_MINER_ORIGINS",
        "miner.example.net:abc, miner.example.net:\u00b2, good.example.net:9000, :1, x:0, y:70000, z:80/path, a@192.0.2.1:80, "
        "b.example.net:8000/, c.example.net:8_000, d.example.net:80?, e.example.net:80#, f.example.net:+80",
    )
    assert settings.configured_miner_origins() == {("good.example.net", 9000)}

    async def handler(request):
        return web.json_response({"ok": True})

    runner, port = await _serve(handler)
    try:
        assert await _request(f"http://127.0.0.1:{port}/executors") == (200, {"ok": True})
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "written, url",
    [
        ("Miner.Example.NET:9000", "http://miner.example.net:9000/x"),
        ("miner.example.net:9000", "http://MINER.Example.net:9000/x"),
        ("[0:0::1]:9000", "http://[::1]:9000/x"),
        ("::1:9000", "http://[::1]:9000/x"),
    ],
)
def test_a_configured_origin_matches_however_its_host_is_written(monkeypatch, written, url):
    """Stand check 2381-03: the entry is compared in the form the request URL's host takes."""
    import yarl

    monkeypatch.setattr(settings, "CONFIGURED_MINER_ORIGINS", written)
    target = yarl.URL(url)
    assert (target.host, target.port) in settings.configured_miner_origins()


@pytest.mark.parametrize(
    "written, url",
    [
        ("Miner.Example.NET", "http://miner.example.net:9000/x"),
        ("[fd00:0:0:0:0:0:0:5]", "http://[fd00::5]:9000/x"),
        ("[fd00::5]", "http://[fd00::5]:9000/x"),
        ("fd00::5", "http://[fd00::5]:9000/x"),
    ],
)
def test_the_debug_local_miner_matches_however_its_address_is_written(monkeypatch, written, url):
    """Stand check 2381-03: MINER_ADDRESS is parsed like a configured origin."""
    import yarl

    _debug_local_miner(monkeypatch, written, 9000)
    target = yarl.URL(url)
    assert (target.host, target.port) in settings.configured_miner_origins()


@pytest.mark.asyncio
async def test_a_userinfo_entry_lets_no_miner_reach_a_private_address(monkeypatch):
    """Stand check 2381-03: 'x@0.0.0.0:P' is not host:port, so it configures nothing; a miner reporting 0.0.0.0:P is
    refused and the server there gets no request."""
    seen = []

    async def handler(request):
        seen.append(request.path)
        return web.json_response({"ok": True})

    runner, port = await _serve(handler)
    try:
        monkeypatch.setattr(settings, "CONFIGURED_MINER_ORIGINS", f"attacker@0.0.0.0:{port}")
        assert settings.configured_miner_origins() == set()
        with pytest.raises(NodeHostError):
            await _request(f"http://0.0.0.0:{port}/executors")
        assert seen == []
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_bare_ipv6_axon_reaches_the_address_check(monkeypatch):
    """A miner reports a bare IPv6 axon address; the URL the callers build brackets it, so the request reaches the
    public-address check (refused here: fd00::/8 is not public) instead of failing on an unparsable URL."""
    from services.miner_service import _miner_base_url

    assert _miner_base_url("2001:db8::1", 8080) == "http://[2001:db8::1]:8080"
    assert _miner_base_url("[2001:db8::1]", "8080") == "http://[2001:db8::1]:8080"
    assert _miner_base_url("203.0.113.7", 8080) == "http://203.0.113.7:8080"

    def no_session(*args, **kwargs):
        raise AssertionError("a session was opened")

    monkeypatch.setattr(aiohttp, "ClientSession", no_session)
    with pytest.raises(NodeHostError):
        await _request(f"{_miner_base_url('fd00::5', 8080)}/executors")
