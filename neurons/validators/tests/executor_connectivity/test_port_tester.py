import asyncio
import threading

import pytest
from aiohttp import web

from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_tester import PortTester

TOKEN = "tok"


async def _serve(handler):
    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
async def test_probes_run_on_their_own_loop_and_never_on_the_callers():
    threads = set()

    class Recording(PortTester):
        async def _probe(self, host, port, token):
            threads.add(threading.current_thread().name)
            return await super()._probe(host, port, token)

    async def answer(_request):
        return web.Response(text=f"{TOKEN}:1")

    runner, port = await _serve(answer)
    try:
        tester = Recording()
        ok, failed = await tester.test_many("127.0.0.1", [PortPair(1, port), PortPair(2, port)], TOKEN)
        assert (ok, failed) == ([PortPair(1, port)], [PortPair(2, port)])
        assert await tester.test_one("127.0.0.1", PortPair(1, port), TOKEN)
    finally:
        await runner.cleanup()
    assert threads == {"port-probes"}


@pytest.mark.asyncio
async def test_open_probes_stay_within_the_bound_and_queued_ones_still_pass():
    open_now = peak = 0

    async def slow_answer(_request):
        nonlocal open_now, peak
        open_now += 1
        peak = max(peak, open_now)
        await asyncio.sleep(0.2)
        open_now -= 1
        return web.Response(text=f"{TOKEN}:1")

    runner, port = await _serve(slow_answer)
    try:
        ok, failed = await PortTester(concurrency=4).test_many("127.0.0.1", [PortPair(1, port)] * 20, TOKEN)
    finally:
        await runner.cleanup()
    assert peak <= 4
    assert len(ok) == 20 and not failed  # 5 rounds of 0.2 s: a queued probe's budget starts at its slot

