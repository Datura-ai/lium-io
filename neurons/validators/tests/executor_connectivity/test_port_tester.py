import asyncio
import threading

import pytest
from aiohttp import web

from services.executor_connectivity.models import PortPair
from services.executor_connectivity import port_tester
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
async def test_a_node_whose_probes_hang_never_delays_another_nodes_probes():
    async def answer(_request):
        return web.Response(text=f"{TOKEN}:1")

    # accepts and never answers: each probe holds its socket for the whole read budget, like dropped packets
    silent = await asyncio.start_server(lambda _r, _w: None, "127.0.0.1", 0)
    silent_port = silent.sockets[0].getsockname()[1]
    runner, port = await _serve(answer)
    tester = PortTester()
    try:
        hanging_ports = [PortPair(1, silent_port)] * port_tester.MAX_PROBE_SOCKETS_PER_NODE
        hanging = asyncio.create_task(tester.test_many("127.0.0.1", hanging_ports, TOKEN))
        await asyncio.sleep(0.2)
        started = asyncio.get_running_loop().time()
        ok, failed = await tester.test_many("127.0.0.1", [PortPair(1, port)] * 50, TOKEN)
        elapsed = asyncio.get_running_loop().time() - started
        assert len(ok) == 50 and not failed
        assert elapsed < 1.5  # the silent node's probes run out at 3 s; nothing here waits for them
        assert await hanging == ([], hanging_ports)
    finally:
        silent.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_wave_over_many_nodes_never_holds_more_sockets_than_the_bounds(monkeypatch):
    # DAH-6812: the unbounded wave of 27 Nov opened every node's probes at once and hit the FD limit (Errno 24)
    monkeypatch.setattr(port_tester, "MAX_PROBE_SOCKETS", 6)
    monkeypatch.setattr(port_tester, "MAX_PROBE_SOCKETS_PER_NODE", 4)
    held = {"total": 0, "peak": 0}
    per_node: dict[str, int] = {}
    per_node_peak: dict[str, int] = {}

    class Counting(PortTester):
        async def _probe_unbounded(self, host, port, token):
            held["total"] += 1
            per_node[host] = per_node.get(host, 0) + 1
            held["peak"] = max(held["peak"], held["total"])
            per_node_peak[host] = max(per_node_peak.get(host, 0), per_node[host])
            await asyncio.sleep(0.02)
            held["total"] -= 1
            per_node[host] -= 1
            return True

    tester = Counting()
    hosts = [f"node{i}" for i in range(5)]
    results = await asyncio.gather(*(tester.test_many(h, [PortPair(p, p) for p in range(20)], TOKEN) for h in hosts))

    assert all(len(ok) == 20 and not failed for ok, failed in results)
    assert held["peak"] == 6
    assert max(per_node_peak.values()) <= 4
