import asyncio

import aiohttp
import pytest
from aiohttp import web

from services.executor_connectivity import port_tester as port_tester_module
from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_tester import PortTester


@pytest.mark.asyncio
async def test_test_many_never_runs_more_probes_than_the_process_cap(mocker):
    mocker.patch.object(port_tester_module, "_probe_slots", asyncio.Semaphore(2))
    in_flight = peak = 0

    async def probe(_session, _host, _port, _token):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return True

    tester = PortTester()
    mocker.patch.object(tester, "_probe", side_effect=probe)
    ports = [PortPair(9000 + i, 9000 + i) for i in range(6)]

    results = await asyncio.gather(
        tester.test_many(mocker.Mock(), "h", ports[:3], "t"), tester.test_many(mocker.Mock(), "h", ports[3:], "t")
    )

    assert peak == 2
    assert results == [(ports[:3], []), (ports[3:], [])]


@pytest.mark.asyncio
async def test_a_probe_queued_for_a_slot_longer_than_its_timeout_still_passes(mocker):
    mocker.patch.object(port_tester_module, "PROBE_TIMEOUT", aiohttp.ClientTimeout(sock_connect=0.2, sock_read=0.2))
    slots = asyncio.Semaphore(1)
    mocker.patch.object(port_tester_module, "_probe_slots", slots)
    app = web.Application()
    app.router.add_get("/", lambda _request: web.Response(text="tok:9000"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    await slots.acquire()  # another probe holds the only slot for longer than the 0.2 s timeout
    asyncio.get_running_loop().call_later(0.5, slots.release)

    try:
        async with aiohttp.ClientSession() as session:
            ok = await PortTester().test_one(session, "127.0.0.1", PortPair(9000, port), "tok")
    finally:
        await runner.cleanup()

    assert ok
