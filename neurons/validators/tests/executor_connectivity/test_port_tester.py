import asyncio
import os
import sys
import threading
import time

import pytest
from aiohttp import web

from services.executor_connectivity import port_tester
from services.executor_connectivity.models import PortPair
from services.executor_connectivity.port_tester import PortTester, ProbeEngine

TOKEN = "tok"

# the probe process gets the stand-ins conftest gives the tests' own process: loopback nodes count as public, and the
# shared config is not fetched from the live API
_STAND_INS = """
from unittest.mock import patch
from lium_core.shared_config import DEFAULT_SHARED_CONFIG
patch("lium_core.shared_config.client.SharedConfigClient._fetch", return_value=DEFAULT_SHARED_CONFIG).start()
import services.node_host
services.node_host._check_literal = lambda host, port: None
"""
_TEST_BOOT = port_tester._CHILD_BOOT.replace("importlib.import_module", _STAND_INS + "importlib.import_module", 1)


class LoopbackTester(PortTester):
    def _child_argv(self):
        argv = super()._child_argv()
        argv[2] = _TEST_BOOT
        return argv


async def _serve(handler):
    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def _answer(_request):
    return web.Response(text=f"{TOKEN}:1")


@pytest.mark.asyncio
async def test_probes_run_in_a_process_of_their_own():
    runner, port = await _serve(_answer)
    tester = LoopbackTester()
    try:
        ok, failed = await tester.test_many("127.0.0.1", [PortPair(1, port), PortPair(2, port)], TOKEN)
        assert (ok, failed) == ([PortPair(1, port)], [PortPair(2, port)])
        assert await tester.test_one("127.0.0.1", PortPair(1, port), TOKEN)
        assert tester._proc.pid != os.getpid()
    finally:
        await runner.cleanup()


# a node in a process of its own, so a busy validator process cannot slow the node's answers, only the probes
_NODE = """
import asyncio
async def answer(reader, writer):
    await reader.readuntil(b"\\r\\n\\r\\n")
    writer.write(b"HTTP/1.1 200 OK\\r\\nContent-Length: 5\\r\\nConnection: close\\r\\n\\r\\ntok:1")
    await writer.drain()
    writer.close()
async def main():
    server = await asyncio.start_server(answer, "127.0.0.1", 0, backlog=4096)
    print(server.sockets[0].getsockname()[1], flush=True)
    await server.serve_forever()
asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_a_busy_validator_process_never_starves_the_probes():
    # review: with the probes on a thread of the validator, a busy validator held the interpreter lock and 2000
    # loopback probes gave 138 ok in 11.35 s
    node = await asyncio.create_subprocess_exec(sys.executable, "-c", _NODE, stdout=asyncio.subprocess.PIPE)
    port = int(await node.stdout.readline())
    stop = threading.Event()

    def busy():
        while not stop.is_set():
            sum(i * i for i in range(10_000))

    tester = LoopbackTester()
    await tester.test_one("127.0.0.1", PortPair(1, port), TOKEN)  # the process is up before the clock starts
    spinner = threading.Thread(target=busy, daemon=True)
    spinner.start()
    try:
        started = time.monotonic()
        ok, failed = await tester.test_many("127.0.0.1", [PortPair(1, port)] * 1000, TOKEN)
        assert len(ok) == 1000 and not failed
        assert time.monotonic() - started < 3  # inside one probe budget
    finally:
        stop.set()
        node.kill()
        await node.wait()


class CrashingOnce(LoopbackTester):
    """Its first probe process reads one request and dies without answering."""

    starts = 0

    def _child_argv(self):
        self.starts += 1
        if self.starts == 1:
            return [sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(1)"]
        return super()._child_argv()


@pytest.mark.asyncio
async def test_a_crashed_probe_process_fails_its_wave_clean_and_the_next_wave_gets_a_new_one():
    runner, port = await _serve(_answer)
    tester = CrashingOnce()
    try:
        ports = [PortPair(1, port)] * 3
        assert await tester.test_many("127.0.0.1", ports, TOKEN) == ([], ports)
        assert await tester.test_many("127.0.0.1", ports, TOKEN) == (ports, [])
        assert tester.starts == 2
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_probe_process_answering_garbage_fails_its_wave_clean():
    class Garbage(PortTester):
        def _child_argv(self):
            return [sys.executable, "-c", "import sys; sys.stdin.readline(); print('nope', flush=True); input()"]

    tester = Garbage()
    assert not await tester.test_one("127.0.0.1", PortPair(1, 1), TOKEN)
    assert tester._proc is None


@pytest.mark.asyncio
async def test_a_cancelled_wave_stops_its_probes_in_the_probe_process():
    open_connections = set()

    async def hold(reader, writer):
        # accepts and never answers, like dropped packets: a probe holds its socket for its 3 s read budget
        open_connections.add(writer)
        await reader.read()
        open_connections.discard(writer)

    silent = await asyncio.start_server(hold, "127.0.0.1", 0)
    silent_port = silent.sockets[0].getsockname()[1]
    tester = LoopbackTester()
    try:
        wave = asyncio.create_task(tester.test_many("127.0.0.1", [PortPair(1, silent_port)] * 20, TOKEN))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if len(open_connections) == 20:
                break
        assert len(open_connections) == 20
        wave.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wave
        for _ in range(20):
            await asyncio.sleep(0.05)
            if not open_connections:
                break
        assert not open_connections  # well before the 3 s budget: the probes were cancelled, not timed out
        assert not tester._waves
    finally:
        silent.close()


@pytest.mark.asyncio
async def test_the_probe_session_keeps_no_cookie_a_node_sends():
    seen = []

    async def answer(request):
        seen.append(request.headers.get("Cookie"))
        response = web.Response(text=f"{TOKEN}:1")
        response.set_cookie("grow", "x" * 100)
        return response

    runner, port = await _serve(answer)
    try:
        engine = ProbeEngine()
        assert await engine.probe("127.0.0.1", PortPair(1, port), TOKEN)
        assert await engine.probe("127.0.0.1", PortPair(1, port), TOKEN)
        # a default jar drops cookies from IP hosts only; a hostname node's cookies would be kept
        assert seen == [None, None]
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_node_whose_probes_hang_never_delays_another_nodes_probes_or_its_own(monkeypatch):
    # the total bound is a quarter of the FD limit; pinned here so the test does not depend on the host's limit
    monkeypatch.setattr(port_tester, "MAX_PROBE_SOCKETS", 400)

    # accepts and never answers: each probe holds its socket for the whole read budget, like dropped packets
    silent = await asyncio.start_server(lambda _r, _w: None, "127.0.0.1", 0)
    silent_port = silent.sockets[0].getsockname()[1]
    runner, port = await _serve(_answer)
    engine = ProbeEngine()
    try:
        # a large node (review: 297 ports) whose every probe hangs
        loop = asyncio.get_running_loop()
        hanging_started = loop.time()
        hanging = asyncio.create_task(engine.probe_many("127.0.0.1", [PortPair(1, silent_port)] * 300, TOKEN))
        await asyncio.sleep(0.2)
        started = loop.time()
        ok, _ = await engine.probe_many("127.0.0.1", [PortPair(1, port)] * 50, TOKEN)
        elapsed = loop.time() - started
        assert all(ok) and len(ok) == 50
        assert elapsed < 1.5  # the silent node's probes run out at 3 s; nothing here waits for them
        hung, _ = await hanging
        assert not any(hung)
        # its own probes run together, not in queued rounds: about one read budget, far inside a batch deadline
        assert loop.time() - hanging_started < 6
    finally:
        silent.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_wave_over_many_nodes_never_holds_more_sockets_than_the_bound():
    # DAH-6812: the unbounded wave of 27 Nov opened every node's probes at once and hit the FD limit (Errno 24)
    held = {"total": 0, "peak": 0}

    class Counting(ProbeEngine):
        async def _probe_unbounded(self, host, port, token):
            held["total"] += 1
            held["peak"] = max(held["peak"], held["total"])
            await asyncio.sleep(0.02)
            held["total"] -= 1
            return True

    engine = Counting(max_sockets=6)
    hosts = [f"node{i}" for i in range(5)]
    results = await asyncio.gather(*(engine.probe_many(h, [PortPair(p, p) for p in range(20)], TOKEN) for h in hosts))

    assert all(all(ok) and len(ok) == 20 for ok, _ in results)
    assert held["peak"] == 6


def test_a_reply_the_validator_has_not_read_yet_never_stalls_the_probe_process():
    # a reply bigger than the pipe buffer (~64 KB) that the busy validator has not read yet: the probe process must
    # keep taking requests instead of freezing in a blocking write (review)
    import json

    class Engine:
        def __init__(self):
            self.calls = threading.Semaphore(0)

        async def probe_many(self, host, ports, token):
            self.calls.release()
            return [True] * 20000, [0.0] * 20000

    engine = Engine()
    req_r, req_w = os.pipe()
    rep_r, rep_w = os.pipe()
    requests, replies = os.fdopen(req_r, "rb", buffering=0), os.fdopen(rep_w, "wb", buffering=0)
    child = threading.Thread(target=asyncio.run, args=(port_tester._serve(engine, requests, replies),), daemon=True)
    child.start()

    def send(wave_id):
        os.write(req_w, json.dumps({"id": wave_id, "host": "h", "ports": [[1, 1]], "token": TOKEN}).encode() + b"\n")

    try:
        send(1)
        assert engine.calls.acquire(timeout=5)
        send(2)  # the first reply is still unread
        assert engine.calls.acquire(timeout=5)
    finally:
        os.close(req_w)
        os.close(rep_r)
        child.join(5)
