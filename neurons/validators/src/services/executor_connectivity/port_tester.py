import asyncio
import json
import logging
import resource
import statistics
import sys
import time
from typing import BinaryIO

import aiohttp

from core.utils import _m, get_extra_info
from services.executor_connectivity.models import PortPair
from services.node_host import public_host_connector

logger = logging.getLogger(__name__)

# per-operation budget instead of a cumulative one: aiohttp's `total` clock starts before the
# connector hands out a socket, so a probe queued behind a full pool burned its budget waiting
# and was recorded as a dead port without ever putting a packet on the wire
PROBE_TIMEOUT = aiohttp.ClientTimeout(sock_connect=3, sock_read=3)
# a node's answer is "<token>:<port>"; reading more than this lets a node grow the validator's memory
PROBE_BODY_LIMIT = 256
# Sockets the probes may hold at once. A wave probes every node's ports together, and with no bound it opened them
# all at once and ran out of file descriptors (Errno 24, 27 Nov). The bound is a quarter of the soft limit. There is no
# per-node bound: it queued a large node's own probes past its batch deadline (review).
_FD_SOFT_LIMIT = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
MAX_PROBE_SOCKETS = 2048 if _FD_SOFT_LIMIT == resource.RLIM_INFINITY else _FD_SOFT_LIMIT // 4

# Runs in the probe process: the protocol goes over the original stdout, and fd 1 then points at stderr, so a stray
# print or log line in an imported module can never corrupt a reply.
_CHILD_BOOT = """
import importlib, json, os, sys
sys.path[:] = json.loads(sys.argv[1])
replies = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
importlib.import_module(sys.argv[2]).child_main(int(sys.argv[3]), sys.stdin.buffer, replies)
"""
# one reply carries a node's whole wave (a result and a duration per port)
_PIPE_LIMIT = 2**24


class ProbeChildGone(Exception):
    """The probe process exited (crash, OOM kill) before it answered a wave."""


class ProbeEngine:
    """The probes themselves, on whatever loop calls them: the probe process's loop in production (see PortTester)."""

    def __init__(self, max_sockets: int | None = None):
        self._max_sockets = max_sockets
        self._session: aiohttp.ClientSession | None = None
        # a probe waits here before it opens its socket, and the probe budget (per connect and per read) starts only
        # after that, so waiting never fails a port
        self._sockets: asyncio.Semaphore | None = None

    async def probe(self, host: str, port: PortPair, token: str) -> bool:
        if self._sockets is None:
            self._sockets = asyncio.Semaphore(self._max_sockets or MAX_PROBE_SOCKETS)
        async with self._sockets:
            return await self._probe_unbounded(host, port, token)

    async def probe_many(self, host: str, ports: list[PortPair], token: str) -> tuple[list[bool], list[float]]:
        durations: list[float] = []

        async def timed(port: PortPair) -> bool:
            started = time.monotonic()
            try:
                return await self.probe(host, port, token)
            finally:
                durations.append(time.monotonic() - started)

        return list(await asyncio.gather(*[timed(p) for p in ports])), durations

    async def _probe_unbounded(self, host: str, port: PortPair, token: str) -> bool:
        if self._session is None:
            # the bound is the semaphore in `probe`, taken before the probe budget starts; a connector limit would
            # start the budget while the probe waits for a socket.
            # No cookie jar: a node's Set-Cookie would be kept for the life of the process
            self._session = aiohttp.ClientSession(
                connector=public_host_connector(limit=0, force_close=True), cookie_jar=aiohttp.DummyCookieJar()
            )
        url = f"http://{host}:{port.external}/"
        expected = f"{token}:{port.internal}"

        try:
            # the node answers its own address only: a redirect would point the probe somewhere else. The answer is
            # asked for and read uncompressed, so the cap counts the bytes the node sent
            async with self._session.get(
                url,
                timeout=PROBE_TIMEOUT,
                allow_redirects=False,
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
            ) as resp:
                body = b""
                while len(body) <= PROBE_BODY_LIMIT:
                    chunk = await resp.content.read(PROBE_BODY_LIMIT + 1 - len(body))
                    if not chunk:
                        break
                    body += chunk
                if len(body) > PROBE_BODY_LIMIT:
                    return False
                return body.decode(errors="replace").strip() == expected
        except Exception:
            return False


def child_main(max_sockets: int, requests: BinaryIO, replies: BinaryIO) -> None:
    """The probe process: one JSON line per request on ``requests``, one JSON line per answered wave on ``replies``.

    Request: {"id", "host", "ports": [[internal, external], ...], "token"} or {"cancel": id}.
    Reply: {"id", "ok": [bool per port], "durations": [seconds]}. Ends when the validator closes ``requests``.
    """
    asyncio.run(_serve(ProbeEngine(max_sockets), requests, replies))


async def _serve(engine: ProbeEngine, requests: BinaryIO, replies: BinaryIO) -> None:
    reader = asyncio.StreamReader(limit=_PIPE_LIMIT)
    await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), requests)
    waves: dict[int, asyncio.Task] = {}

    async def run(request: dict) -> None:
        try:
            ports = [PortPair(*p) for p in request["ports"]]
            ok, durations = await engine.probe_many(request["host"], ports, request["token"])
            # shortcut: a blocking write; a reply is a few KB and the validator reads every line as it comes
            replies.write(json.dumps({"id": request["id"], "ok": ok, "durations": durations}).encode() + b"\n")
        finally:
            waves.pop(request["id"], None)

    while line := await reader.readline():
        request = json.loads(line)
        if "cancel" in request:
            if task := waves.pop(request["cancel"], None):
                task.cancel()
        else:
            waves[request["id"]] = asyncio.create_task(run(request))


class PortTester:
    """Tests port connectivity via HTTP, in a probe process of its own.

    Probes never run in the validator's process: on its loop a wave of probe callbacks delayed every other reply
    (Redis replies sat unread), and on a thread of its own the probes waited for the interpreter lock whenever the
    validator was busy, so they ran past their 3 s budgets and live ports counted as dead (review: 138 of 2000 ok).
    The process has its own interpreter and its own file descriptors; the validator only writes a request line and
    reads the answer.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._waves: dict[int, tuple[asyncio.subprocess.Process, asyncio.Future]] = {}
        self._next_id = 0

    def _child_argv(self) -> list[str]:
        return [sys.executable, "-c", _CHILD_BOOT, json.dumps(sys.path), __name__, str(MAX_PROBE_SOCKETS)]

    async def _child(self) -> asyncio.subprocess.Process:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            # the process's pipes belong to the loop that started it
            self._loop, self._lock, self._proc = loop, asyncio.Lock(), None
        async with self._lock:
            if self._proc is None:
                proc = await asyncio.create_subprocess_exec(
                    *self._child_argv(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=_PIPE_LIMIT
                )
                self._proc = proc
                loop.create_task(self._read_replies(proc))
            return self._proc

    async def _read_replies(self, proc: asyncio.subprocess.Process) -> None:
        try:
            while line := await proc.stdout.readline():
                reply = json.loads(line)
                _, future = self._waves.pop(reply["id"], (None, None))
                if future is not None and not future.done():
                    future.set_result((reply["ok"], reply["durations"]))
        except Exception as exc:
            logger.warning(_m(f"port probe process answered garbage: {str(exc)[:100]}", extra=get_extra_info({})))
        finally:
            # the waves it still owed fail clean (every port in them unanswered); the next wave starts a new process
            if self._proc is proc:
                self._proc = None
            for wave_id, (owner, future) in list(self._waves.items()):
                if owner is proc:
                    del self._waves[wave_id]
                    if not future.done():
                        future.set_exception(ProbeChildGone())
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

    async def _wave(self, host: str, ports: list[PortPair], token: str) -> tuple[list[bool], list[float]]:
        proc = await self._child()
        self._next_id += 1
        wave_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._waves[wave_id] = (proc, future)
        request = {"id": wave_id, "host": host, "ports": [[p.internal, p.external] for p in ports], "token": token}
        try:
            try:
                proc.stdin.write(json.dumps(request).encode() + b"\n")
            except ConnectionError as exc:
                raise ProbeChildGone() from exc
            return await future
        finally:
            # still listed: the caller was cancelled (its deadline), so the process stops this wave's probes too
            if self._waves.pop(wave_id, None) is not None and not proc.stdin.is_closing():
                proc.stdin.write(json.dumps({"cancel": wave_id}).encode() + b"\n")

    async def test_one(self, host: str, port: PortPair, token: str) -> bool:
        """Test single port connectivity via HTTP."""
        try:
            results, _ = await self._wave(host, [port], token)
        except ProbeChildGone:
            return False
        return results[0]

    async def test_many(
        self,
        host: str,
        ports: list[PortPair],
        token: str,
        log_ctx: dict | None = None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Test multiple ports concurrently."""
        wall_started = time.monotonic()
        try:
            results, durations = await self._wave(host, ports, token)
        except ProbeChildGone:
            logger.warning(_m("port probe process exited mid-wave", extra=get_extra_info(log_ctx or {})))
            results, durations = [False] * len(ports), [0.0]
        wall = time.monotonic() - wall_started

        successful = [p for p, ok in zip(ports, results) if ok]
        failed = [p for p, ok in zip(ports, results) if not ok]

        sorted_durations = sorted(durations)
        p95 = sorted_durations[min(int(len(sorted_durations) * 0.95), len(sorted_durations) - 1)]
        logger.info(
            _m(
                f"probed {len(ports)} ports in {wall:.2f}s: {len(successful)} ok, "
                f"per-probe med={statistics.median(sorted_durations):.2f}s p95={p95:.2f}s max={sorted_durations[-1]:.2f}s",
                extra=get_extra_info(log_ctx or {}),
            )
        )
        return successful, failed
