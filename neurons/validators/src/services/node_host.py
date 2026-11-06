"""A node registered by hostname (dynamic DNS) is resolved before the validator dials it.

The hostname stays the node's address everywhere (its identity does not change when the IP behind
it does); this check only refuses a name that points anywhere but the public internet.
"""

import asyncio
import contextlib
import contextvars
import functools
import ipaddress
import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)


class NodeHostError(OSError):
    """The node's hostname does not resolve to public addresses only.

    An OSError so the connect site classifies it like any other failed dial: the node is reported
    EXECUTOR_SSH_UNREACHABLE (an availability error) and hidden, not scored as a generic failure.
    """


def is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


class _CheckedAddresses(list):
    """``resolve_public_host``'s results for a name; keeps the name so the dial can log both."""

    host: str = ""


def _refused(host: str, port: int, reason: str, addresses: list[str] | None = None) -> NodeHostError:
    logger.warning(
        _m(
            f"Refused hostname node {host}:{port}: {reason}",
            extra=get_extra_info({"host": host, "port": port, "addresses": addresses or [], "reason": reason}),
        )
    )
    return NodeHostError(f"{host} {reason}")


def _log_dialed(infos: list, sockaddr: tuple) -> None:
    host = getattr(infos, "host", "")
    if not host:
        return
    address, port = sockaddr[0], sockaddr[1]
    logger.info(
        _m(
            f"Dialed hostname node {host}:{port} at {address}",
            extra=get_extra_info({"host": host, "port": port, "address": address}),
        )
    )


# A hostname node's DNS is the miner's: a zone that never answers holds each lookup until the system
# resolver gives up, and the thread cannot be cancelled. These lookups run in their own pool, never the
# loop's default executor, where the validator's Redis/WebSocket DNS and other blocking calls run, and
# each waits at most NODE_DNS_TIMEOUT_SECONDS.
# Inside the pool, one MINER holds at most NODE_DNS_PER_MINER threads, counted until each thread really
# finishes, so a miner whose DNS stalls delays only its own nodes. The key is the authenticated miner
# hotkey the validator is working for (node_owner() / for_payload_miner, set where per-miner work starts), never a part of
# the hostname: a miner chooses its names, so a name-derived key can be split across domains to fill
# the pool, or shared with other miners' names (co.uk, duckdns.org). Work with no miner set (none in the
# validator today) is keyed by the full name.
NODE_DNS_MAX_WORKERS = 32
NODE_DNS_PER_MINER = 4
NODE_DNS_TIMEOUT_SECONDS = 15.0
NODE_DNS_EXECUTOR = ThreadPoolExecutor(max_workers=NODE_DNS_MAX_WORKERS, thread_name_prefix="node-dns")
_NODE_OWNER: contextvars.ContextVar[str | None] = contextvars.ContextVar("node_owner", default=None)
_dns_slot_lock = threading.Lock()
_dns_busy: dict[str, int] = {}
_dns_waiters: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Future]]] = {}
# one lookup per (loop, quota key, name) at a time: a port check dials up to 300 ports of one host at once
_dns_inflight: dict[tuple, asyncio.Task] = {}


@contextlib.contextmanager
def node_owner(miner_hotkey: str | None):
    """Attribute the node lookups made inside this block (and in tasks created inside it) to a miner."""
    token = _NODE_OWNER.set(miner_hotkey or None)
    try:
        yield
    finally:
        _NODE_OWNER.reset(token)


def for_payload_miner(handler):
    """Decorates an async ``handler(self, payload, ...)`` of per-miner work: the node lookups it makes,
    and those of every task it creates, are attributed to ``payload.miner_hotkey``."""

    @functools.wraps(handler)
    async def wrapper(self, payload, *args, **kwargs):
        with node_owner(getattr(payload, "miner_hotkey", None)):
            return await handler(self, payload, *args, **kwargs)

    return wrapper


def dns_quota_key(host: str) -> str:
    """Whose share of the pool a lookup of ``host`` uses: the miner set by node_owner(), else the name."""
    owner = _NODE_OWNER.get()
    return f"miner:{owner}" if owner else f"name:{host.rstrip('.').lower()}"


async def _take_dns_slot(key: str) -> None:
    """Wait for one of the key's NODE_DNS_PER_MINER slots. A waiter sleeps until a slot is given back (by a pool
    thread, hence call_soon_threadsafe), never polls."""
    loop = asyncio.get_running_loop()
    while True:
        with _dns_slot_lock:
            if _dns_busy.get(key, 0) < NODE_DNS_PER_MINER:
                _dns_busy[key] = _dns_busy.get(key, 0) + 1
                return
            woken = loop.create_future()
            _dns_waiters.setdefault(key, []).append((loop, woken))
        try:
            await woken
        finally:
            with _dns_slot_lock:
                waiters = _dns_waiters.get(key, [])
                if (loop, woken) in waiters:
                    waiters.remove((loop, woken))
                if not waiters:
                    _dns_waiters.pop(key, None)


def _wake(woken: asyncio.Future) -> None:
    if not woken.done():
        woken.set_result(None)


def _give_dns_slot(key: str) -> None:
    with _dns_slot_lock:
        left = _dns_busy.get(key, 0) - 1
        if left > 0:
            _dns_busy[key] = left
        else:
            _dns_busy.pop(key, None)
        waiters = _dns_waiters.pop(key, [])
    for loop, woken in waiters:  # every waiter retries; those that lose wait again
        with contextlib.suppress(RuntimeError):  # its loop is closed
            loop.call_soon_threadsafe(_wake, woken)


class _QuotaSlotExecutor:
    """Runs one lookup in NODE_DNS_EXECUTOR and frees its quota slot when the thread finishes (not when
    the caller stops waiting: an abandoned lookup still occupies its thread)."""

    def __init__(self, key: str):
        self.key = key
        self.submitted = False

    def submit(self, fn, *args, **kwargs):
        future = NODE_DNS_EXECUTOR.submit(fn, *args, **kwargs)
        self.submitted = True
        future.add_done_callback(lambda _: _give_dns_slot(self.key))
        return future


async def resolve_public_host(host: str, port: int = 22, executor=None) -> list:
    """The getaddrinfo results to dial for ``host``: [] for an IP literal (dial it as is), else every
    address the name resolves to, all of them global; NodeHostError otherwise.

    ``executor``: the thread pool the blocking lookup runs in. None is the loop's default executor
    (``loop.getaddrinfo``); a caller whose lookups a miner's DNS can stall passes its own pool, so
    they cannot fill the one every other connection of the validator resolves in."""
    if is_ip_literal(host):
        return []
    loop = asyncio.get_running_loop()
    try:
        if executor is None:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        else:
            infos = await loop.run_in_executor(
                executor, functools.partial(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
            )
    except socket.gaierror as exc:
        raise _refused(host, port, f"does not resolve: {exc}") from exc
    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    if not addresses:
        raise _refused(host, port, "resolves to no address")
    blocked = sorted(str(a) for a in addresses if not a.is_global)
    if blocked:
        raise _refused(host, port, f"resolves to a non-public address: {', '.join(blocked)}", blocked)
    checked = _CheckedAddresses(infos)
    checked.host = host
    return checked


async def resolve_node_host(host: str, port: int) -> list:
    """``resolve_public_host`` in NODE_DNS_EXECUTOR, refused (NodeHostError) when the miner's DNS has
    not answered within NODE_DNS_TIMEOUT_SECONDS. A caller's own, shorter deadline still applies.
    Concurrent lookups of one name by one miner share a single lookup, whatever port each dials."""
    if is_ip_literal(host):
        return []
    loop = asyncio.get_running_loop()
    key = dns_quota_key(host)
    inflight_key = (loop, key, host.rstrip(".").lower())
    lookup = _dns_inflight.get(inflight_key)
    if lookup is None:
        lookup = loop.create_task(_resolve_in_quota(host, port, key))
        _dns_inflight[inflight_key] = lookup
        lookup.add_done_callback(functools.partial(_lookup_done, inflight_key))
    # shield: a caller that gives up must not cancel the lookup the others wait for (it ends at the deadline)
    infos = await asyncio.shield(lookup)
    return _at_port(infos, port)


def _lookup_done(inflight_key: tuple, lookup: asyncio.Task) -> None:
    if _dns_inflight.get(inflight_key) is lookup:
        del _dns_inflight[inflight_key]
    if not lookup.cancelled():
        lookup.exception()  # retrieved: every caller may have given up already


def _at_port(infos: list, port: int) -> list:
    """The shared lookup's results with this caller's port in each address."""
    checked = _CheckedAddresses(
        (family, sock_type, proto, canon, (sockaddr[0], port, *sockaddr[2:]))
        for family, sock_type, proto, canon, sockaddr in infos
    )
    checked.host = getattr(infos, "host", "")
    return checked


async def _resolve_in_quota(host: str, port: int, key: str) -> list:
    try:
        async with asyncio.timeout(NODE_DNS_TIMEOUT_SECONDS):
            await _take_dns_slot(key)  # returns once this miner has a lookup thread free
            slot = _QuotaSlotExecutor(key)
            try:
                return await resolve_public_host(host, port, executor=slot)
            finally:
                if not slot.submitted:
                    _give_dns_slot(key)
    except TimeoutError as exc:
        raise _refused(host, port, f"did not resolve within {NODE_DNS_TIMEOUT_SECONDS:g} s") from exc


async def public_host_sock(host: str, port: int = 22, connect_timeout: float | None = None) -> dict:
    """Return the extra ``asyncssh.connect`` kwargs that make it dial only a validated address.

    An IP literal needs nothing ({}: unchanged behaviour). A name is resolved once and refused
    (NodeHostError) unless every address is global; a TCP socket is then opened to one of those
    addresses and returned as ``{"sock": ...}``. asyncssh runs over that socket instead of
    resolving the name again (which a rebinding DNS server could answer with a private address),
    while still matching known_hosts against ``host``, the name.
    """
    # one budget for the lookup and the dial: a stalled DNS answer must not outlast connect_timeout
    async with asyncio.timeout(connect_timeout):
        infos = await resolve_node_host(host, port)
        if not infos:
            return {}
        return {"sock": await _dial_first(asyncio.get_running_loop(), infos)}


async def open_public_connection(
    host: str, port: int, **kwargs
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """``asyncio.open_connection`` that dials a hostname only at an address it was validated against."""
    pinned = await public_host_sock(host, port)
    if pinned:
        return await asyncio.open_connection(**pinned, **kwargs)
    return await asyncio.open_connection(host, port, **kwargs)


def dial_validated(infos: list, timeout: float | None) -> socket.socket:
    """Blocking TCP connect to the first reachable address of ``resolve_public_host``'s results (no lookup)."""
    last_error: OSError | None = None
    for family, sock_type, proto, _, sockaddr in infos:
        sock = socket.socket(family, sock_type, proto)
        try:
            sock.settimeout(timeout)
            sock.connect(sockaddr)
        except OSError as exc:
            sock.close()
            last_error = exc
            continue
        except BaseException:
            sock.close()
            raise
        _log_dialed(infos, sockaddr)
        return sock
    raise last_error


async def _dial_first(loop: asyncio.AbstractEventLoop, infos: list) -> socket.socket:
    last_error: OSError | None = None
    for family, sock_type, proto, _, sockaddr in infos:
        sock = socket.socket(family, sock_type, proto)
        try:
            sock.setblocking(False)
            await loop.sock_connect(sock, sockaddr)
        except OSError as exc:
            sock.close()
            last_error = exc
            continue
        except BaseException:
            sock.close()
            raise
        _log_dialed(infos, sockaddr)
        return sock
    raise last_error


class PublicHostResolver(AbstractResolver):
    """aiohttp resolver for requests to a node: a name resolves through ``resolve_public_host``.

    aiohttp connects to exactly the addresses returned here (no second lookup), so a name that is
    refused or that rebinds to a private address after the check is never dialled; the request's
    Host header and TLS SNI keep the name. IP literals never reach a resolver.
    """

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        infos = await resolve_node_host(host, port)
        if not infos:  # an IP literal
            infos = [(family or socket.AF_INET, socket.SOCK_STREAM, 0, "", (host.strip("[]"), port))]
        else:
            addresses = sorted({info[4][0] for info in infos if not family or info[0] == family})
            logger.info(
                _m(
                    f"Resolved hostname node {host}:{port} to {', '.join(addresses)}",
                    extra=get_extra_info({"host": host, "port": port, "addresses": addresses}),
                )
            )
        return [
            ResolveResult(
                hostname=host,
                host=sockaddr[0],
                port=sockaddr[1],
                family=info_family,
                proto=proto,
                flags=socket.AI_NUMERICHOST | socket.AI_NUMERICSERV,
            )
            for info_family, _, proto, _, sockaddr in infos
            if not family or info_family == family
        ]

    async def close(self) -> None:
        pass


def public_host_connector(**kwargs) -> aiohttp.TCPConnector:
    """A TCPConnector for HTTP requests to a node's address (a hostname is pinned, see above)."""
    return aiohttp.TCPConnector(resolver=PublicHostResolver(), **kwargs)
