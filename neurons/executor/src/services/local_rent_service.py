"""The validator's one-call rental create, run on the executor (`POST /rent`, liumd deploy).

What the SSH path does in four round trips — `docker run -d` through the Docker SDK over the SSH
tunnel, then a `docker ps` poll until the container runs, then the first exec that proves sshd is
up — this does in-process against the host's Docker socket from one signed intent:

    image      `docker image inspect`: is the image here, which digest (a fact for the validator)
    container  the validator's own `ContainerRunSpec`, created and started with docker-py exactly
               as the validator would (`datura.rental_spec.create_and_start`: ONE definition) on
               the icc-off rental network it names (`ensure_rental_network`: the same check)
    ready      poll `inspect` until State.Running; optionally wait for the SSH banner on the
               published sshd port from this host

What the executor refuses, whoever signed (`refuse_spec`): a spec with anything private — a
command, an entrypoint, an environment beyond the validator's own — because the API is plain HTTP
and because a rental container's start is the image's own; a bind mount of a host path (a rental
mounts named volumes; the one host path the validator ever binds is the quote broker socket);
a runtime, capability or device outside what a rental uses; a published port outside this
executor's rental ports or on its sshd port. The intent is bound to THIS executor by the SSH host
key the validator already knows (`ssh_host_key_sha256`), so a captured intent does not create a
container elsewhere.

Rollback never goes by name: the container the executor made is removed by its id, or by the
label it was created with (`lium.local_rent.nonce=<nonce>`) when the create was cut before the
daemon answered — so a container the validator's SSH fallback makes under the same name is never
touched. `rolled_back=True` in the answer means "nothing of ours remains, proven"; when the
executor cannot prove it (a cut create), it says False and the validator frees the name itself
before its own `docker run`. What is never here: the sshd bootstrap, authorized keys, environment
injection, rm — those stay the validator's SSH execs (LIUMD_PHASE2 NEVER-replace list).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import socket
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from datura.rental_spec import (
    NAME_PATTERN,
    RENTAL_CONTAINER_NAME_PREFIXES,
    RENTAL_NETWORK_NAME,
    ContainerRunSpec,
    WireError,
    carries_only_public_fields,
    create_and_start,
    ensure_rental_network,
    spec_from_wire,
)
from payloads.rent import STEP_NAMES, ReadyStep, RentIntentBody, RentResult, RentStepResult
from services.local_verify_service import parse_port_range

logger = logging.getLogger(__name__)

# docker-py is synchronous; its calls run here, never on the event loop. Two workers: a create in
# flight and its rollback, nothing wider — the route runs one intent at a time.
_rent_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="local-rent")

INSPECT_TIMEOUT_SECONDS = 10
CREATE_TIMEOUT_SECONDS = 60
# The rollback's own bound; the validator's call budget leaves room for it after the deadline
# (LOCAL_RENT_TIMEOUT_SECONDS − deadline_s ≥ this + a margin).
REMOVE_TIMEOUT_SECONDS = 10
# A create cut by the deadline may still be in the daemon's hands when the first rollback runs;
# the second, this much later, takes what the first was too early for — by label, never by name.
ROLLBACK_RETRY_SECONDS = 5
RUNNING_POLL_INTERVAL_SECONDS = 0.2
SSH_CONNECT_ATTEMPT_SECONDS = 1.0
SSH_BANNER_READ_SECONDS = 2.0
SSH_POLL_INTERVAL_SECONDS = 0.25
# The label the executor puts on every container it creates here: what its rollback goes by.
NONCE_LABEL = "lium.local_rent.nonce"
# Where a TCP connect to the host's published ports goes from inside the executor container: the
# default gateway of its bridge (the host). On a host-network executor the loopback is the host.
_HOST_GATEWAY_ROUTE = "/proc/net/route"

# What a rental container uses (docker_service.build_run_spec, nvidia_devices, cvm_quote_broker):
# a signed intent asking for more is refused, whoever signed it.
# What `_build_rental_container_run_spec` emits: the daemon default, or sysbox for a DinD template.
ALLOWED_RUNTIMES = (None, "sysbox-runc")
ALLOWED_CAPABILITIES = frozenset({"NET_ADMIN", "IPC_LOCK"})
# The one sysctl a rental sets (WireGuard's mark routing) and the one ulimit (memlock for RDMA).
ALLOWED_SYSCTLS = {"net.ipv4.conf.all.src_valid_mark": "1"}
ALLOWED_ULIMIT_NAMES = frozenset({"memlock"})
ALLOWED_DEVICE_PREFIXES = ("/dev/nvidia", "/dev/infiniband/", "/dev/net/tun", "/dev/fuse", "/dev/dri/")
ALLOWED_HOST_BIND_PREFIXES = ("/var/run/lium-dstack/",)
# The daemon's default bridge (the CVM quote broker) or the icc-off rental bridge (DAH-3199) —
# never `host`, `none` or `container:<id>`, which would put the pod in the host's or another
# container's network namespace.
ALLOWED_NETWORKS = (None, RENTAL_NETWORK_NAME)


def _docker_api():
    import docker  # local import: the module is mocked in unit tests

    return docker.APIClient(base_url="unix:///var/run/docker.sock", timeout=CREATE_TIMEOUT_SECONDS)


def host_gateway_ip(route_table: str | None = None) -> str | None:
    """The default gateway of this container's network namespace (Linux `/proc/net/route`), which
    on a bridge is the host itself; None when not readable."""
    try:
        text = route_table if route_table is not None else open(_HOST_GATEWAY_ROUTE).read()
    except OSError:
        return None
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[1] != "00000000":
            continue
        try:
            packed = bytes.fromhex(fields[2])
        except ValueError:
            continue
        if len(packed) != 4:
            continue
        return socket.inet_ntoa(packed[::-1])
    return None


def host_key_sha256(host_key_line: str | None) -> str | None:
    """The digest both ends compute of the executor's SSH host public key line (the one the miner
    reports and the validator pins for its SSH connections)."""
    if not host_key_line or not host_key_line.strip():
        return None
    return hashlib.sha256(host_key_line.strip().encode("utf-8")).hexdigest()


def refuse_spec(spec: ContainerRunSpec) -> str | None:
    """None when the spec is a rental container's; otherwise why it is refused (the validator's
    log label). Enforced HERE, not only by the sender: the validator hotkey alone must not be able
    to run anything but a rental's image with a rental's mounts on this host."""
    if not carries_only_public_fields(spec):
        return "spec carries a command, an entrypoint or a private environment"
    if not spec.name.startswith(RENTAL_CONTAINER_NAME_PREFIXES):
        return f"container name {spec.name!r} is not a rental's (pod_/filler_)"
    for volume in spec.volumes:
        if volume.source.startswith("/"):
            if not volume.source.startswith(ALLOWED_HOST_BIND_PREFIXES) or ".." in volume.source:
                return f"volume source {volume.source!r} is a host path outside a rental's"
        elif not NAME_PATTERN.fullmatch(volume.source):
            return f"volume source {volume.source!r} is not a docker volume name"
    if spec.runtime not in ALLOWED_RUNTIMES:
        return f"runtime {spec.runtime!r} is not a rental's"
    if not set(spec.cap_add) <= ALLOWED_CAPABILITIES:
        return f"capabilities {sorted(set(spec.cap_add) - ALLOWED_CAPABILITIES)} are not a rental's"
    if any(ALLOWED_SYSCTLS.get(k) != v for k, v in spec.sysctls.items()):
        return f"sysctls {sorted(k for k, v in spec.sysctls.items() if ALLOWED_SYSCTLS.get(k) != v)} are not a rental's"
    if not {u.name for u in spec.ulimits} <= ALLOWED_ULIMIT_NAMES:
        return f"ulimits {sorted({u.name for u in spec.ulimits} - ALLOWED_ULIMIT_NAMES)} are not a rental's"
    for device in spec.devices:
        if not device.path_on_host.startswith(ALLOWED_DEVICE_PREFIXES) or ".." in device.path_on_host:
            return f"device {device.path_on_host!r} is not a rental's"
    if spec.network not in ALLOWED_NETWORKS:
        return f"network {spec.network!r} is not a rental's"
    return None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def refuse_ready(spec: ContainerRunSpec, ready: ReadyStep | None) -> str | None:
    """The sshd banner probe may only dial a host port THIS spec publishes: the intent must not
    turn the executor into a probe of the host's other ports."""
    if ready is None or ready.ssh_host_port is None:
        return None
    if ready.ssh_host_port not in {b.host_port for b in spec.ports if b.protocol == "tcp"}:
        return f"ready.ssh_host_port {ready.ssh_host_port} is not a host port this spec publishes"
    return None


class LocalRentService:
    """Runs one rent intent. One at a time per executor: two rentals never land on the same
    executor at once by design (the backend serialises them), and one create at a time keeps the
    rollback story simple."""

    def __init__(
        self,
        *,
        executor_version: str,
        max_deadline_s: int,
        port_range: str | None = None,
        port_mappings: str | None = None,
        ssh_port: int = 22,
        host_key: Callable[[], str | None] = lambda: None,
        docker_api: Callable[[], Any] | None = None,
        gateway_ip: Callable[[], str | None] = host_gateway_ip,
    ) -> None:
        self.executor_version = executor_version
        self.max_deadline_s = max_deadline_s
        self.port_range = port_range
        self.port_mappings = port_mappings
        self.ssh_port = ssh_port
        self._host_key = host_key
        # Resolved per call, so a test's stand-in for the module's `_docker_api` is honoured.
        self._docker_api = docker_api or (lambda: _docker_api())
        self._gateway_ip = gateway_ip
        self._busy = asyncio.Lock()
        # The second label pass of a cut create runs off its request; held here so the loop's weak
        # reference is not the only one.
        self._late_rollbacks: set[asyncio.Task] = set()

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def hold_late_rollback(self, task: asyncio.Task) -> None:
        self._late_rollbacks.add(task)
        task.add_done_callback(self._late_rollbacks.discard)

    def refuse_intent(self, body: RentIntentBody) -> str | None:
        """Why this intent is not for this executor (401 at the route), else None: the host-key
        digest the validator signed must be this executor's own. No key here → not for us."""
        own = host_key_sha256(self._host_key())
        if own is None:
            return "this executor has no SSH host key to be bound to"
        if body.ssh_host_key_sha256 != own:
            return "intent is bound to another executor's SSH host key"
        return None

    async def run(self, body: RentIntentBody) -> RentResult:
        if self._busy.locked():
            raise BusyError("a rental create is already running")
        async with self._busy:
            return await self._run(body)

    # --- the run -------------------------------------------------------------------------------

    async def _run(self, body: RentIntentBody) -> RentResult:
        started_wall = int(time.time())
        started = time.perf_counter()
        deadline = min(body.deadline_s, self.max_deadline_s)
        results: dict[str, RentStepResult] = {}
        try:
            # docker-py's client constructor asks the daemon for its API version: off the loop too.
            api = await self._open_api()
        except Exception as exc:  # noqa: BLE001 — no client, nothing created: the validator falls back
            logger.error("local rent: docker client unavailable: %r", exc)
            results["container"] = RentStepResult(status="failed", error=f"docker client: {type(exc).__name__}: {exc}")
            return self._result(body, started_wall, started, deadline, results, deadline_hit=False, rolled_back=True)
        run = _Run(self, api, body, results)

        task = asyncio.ensure_future(run.steps())
        try:
            done, pending = await asyncio.wait({task}, timeout=deadline)
        except asyncio.CancelledError:
            # The request itself was cancelled (client gone): nothing of ours may stay behind.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.shield(run.rollback())
            _close(api)
            raise
        deadline_hit = bool(pending)
        if pending:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        elif task.exception() is not None:
            logger.error("local rent failed: %r", task.exception())
            results.setdefault(
                "container",
                RentStepResult(status="failed", error=f"{type(task.exception()).__name__}: {task.exception()}"),
            )

        # "Nothing of ours remains": trivially so when no create was ever issued (image absent,
        # spec or port refused, deadline before the create); otherwise the rollback must prove it.
        rolled_back = not run.create_attempted
        if run.create_attempted and not self._succeeded(body, results, deadline_hit):
            rolled_back = await asyncio.shield(run.rollback())
        _close(api)
        return self._result(body, started_wall, started, deadline, results, deadline_hit=deadline_hit, rolled_back=rolled_back)

    async def _open_api(self) -> Any:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(loop.run_in_executor(_rent_executor, self._docker_api), timeout=INSPECT_TIMEOUT_SECONDS)

    def _result(
        self,
        body: RentIntentBody,
        started_wall: int,
        started: float,
        deadline: int,
        results: dict[str, RentStepResult],
        *,
        deadline_hit: bool,
        rolled_back: bool,
    ) -> RentResult:
        for name in STEP_NAMES:
            if name not in results:
                wanted = (
                    (name == "image" and body.steps.image)
                    or (name == "container" and body.steps.container is not None)
                    or (name == "ready" and body.steps.ready is not None)
                )
                if wanted and deadline_hit:
                    results[name] = RentStepResult(
                        status="timeout", error=f"not finished within the {deadline}s deadline"
                    )
                else:
                    results[name] = RentStepResult(status="skipped")

        return RentResult(
            nonce=body.nonce,
            executor_uuid=body.executor_uuid,
            executor_version=self.executor_version,
            started_at=started_wall,
            elapsed_ms=_elapsed_ms(started),
            deadline_hit=deadline_hit,
            rolled_back=rolled_back,
            steps=results,
        )

    @staticmethod
    def _succeeded(body: RentIntentBody, results: dict[str, RentStepResult], deadline_hit: bool) -> bool:
        if deadline_hit:
            return False
        container = results.get("container")
        if container is None or container.status != "ok":
            return False
        if body.steps.ready is not None:
            ready = results.get("ready")
            return ready is not None and ready.status == "ok"
        return True

    def offered_ports(self) -> set[int]:
        """The host ports docker may publish on here: the INTERNAL side of this executor's port
        mappings (the validator's spec names that side — `_published_ports` binds
        `host_port=internal_port`; the external side is the address a renter dials through the
        provider's NAT), its range, else the validator's default allocation for an executor that
        configured none (the same default `parse_port_range` answers the facts with)."""
        return {internal for internal, _external in parse_port_range(self.port_range, self.port_mappings)}

    def refuse_ports(self, spec: ContainerRunSpec) -> str | None:
        """Every published host port must be one of this executor's rental ports and never its own
        sshd port — the intent chooses among what the executor already offers."""
        offered = self.offered_ports()
        for binding in spec.ports:
            if binding.host_port == self.ssh_port or binding.host_port not in offered:
                return f"host port {binding.host_port} is not one of this executor's rental ports"
        return None


class _Run:
    """One intent's steps against one docker-py client; owns what was created for the rollback."""

    def __init__(self, service: LocalRentService, api: Any, body: RentIntentBody, results: dict[str, RentStepResult]):
        self.service = service
        self.api = api
        self.body = body
        self.results = results
        self.spec: ContainerRunSpec | None = None
        self.create_attempted = False
        self.container_id: str | None = None
        # True when the daemon itself answered the create with an error (an HTTP status): then
        # nothing of ours exists. A transport error on the socket proves nothing either way.
        self.daemon_refused_create = False

    async def steps(self) -> None:
        body, results = self.body, self.results
        if body.steps.container is not None:
            try:
                self.spec = spec_from_wire(body.steps.container)
            except WireError as exc:
                results["container"] = RentStepResult(status="failed", error=f"spec: {exc}")
                return
            refused = (
                refuse_spec(self.spec)
                or self.service.refuse_ports(self.spec)
                or refuse_ready(self.spec, body.steps.ready)
            )
            if refused:
                results["container"] = RentStepResult(status="failed", error=refused)
                return
        if body.steps.image:
            if self.spec is None:
                results["image"] = RentStepResult(status="failed", error="no container spec to name the image")
                return
            results["image"] = await self._image(self.spec.image)
            if results["image"].status != "ok" or not (results["image"].data or {}).get("present"):
                return
        if self.spec is None:
            return
        results["container"] = await self._container(self.spec)
        if results["container"].status != "ok":
            return
        if body.steps.ready is not None:
            results["ready"] = await self._ready(self.spec, body.steps.ready)

    async def _in_thread(self, func: Callable[[], Any], timeout: float) -> Any:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(loop.run_in_executor(_rent_executor, func), timeout=timeout)

    async def _image(self, reference: str) -> RentStepResult:
        started = time.perf_counter()

        def inspect() -> dict[str, Any]:
            try:
                info = self.api.inspect_image(reference)
            except Exception as exc:  # noqa: BLE001 — "not found" is the fact we came for
                if _is_not_found(exc):
                    return {"present": False}
                raise
            digests = info.get("RepoDigests") or []
            return {"present": True, "id": info.get("Id"), "digest": digests[0] if digests else None}

        try:
            data = await self._in_thread(inspect, INSPECT_TIMEOUT_SECONDS)
            return RentStepResult(status="ok", ms=_elapsed_ms(started), data=data)
        except TimeoutError:
            return RentStepResult(status="timeout", ms=_elapsed_ms(started), error="image inspect timed out")
        except Exception as exc:  # noqa: BLE001 — the validator reads the error and falls back to SSH
            return RentStepResult(status="failed", ms=_elapsed_ms(started), error=f"{type(exc).__name__}: {exc}")

    async def _container(self, spec: ContainerRunSpec) -> RentStepResult:
        started = time.perf_counter()
        if spec.network:
            # What the validator's SSH path does before its own create (DAH-3199): the icc-off
            # rental network exists, or the rental is refused. Before `create_attempted`: a refusal
            # here made nothing, so the answer's `rolled_back` is True without a search.
            try:
                await self._in_thread(lambda: ensure_rental_network(self.api, spec.network), INSPECT_TIMEOUT_SECONDS)
            except TimeoutError:
                return RentStepResult(status="timeout", ms=_elapsed_ms(started), error="rental network inspect/create timed out")
            except Exception as exc:  # noqa: BLE001 — the reason is the evidence; the SSH path meets the same refusal
                return RentStepResult(status="failed", ms=_elapsed_ms(started), error=f"{type(exc).__name__}: {exc}")
        # Marked before the create is issued: a cancellation mid-flight (the deadline) must still
        # run the rollback, which finds what the daemon may have made by the nonce label.
        self.create_attempted = True
        labels = {NONCE_LABEL: self.body.nonce}

        def created(container_id: str) -> None:
            self.container_id = container_id  # known the moment the daemon answers the create

        try:
            await self._in_thread(
                lambda: create_and_start(self.api, spec, labels=labels, on_created=created), CREATE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return RentStepResult(status="timeout", ms=_elapsed_ms(started), error="create/start timed out")
        except Exception as exc:  # noqa: BLE001 — the daemon's reason is the evidence
            self.daemon_refused_create = self.container_id is None and _is_daemon_answer(exc)
            return RentStepResult(status="failed", ms=_elapsed_ms(started), error=f"{type(exc).__name__}: {exc}")
        return RentStepResult(
            status="ok",
            ms=_elapsed_ms(started),
            data={"container_name": spec.name, "container_id": self.container_id},
        )

    async def _ready(self, spec: ContainerRunSpec, step: ReadyStep) -> RentStepResult:
        started = time.perf_counter()
        running_by = time.perf_counter() + step.running_timeout_s
        target = self.container_id or spec.name
        state: dict[str, Any] = {}
        while True:
            try:
                state = await self._in_thread(
                    lambda: (self.api.inspect_container(target) or {}).get("State") or {}, INSPECT_TIMEOUT_SECONDS
                )
            except Exception as exc:  # noqa: BLE001 — gone or unreadable: the validator decides
                return RentStepResult(status="failed", ms=_elapsed_ms(started), error=f"inspect: {type(exc).__name__}: {exc}")
            if state.get("Running"):
                break
            if state.get("Status") in ("exited", "dead") or state.get("Error"):
                return RentStepResult(
                    status="failed",
                    ms=_elapsed_ms(started),
                    error=f"container is {state.get('Status')}: exit {state.get('ExitCode')} {state.get('Error') or ''}".strip(),
                    data={"state": _public_state(state)},
                )
            if time.perf_counter() >= running_by:
                return RentStepResult(
                    status="timeout",
                    ms=_elapsed_ms(started),
                    error=f"not running after {step.running_timeout_s}s",
                    data={"state": _public_state(state)},
                )
            await asyncio.sleep(RUNNING_POLL_INTERVAL_SECONDS)

        data: dict[str, Any] = {"state": _public_state(state), "running_ms": _elapsed_ms(started)}
        if step.ssh_host_port is not None:
            hosts = ["127.0.0.1"]
            gateway = self.service._gateway_ip()
            if gateway and gateway not in hosts:
                hosts.append(gateway)
            answered_on = await _wait_ssh_banner(hosts, step.ssh_host_port, step.ssh_timeout_s)
            data.update({"ssh_port": step.ssh_host_port, "ssh_answered": answered_on is not None, "ssh_probe_host": answered_on})
            if answered_on is None:
                return RentStepResult(
                    status="timeout",
                    ms=_elapsed_ms(started),
                    error=f"sshd on host port {step.ssh_host_port} did not answer within {step.ssh_timeout_s}s",
                    data=data,
                )
        return RentStepResult(status="ok", ms=_elapsed_ms(started), data=data)

    # --- rollback: by id or by our label, never by name ----------------------------------------

    async def rollback(self) -> bool:
        """Remove what this run created. True when nothing of ours provably remains: removed by id,
        removed by label, or a create the daemon itself refused (nothing exists). A create that
        neither answered nor is found by label yet (cut mid-create, or failed on the socket) is
        reported False and looked for once more later — the validator frees the name itself meanwhile."""
        nonce = self.body.nonce
        if self.container_id is not None:
            return await self._remove_by_id(self.container_id)
        if self.daemon_refused_create:
            # The daemon answered the create with an error: nothing of ours exists (a name conflict
            # is somebody else's container — left alone). A create that failed on the socket instead
            # (the daemon restarted, the read timed out) may have gone through: looked up by label.
            return True
        found = await self._remove_by_label(nonce)
        if found is True:
            return True  # the daemon had finished it after all; ours by label, removed
        # Cut mid-create and nothing to see yet: the daemon may still finish it. Once more, later,
        # off this request — and the validator frees the name itself meanwhile. The service holds
        # the task: the loop keeps only weak references to tasks.
        self.service.hold_late_rollback(asyncio.get_running_loop().create_task(self._remove_later(nonce)))
        return False

    async def _remove_later(self, nonce: str) -> None:
        await asyncio.sleep(ROLLBACK_RETRY_SECONDS)
        try:
            api = await self.service._open_api()
        except Exception as exc:  # noqa: BLE001 — logged; the validator's cleanup is the backstop
            logger.warning("local rent: late rollback has no docker client: %s", exc)
            return
        try:
            self.api = api
            await self._remove_by_label(nonce)
        finally:
            _close(api)

    async def _remove_by_id(self, container_id: str) -> bool:
        def remove() -> None:
            self.api.remove_container(container_id, v=True, force=True)

        try:
            await self._in_thread(remove, REMOVE_TIMEOUT_SECONDS)
            logger.info("local rent: rolled back %s", container_id[:12])
            return True
        except Exception as exc:  # noqa: BLE001 — logged; the validator's cleanup is the backstop
            if _is_not_found(exc):
                return True
            logger.warning("local rent: rollback of %s failed: %s", container_id[:12], exc)
            return False

    async def _remove_by_label(self, nonce: str) -> bool | None:
        """Remove every container carrying our nonce label. True: found and removed; None: none
        found; False: found and the removal failed."""

        def find() -> list[str]:
            listed = self.api.containers(all=True, filters={"label": f"{NONCE_LABEL}={nonce}"}, quiet=True) or []
            return [c.get("Id") for c in listed if isinstance(c, dict) and c.get("Id")]

        try:
            ids = await self._in_thread(find, INSPECT_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("local rent: listing by label failed: %s", exc)
            return False
        if not ids:
            return None
        ok = True
        for container_id in ids:
            ok = await self._remove_by_id(container_id) and ok
        return ok


def _close(api: Any) -> None:
    close = getattr(api, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 — a client that will not close is not the rental's problem
            pass


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {k: state.get(k) for k in ("Status", "Running", "ExitCode", "Error", "StartedAt") if k in state}


def _is_not_found(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 404 or type(exc).__name__ in ("NotFound", "ImageNotFound")


def _is_daemon_answer(exc: Exception) -> bool:
    """docker-py's `APIError` carries the daemon's HTTP response; a socket-level failure
    (`requests` ConnectionError / ReadTimeout) carries none."""
    return getattr(getattr(exc, "response", None), "status_code", None) is not None


async def _wait_ssh_banner(hosts: list[str], port: int, timeout_s: float) -> str | None:
    """The host among `hosts` on which host:port answered a TCP connection WITH an SSH banner
    (`SSH-…`) within timeout_s, else None. A bare connect is not enough: docker's userland proxy
    accepts on the published port before anything listens inside the container (measured: 8 ms
    after start for an sshd that takes seconds), so only the banner says sshd is up. Hosts are
    tried in turn: the loopback (a host-network executor) and the bridge gateway (the host)."""
    until = time.perf_counter() + timeout_s
    attempt = 0
    while True:
        remaining = until - time.perf_counter()
        if remaining <= 0:
            return None
        host = hosts[attempt % len(hosts)]
        attempt += 1
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=min(SSH_CONNECT_ATTEMPT_SECONDS, remaining)
            )
        except (OSError, TimeoutError):
            await asyncio.sleep(min(SSH_POLL_INTERVAL_SECONDS, max(0.0, until - time.perf_counter())))
            continue
        try:
            head = await asyncio.wait_for(reader.read(4), timeout=min(SSH_BANNER_READ_SECONDS, max(0.05, remaining)))
        except (OSError, TimeoutError):
            head = b""
        finally:
            writer.close()
        if head.startswith(b"SSH-"):
            return host
        await asyncio.sleep(min(SSH_POLL_INTERVAL_SECONDS, max(0.0, until - time.perf_counter())))


class BusyError(RuntimeError):
    pass
