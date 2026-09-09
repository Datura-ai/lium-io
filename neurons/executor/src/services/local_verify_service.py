"""One validator-signed intent in, one result out: the verification suite run locally.

Today the validator drives every check over SSH: `python decrypt_challenge.py …` for the matmul,
`python verifyx_executor.py …` for VerifyX, `docker …` for the runtime facts — one round trip per
command, 25–40 per cycle (speed/SWEEP_provider_verify.md §1). This service runs the same scripts
with the same arguments as local subprocesses, side by side where they are independent, and hands
the raw output back in one document. Nothing here decides pass or fail: the validator unseals the
matmul blob and decrypts the VerifyX response with the keys only it holds, exactly as it does with
SSH output, so the two paths are judged by one function and stay comparable.

Off unless `EXECUTOR_LOCAL_VERIFY_ENABLED=true`; `/version` advertises `local_verify/1` only then.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datura.requests.validator_requests import local_verify_signing_blob
from payloads.verify import (
    STEP_NAMES,
    CardRun,
    ContainerFact,
    DiskFact,
    DockerFacts,
    InspectorFacts,
    MatmulData,
    MatmulStep,
    PortFacts,
    StepData,
    StepResult,
    VerifyIntentBody,
    VerifyResult,
    VerifyXData,
    VerifyXStep,
)

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent.parent
MATMUL_SCRIPT = SRC_DIR / "decrypt_challenge.py"
VERIFYX_SCRIPT = SRC_DIR / "verifyx_executor.py"
INSPECTOR_SCRIPT = SRC_DIR / "inspector_executor.py"
LIBVERIFYX_PATH = "/usr/lib/libverifyx.so"
LIBINSPECTOR_PATH = "/usr/lib/libinspector.so"

# Same caps the validator applies to the SSH-driven runs (matrix_validation_service /
# verifyx_validation_service): a local run that outlives them would be judged a timeout there too.
MATMUL_TIMEOUT_SECONDS = 120
VERIFYX_TIMEOUT_SECONDS = 600
FAST_STEP_TIMEOUT_SECONDS = 20
STDERR_TAIL_BYTES = 2048
STDOUT_MAX_BYTES = 256 * 1024
PORT_SAMPLE_MAX = 4096
KILL_WAIT_SECONDS = 5

# The fact collectors call docker-py, which must not run in asyncio's default executor (routes/
# apis.py keeps its metrics pool separate for the same reason): a wedged daemon leaks a thread into
# this pool only, never into the one asyncio uses for getaddrinfo and friends.
_facts_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="local-verify-facts")

# The bytes the validator signs: the intent without `signature`, canonical JSON — ONE definition
# for both sides, in datura (validator: services/local_verify_client.py imports the same function).
canonical_intent_message = local_verify_signing_blob


class NonceCache:
    """Refuses a nonce seen before its expiry. In-memory: a restart forgets, and a replayed
    intent is then still refused by the issued_at window."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, nonce: str, expires_at: float, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            for stale in [n for n, exp in self._seen.items() if exp <= now]:
                del self._seen[stale]
            if nonce in self._seen:
                return False
            self._seen[nonce] = expires_at
            return True


def check_intent_window(body: VerifyIntentBody, now: float, window_s: int) -> str | None:
    """None when the intent is fresh; otherwise why it is refused."""
    if body.expires_at <= now:
        return "intent expired"
    if abs(body.issued_at - now) > window_s:
        return "issued_at outside the accepted window"
    if body.expires_at - body.issued_at > 2 * window_s + 60:
        return "expiry too far from issued_at"
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_of_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _tail(data: bytes) -> str:
    return data[-STDERR_TAIL_BYTES:].decode("utf-8", errors="ignore")


def _cap(data: bytes) -> str:
    return data[:STDOUT_MAX_BYTES].decode("utf-8", errors="ignore")


# --- the steps ---------------------------------------------------------------------------------
# Each returns a StepResult and never raises: an exception is `failed` with the error text, so one
# broken step cannot take the others' evidence down with it.


async def run_script(
    argv: list[str], *, timeout: float, env: dict[str, str] | None = None
) -> StepResult:
    """Run one of the executor's own scripts the way an SSH session would, capturing everything."""
    started = time.perf_counter()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(env or {})},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            # An SSH timeout leaves the remote matmul holding VRAM until a separate kill; here the
            # process is ours to end — and to wait for, so the next intent's matmul does not start
            # while this one's VRAM is still being torn down.
            await _kill(proc)
            return StepResult(
                status="timeout",
                ms=int((time.perf_counter() - started) * 1000),
                error=f"timed out after {timeout:.0f}s",
            )
        stdout_text = _cap(stdout)
        return StepResult(
            status="ok" if proc.returncode == 0 else "failed",
            ms=int((time.perf_counter() - started) * 1000),
            exit_status=proc.returncode,
            stdout=stdout_text,
            stderr_tail=_tail(stderr),
            stdout_sha256=_sha256_text(stdout_text),
        )
    except asyncio.CancelledError:
        if proc is not None:
            await _kill(proc)
        raise
    except Exception as exc:  # noqa: BLE001 — evidence, not control flow
        return StepResult(
            status="failed",
            ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


async def _kill(proc) -> None:
    """SIGKILL the child and reap it (bounded), so its transport closes and its VRAM is released
    before the suite lock is."""
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=KILL_WAIT_SECONDS)
    except TimeoutError:
        # The SIGKILL is sent; the child watcher reaps it. A CancelledError propagates: no step
        # may return a result after its task was cancelled (the group would run on, see `_run`).
        pass


def matmul_argv(
    step: MatmulStep, python: str, *, seed: int | None = None, cipher_text: str | None = None
) -> list[str]:
    # Byte-for-byte the SSH path's `python decrypt_challenge.py --dim_n … --dim_k … --seed …
    # --cipher_text …` (VerifierParams.__str__ on the validator); a pinned card run passes its own
    # seed and cipher text.
    return [
        python,
        str(MATMUL_SCRIPT),
        "--dim_n",
        str(step.dim_n),
        "--dim_k",
        str(step.dim_k),
        "--seed",
        str(step.seed if seed is None else seed),
        "--cipher_text",
        step.cipher_text if cipher_text is None else cipher_text,
    ]


def verifyx_argv(step: VerifyXStep, python: str) -> list[str]:
    # The SSH path's `python verifyx_executor.py --seed … --cipher_text …`.
    return [
        python,
        str(VERIFYX_SCRIPT),
        "--seed",
        str(step.seed),
        "--cipher_text",
        step.cipher_text,
    ]


async def run_matmul(step: MatmulStep, *, python: str = sys.executable) -> StepResult:
    if step.devices:
        # The all-cards work-proof: one run pinned per card, together, like the validator's fan-out.
        started = time.perf_counter()
        # Each card answers its own challenge (DeviceChallenge): one computation cannot stand in for
        # every card the host claims.
        per_card = await asyncio.gather(
            *(
                run_script(
                    matmul_argv(step, python, seed=device.seed, cipher_text=device.cipher_text),
                    timeout=MATMUL_TIMEOUT_SECONDS,
                    env={"CUDA_VISIBLE_DEVICES": str(device.index)},
                )
                for device in step.devices
            )
        )
        cards = [
            CardRun(card_index=device.index, **card.model_dump(exclude={"data"}))
            for device, card in zip(step.devices, per_card)
        ]
        failed = [c for c in per_card if c.status != "ok"]
        return StepResult(
            status="ok"
            if not failed
            else ("timeout" if any(c.status == "timeout" for c in failed) else "failed"),
            ms=int((time.perf_counter() - started) * 1000),
            data=MatmulData(per_card=cards),
        )
    return await run_script(matmul_argv(step, python), timeout=MATMUL_TIMEOUT_SECONDS)


async def run_verifyx(step: VerifyXStep, *, python: str = sys.executable) -> StepResult:
    # The validator compares the library digest before it trusts a response (core/checksums);
    # over SSH that is one more command, here it rides along — hashed off the event loop (10 MB) on
    # the facts pool like `_inspector_facts`, started BEFORE the script so it is normally long done
    # when the script's seconds are over. A deadline that still lands in the digest drops this run:
    # re-raised, never swallowed (a swallowed cancellation would let the GPU group run on, `_run`).
    loop = asyncio.get_running_loop()
    digest_future = loop.run_in_executor(_facts_executor, sha256_of_file, LIBVERIFYX_PATH)
    try:
        result = await run_script(verifyx_argv(step, python), timeout=VERIFYX_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        digest_future.cancel()
        raise
    try:
        digest = await asyncio.wait_for(digest_future, timeout=FAST_STEP_TIMEOUT_SECONDS)
    except TimeoutError:
        digest = None
    result.data = VerifyXData(lib_sha256=digest)
    return result


def _docker_facts() -> DockerFacts:
    import docker  # local import: the module is mocked in unit tests

    client = docker.from_env(timeout=FAST_STEP_TIMEOUT_SECONDS)
    info = client.info()
    root_dir = info.get("DockerRootDir")
    runtimes = sorted((info.get("Runtimes") or {}).keys())
    containers = []
    for container in client.containers.list(all=True):
        attrs = container.attrs or {}
        containers.append(
            ContainerFact(
                name=container.name,
                status=container.status,
                image=(attrs.get("Config") or {}).get("Image"),
                created=attrs.get("Created"),
            )
        )
    disk = None
    if root_dir and os.path.isdir(root_dir):
        usage = shutil.disk_usage(root_dir)
        disk = DiskFact(total_bytes=usage.total, free_bytes=usage.free, used_bytes=usage.used)
    return DockerFacts(
        server_version=info.get("ServerVersion"),
        root_dir=root_dir,
        runtimes=runtimes,
        default_runtime=info.get("DefaultRuntime"),
        sysbox_runtime="sysbox-runc" in runtimes,
        disk=disk,
        containers=containers,
    )


def _published_host_ports() -> set[int]:
    import docker  # local import: the module is mocked in unit tests

    client = docker.from_env(timeout=FAST_STEP_TIMEOUT_SECONDS)
    published: set[int] = set()
    for container in client.containers.list():
        for bindings in (container.ports or {}).values():
            for binding in bindings or []:
                try:
                    published.add(int(binding.get("HostPort")))
                except (TypeError, ValueError):
                    continue
    return published


# What the validator counts when an executor configures neither setting (port_utils.DEFAULT_PORT_RANGE):
# the same default here, or the facts would say "none" for a host the validator rents on 20000–65535.
DEFAULT_PORT_RANGE = (20000, 65536)


def parse_port_range(port_range: str | None, port_mappings: str | None) -> list[tuple[int, int]]:
    """(internal, external) pairs the way the validator's `port_utils.get_all_ports` reads them."""
    if port_mappings:
        return sorted((int(i), int(e)) for i, e in json.loads(port_mappings))
    if port_range:
        if "-" in port_range:
            lo, hi = (int(p.strip()) for p in port_range.split("-"))
            return [(p, p) for p in range(lo, hi + 1)]
        return sorted((int(p.strip()), int(p.strip())) for p in port_range.split(","))
    return [(p, p) for p in range(*DEFAULT_PORT_RANGE)]


def _port_facts(port_range: str | None, port_mappings: str | None, ssh_port: int) -> PortFacts:
    pairs = [
        (i, e)
        for i, e in parse_port_range(port_range, port_mappings)
        if i != ssh_port and e != ssh_port
    ]
    sample = pairs[:PORT_SAMPLE_MAX]
    published = _published_host_ports()
    in_use = sorted(e for _, e in sample if e in published)
    return PortFacts(
        port_range=port_range,
        port_mappings=port_mappings,
        configured=len(pairs),
        sampled=len(sample),
        published_by_docker=in_use,
        free=len(sample) - len(in_use),
    )


def _inspector_facts() -> InspectorFacts:
    return InspectorFacts(
        lib_present=os.path.isfile(LIBINSPECTOR_PATH),
        lib_sha256=sha256_of_file(LIBINSPECTOR_PATH),
        script_present=INSPECTOR_SCRIPT.is_file(),
    )


async def run_facts(name: str, func: Callable[[], StepData]) -> StepResult:
    """A read-only fact collector, off the event loop, bounded like every other step."""
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        data = await asyncio.wait_for(
            loop.run_in_executor(_facts_executor, func), timeout=FAST_STEP_TIMEOUT_SECONDS
        )
        return StepResult(status="ok", ms=int((time.perf_counter() - started) * 1000), data=data)
    except TimeoutError:
        return StepResult(
            status="timeout",
            ms=int((time.perf_counter() - started) * 1000),
            error=f"{name} timed out after {FAST_STEP_TIMEOUT_SECONDS}s",
        )
    except Exception as exc:  # noqa: BLE001 — a fact collector's error is evidence for the validator, not a 500
        return StepResult(
            status="failed",
            ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )


# --- the runner --------------------------------------------------------------------------------

StepRunner = Callable[[], Awaitable[StepResult]]


class LocalVerifyService:
    """Runs one intent. One at a time per executor: a second concurrent suite would compete for
    the GPU and RAM the first is measuring and make both results wrong."""

    def __init__(
        self,
        *,
        executor_version: str,
        max_deadline_s: int,
        port_range: str | None = None,
        port_mappings: str | None = None,
        ssh_port: int = 22,
        python: str = sys.executable,
    ) -> None:
        self.executor_version = executor_version
        self.max_deadline_s = max_deadline_s
        self.port_range = port_range
        self.port_mappings = port_mappings
        self.ssh_port = ssh_port
        self.python = python
        self._busy = asyncio.Lock()

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def _runners(self, body: VerifyIntentBody) -> tuple[dict[str, StepRunner], list[str]]:
        steps = body.steps
        gpu: dict[str, StepRunner] = {}
        facts: dict[str, StepRunner] = {}
        if steps.matmul is not None:
            gpu["matmul"] = lambda: run_matmul(steps.matmul, python=self.python)
        if steps.verifyx is not None:
            gpu["verifyx"] = lambda: run_verifyx(steps.verifyx, python=self.python)
        if steps.docker:
            facts["docker"] = lambda: run_facts("docker", _docker_facts)
        if steps.ports:
            facts["ports"] = lambda: run_facts(
                "ports", lambda: _port_facts(self.port_range, self.port_mappings, self.ssh_port)
            )
        if steps.inspector:
            facts["inspector"] = lambda: run_facts("inspector", _inspector_facts)
        gpu_order = list(gpu)
        return {**gpu, **facts}, gpu_order

    async def run(self, body: VerifyIntentBody) -> VerifyResult:
        if self._busy.locked():
            raise BusyError("a verification is already running")
        async with self._busy:
            return await self._run(body)

    async def _run(self, body: VerifyIntentBody) -> VerifyResult:
        started_wall = int(time.time())
        started = time.perf_counter()
        deadline = min(body.deadline_s, self.max_deadline_s)
        runners, gpu_order = self._runners(body)
        results: dict[str, StepResult] = {}

        async def step(name: str) -> None:
            # Each step writes its own result the moment it finishes, so a deadline that cancels
            # the group keeps the evidence of every sibling that had already completed.
            results[name] = await runners[name]()

        async def gpu_group() -> None:
            # VerifyX then matmul, the pipeline's order, unless the validator asked for both at once.
            if body.parallel_gpu:
                await asyncio.gather(*(step(n) for n in gpu_order))
                return
            for name in sorted(gpu_order, key=lambda n: 0 if n == "verifyx" else 1):
                await step(name)

        tasks = [asyncio.ensure_future(step(n)) for n in runners if n not in gpu_order]
        if gpu_order:
            tasks.append(asyncio.ensure_future(gpu_group()))

        deadline_hit = False
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=deadline)
            if pending:
                deadline_hit = True
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    logger.error("local verify step group failed: %s", exc)

        for name in runners:
            results.setdefault(
                name,
                StepResult(status="timeout", error=f"not finished within the {deadline}s deadline"),
            )
        for name in STEP_NAMES:
            results.setdefault(name, StepResult(status="skipped"))

        return VerifyResult(
            nonce=body.nonce,
            executor_uuid=body.executor_uuid,
            executor_version=self.executor_version,
            started_at=started_wall,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            deadline_hit=deadline_hit,
            steps=results,
        )


class BusyError(RuntimeError):
    pass
