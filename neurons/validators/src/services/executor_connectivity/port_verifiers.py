import asyncio
import logging
import random
import uuid
from dataclasses import dataclass

import aiohttp
from asyncssh import SSHClientConnection

from core.utils import _m, get_extra_info
from services.const import MIN_PORT_COUNT
from services.executor_connectivity.container_runner import ContainerRunner
from services.executor_connectivity.external_probe import ExternalPortProbe
from services.executor_connectivity.models import BatchResult, ContainerStartResult, PortPair
from services.executor_connectivity.netcat_script import NetcatScript
from services.executor_connectivity.port_tester import PortTester
from services.node_host import public_host_connector

logger = logging.getLogger(__name__)


# DAH-4537: ports fetched from outside first; each netcat listener answers once, so these are not
# fetched again by the validator unless the outside fetch missed them. DAH-5597: they are published
# with -p in their own container, so the outside fetch takes a renter pod's path
EXTERNAL_PROBE_SAMPLE = 3


@dataclass
class ContainerStartWatch:
    """DAH-5682: whether `docker run` of an attempt's test container has not returned yet."""

    in_progress: int = 0  # `docker run` calls not returned yet: the host listener starts alongside the outside one
    started_at: float = 0.0  # loop time the oldest in-progress start began
    missed_container: str | None = None  # the late host listener for sampled ports the outside reading missed
    # the ports that answered so far: an attempt whose deadline fires after they answered still returns them
    verified: BatchResult | None = None


class BatchVerifier:
    """Verifies ports using --network=host in batches."""

    REPROBE_DELAY_SEC = 3
    REPROBE_ROUNDS = 5  # ~15s budget: a filler download can hold a loaded host past one 3s window
    # the published sample already met the floor: a late host listener answers within seconds, while a host that
    # drops its host ports (ufw default deny) never does and must reach the semi-batch tier without the full budget
    SAMPLE_ONLY_REPROBE_ROUNDS = 2
    ATTEMPT_TIMEOUT_SEC = 60
    # a start the attempt deadline cut off counts as hung only if it had this long: on a slow host the outside
    # probe's container can use most of the 60 s, leaving the batch start seconds before the deadline
    HUNG_START_MIN_SEC = 30

    def __init__(
        self, port_tester: PortTester, runner: ContainerRunner, external_probe: ExternalPortProbe | None = None
    ):
        self.port_tester = port_tester
        self.runner = runner
        self.external_probe = external_probe

    async def verify(
        self,
        ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
        min_ports: int = MIN_PORT_COUNT,
    ) -> BatchResult:
        """Verify ports with retries; refused ports are re-probed until `min_ports` answer."""
        log_ctx = log_ctx or {}
        timeout_sec = self.ATTEMPT_TIMEOUT_SEC
        max_attempts = 2

        external = None  # the last attempt's outside reading, kept when no attempt verified a port
        start_hung = 0
        for attempt in range(1, max_attempts + 1):
            token = uuid.uuid4().hex
            container_name = f"port_test_{token[:8]}"
            start_watch = ContainerStartWatch()

            logger.info(
                _m(
                    f"testing {len(ports)} ports (attempt {attempt}/{max_attempts}, timeout={timeout_sec}s)",
                    extra=get_extra_info(log_ctx),
                )
            )

            try:
                attempt_result = await asyncio.wait_for(
                    self._attempt(ports, token, container_name, ssh_client, host, start_watch, log_ctx, min_ports),
                    timeout=timeout_sec
                )
                external = attempt_result.external
                if attempt_result.successful:
                    logger.info(
                        _m(
                            f"complete: {len(attempt_result.successful)}/{len(ports)} verified",
                            extra=get_extra_info(log_ctx),
                        )
                    )
                    return attempt_result

                if attempt < max_attempts:
                    logger.debug(
                        _m(f"attempt {attempt} failed, retrying in 2s", extra=get_extra_info(log_ctx))
                    )
                    await asyncio.sleep(2)

            except asyncio.TimeoutError:
                if start_watch.verified and len(start_watch.verified.successful) >= min_ports:
                    # the deadline fired after enough ports answered (a loaded host's starts use
                    # most of the 60 s); fewer than min_ports would fail the port check, so
                    # attempt 2 and the slower tiers run
                    logger.info(
                        _m(
                            f"complete: {len(start_watch.verified.successful)}/{len(ports)} verified at the deadline",
                            extra=get_extra_info(log_ctx),
                        )
                    )
                    return start_watch.verified
                start_hung += start_watch.in_progress and (
                    asyncio.get_running_loop().time() - start_watch.started_at >= self.HUNG_START_MIN_SEC
                )
                logger.debug(
                    _m(
                        f"attempt {attempt} timed out after {timeout_sec}s"
                        + (" inside the container start" if start_watch.in_progress else ""),
                        extra=get_extra_info(log_ctx),
                    )
                )
                if attempt < max_attempts:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(
                    _m(f"attempt {attempt} failed: {e}", extra=get_extra_info(log_ctx)),
                    exc_info=True,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(2)

            finally:
                # DAH-5868: outside wait_for, so a slow cleanup cannot time out an attempt whose ports answered
                try:
                    await self.runner.cleanup(ssh_client, container_name)
                    if start_watch.missed_container:
                        await self.runner.cleanup(ssh_client, start_watch.missed_container)
                except Exception as e:
                    logger.debug(_m(f"cleanup failed: {str(e)[:50]}", extra=get_extra_info(log_ctx)))

        # DAH-3593: the outcome is the PortCountCheck verdict ("Port verification failed"); the
        # attempts and tiers on the way there are DEBUG. An exception inside an attempt keeps
        # ERROR above: that is the validator's own failure, not the node's ports.
        logger.debug(_m(f"all {max_attempts} attempts failed", extra=get_extra_info(log_ctx)))
        return BatchResult([], ports, external, start_hung=start_hung == max_attempts)

    async def _attempt(
        self,
        ports: list[PortPair],
        token: str,
        name: str,
        ssh_client,
        host: str,
        start_watch: ContainerStartWatch,
        log_ctx: dict | None = None,
        min_ports: int = MIN_PORT_COUNT,
    ) -> BatchResult:
        """Single verification attempt."""
        log_ctx = log_ctx or {}
        sample = random.sample(ports, min(EXTERNAL_PROBE_SAMPLE, len(ports))) if self.external_probe else []
        rest = [p for p in ports if p not in sample]
        loop = asyncio.get_running_loop()
        started = loop.time()
        timings: dict[str, float] = {}

        async def timed(key: str, step):
            began = loop.time()
            result = await step
            timings[key] = loop.time() - began
            return result

        def host_start(listen: list[PortPair], container: str, idx: int, key: str, watch=start_watch):
            return timed(key, self._start_container(
                watch, ssh_client, container, NetcatScript.batch(listen, token, idx), "host", 60
            ))

        async def optional_start(listen: list[PortPair], container: str) -> ContainerStartResult:
            # the missed ports' listener is extra: its failure must not discard the ports that already answered
            try:
                # its own watch: a slow extra start is not a hung main start, so it never makes `start_hung` skip
                # the semi-batch and fallback tiers; cancel and cleanup still go through `missed_start`
                return await host_start(listen, container, 1, "missed_start", ContainerStartWatch())
            except Exception as exc:  # not CancelledError: a cancelled start still ends the attempt
                logger.warning(_m(f"missed-port listener start failed: {exc!r}", extra=get_extra_info(log_ctx)))
                return ContainerStartResult(ok=False, container_id=None, status=repr(exc))

        outside = timed("outside", self._outside_reading(sample, token, ssh_client, host, log_ctx, start_watch, timings))
        # DAH-6133: on a busy host each `docker run` takes seconds, and run one after another they used most of the
        # 60 s; the host listener for the ports outside the sample starts alongside the outside reading. Not when
        # those ports alone cannot reach the floor (a split node needing every port): then it waits for the sample
        parallel = bool(sample) and len(rest) >= min_ports
        if parallel:
            steps = [asyncio.ensure_future(outside), asyncio.ensure_future(host_start(rest, name, 0, "host_start"))]
            try:
                (reached, first_round), start_result = await asyncio.gather(*steps)
            except BaseException:
                # a start left running would bind its listener after the cleanup and answer attempt 2 stale
                for step in steps:
                    step.cancel()
                raise
        else:
            reached, first_round = await outside
        sampled_ok = [p for p in sample if p in (reached or []) or p in first_round]
        missed = [p for p in sample if p not in sampled_ok]
        external_blocked = reached == [] and all(p in first_round for p in sample)
        if external_blocked:
            logger.warning(
                _m(
                    f"ports {[p.external for p in sample]} answer the validator but not the outside prober",
                    extra=get_extra_info(log_ctx),
                )
            )
        external = "blocked" if external_blocked else "passed" if reached else None

        def ports_answered_so_far() -> BatchResult:
            ok = set(sampled_ok + successful)
            return BatchResult([p for p in ports if p in ok], [p for p in ports if p not in ok], external)

        missed_start: asyncio.Future | None = None
        if parallel:
            ports_left = rest
            if start_result.ok and missed:
                # the outside container held these ports; their host listener can bind now that it is gone.
                # It starts beside the probe of `rest`: a slow start must not spend the deadline those ports need
                start_watch.missed_container = f"{name}_s"
                missed_start = asyncio.ensure_future(optional_start(missed, start_watch.missed_container))
        else:
            ports_left = [p for p in ports if p not in sampled_ok]
            if not ports_left:
                return BatchResult(sampled_ok, [], external)
            start_result = await host_start(ports_left, name, 0, "host_start")
        try:
            if not start_result.ok:
                logger.debug(
                    _m(
                        f"batch start failed: status={start_result.status} logs={start_result.logs}",
                        extra=get_extra_info(log_ctx),
                    )
                )
                return BatchResult([], ports, external)
            logger.info(
                _m(
                    f"containers ready in {loop.time() - started:.1f}s ({'parallel' if parallel else 'sequential'}): "
                    + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()),
                    extra=get_extra_info(log_ctx),
                )
            )
            async with aiohttp.ClientSession(connector=public_host_connector(limit=0)) as session:
                missed_pending = missed_start is not None

                def missed_ready() -> list[PortPair]:
                    # never wait on the missed ports' start: the refused ports' re-probes run on schedule
                    nonlocal missed_pending
                    if not missed_pending or not missed_start.done():
                        return []
                    missed_pending = False
                    return missed if missed_start.result().ok else []

                successful, failed = await self.port_tester.test_many(session, host, ports_left, token, log_ctx)
                start_watch.verified = ports_answered_so_far() if successful else None
                # a loaded host binds its nc listeners late; a refused port's listener is still waiting, so
                # re-probe the refused ports while the count is below what PortCountCheck needs to list the node;
                # the published sample alone never ends it: with no host-listener answer the attempt fails below
                rounds = self.SAMPLE_ONLY_REPROBE_ROUNDS if len(sampled_ok) >= min_ports else self.REPROBE_ROUNDS

                async def reprobe(refused: list[PortPair]) -> list[PortPair]:
                    nonlocal successful
                    round_no = 0
                    while round_no < rounds:
                        if not refused or (successful and len(sampled_ok) + len(successful) >= min_ports):
                            break
                        round_no += 1
                        await asyncio.sleep(self.REPROBE_DELAY_SEC)
                        joined = missed_ready()
                        if joined:
                            round_no = 0  # their first probe: late listeners joining any round get every re-probe after it
                        late, refused = await self.port_tester.test_many(
                            session, host, refused + joined, token, log_ctx
                        )
                        successful = successful + late
                        start_watch.verified = ports_answered_so_far() if successful else None
                        logger.debug(
                            _m(f"re-probe {round_no}/{rounds}: {len(late)} late", extra=get_extra_info(log_ctx))
                        )
                    return refused

                failed = await reprobe(failed)
                if missed_pending:
                    # the re-probes are done, so nothing waits behind the missed ports' start now
                    await asyncio.wait([missed_start])
                if ready := missed_ready():
                    late, still = await self.port_tester.test_many(session, host, ready, token, log_ctx)
                    successful = successful + late
                    start_watch.verified = ports_answered_so_far() if successful else None
                    # their listener started late too: a refused first probe gets the same re-probes
                    failed = failed + await reprobe(still)
                logger.info(
                    _m(f"progress: {len(successful)}/{len(ports)} verified", extra=get_extra_info(log_ctx))
                )

            if not successful:
                # DAH-5597: only the host listener decides this tier; a host that drops INPUT but forwards published
                # ports answers the published sample, and must still reach the semi-batch tier
                return BatchResult([], ports, external)
            return ports_answered_so_far()
        finally:
            if missed_start is not None and not missed_start.done():
                # a start left running would bind its listener after the cleanup and answer attempt 2 stale
                missed_start.cancel()

    async def _start_container(
        self,
        start_watch: ContainerStartWatch,
        ssh_client: SSHClientConnection,
        name: str,
        script: str,
        network_mode: str,
        timeout: int,
    ) -> ContainerStartResult:
        # no `finally`: a cancelled start must stay marked in progress for the attempt's timeout handler
        if not start_watch.in_progress:
            start_watch.started_at = asyncio.get_running_loop().time()
        start_watch.in_progress += 1
        result = await self.runner.run(ssh_client, name, script, network_mode, timeout)
        start_watch.in_progress -= 1
        return result

    async def _outside_reading(
        self,
        sample: list[PortPair],
        token: str,
        ssh_client,
        host: str,
        log_ctx: dict,
        start_watch: ContainerStartWatch,
        timings: dict[str, float] | None = None,
    ) -> tuple[list[PortPair] | None, list[PortPair]]:
        """DAH-5597: the outside prober fetches ports published with `-p`, as a renter's pod publishes them, so it
        goes through the docker DNAT/FORWARD path; a --network=host listener would test the host INPUT chain
        instead. Returns the outside reading (None: no reading) and the sampled ports the validator then reached
        on the same listener, first try only: the prober ran before them, so a late bind refuses it too."""
        if not sample:
            return None, []
        name = f"port_test_out_{token[:8]}"
        publish_flags = " ".join(f"-p {p.internal}:{p.internal}" for p in sample)
        began = asyncio.get_running_loop().time()
        start = await self._start_container(
            start_watch, ssh_client, name, NetcatScript.batch(sample, token, 0), publish_flags, 30
        )
        if timings is not None:
            timings["outside_start"] = asyncio.get_running_loop().time() - began
        try:
            if not start.ok:
                logger.debug(
                    _m(f"outside probe start failed: status={start.status}", extra=get_extra_info(log_ctx))
                )
                return None, []
            reached = await self.external_probe.reached(host, sample, token)
            remaining = [p for p in sample if p not in (reached or [])]
            if not remaining:
                return reached, []
            async with aiohttp.ClientSession(connector=public_host_connector(limit=0)) as session:
                first_round, _ = await self.port_tester.test_many(session, host, remaining, token, log_ctx)
            return reached, first_round
        finally:
            await self.runner.cleanup(ssh_client, name)


class FallbackVerifier:
    """Verifies ports sequentially using -p publish."""

    # DAH-5419: `docker run` over SSH has no timeout of its own; a stuck daemon held it until the task was dropped
    START_TIMEOUT_SEC = 30

    def __init__(self, port_tester: PortTester, runner: ContainerRunner):
        self.port_tester = port_tester
        self.runner = runner

    async def verify(
        self,
        ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        max_ports: int = 10,
        log_ctx: dict | None = None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Test ports one by one."""
        log_ctx = log_ctx or {}
        ports_to_test = ports[:max_ports]
        logger.info(
            _m(f"fallback: testing {len(ports_to_test)} ports sequentially", extra=get_extra_info(log_ctx))
        )

        successful, failed = [], []

        async with aiohttp.ClientSession(connector=public_host_connector()) as session:
            for idx, port in enumerate(ports_to_test, 1):
                token = uuid.uuid4().hex
                name = f"port_test_seq_{token[:8]}"
                script = NetcatScript.single(port, token)
                network_flag = f"-p {port.internal}:{port.internal}"

                started = False
                try:
                    start_result = await asyncio.wait_for(
                        self.runner.run(ssh_client, name, script, network_flag, 10), self.START_TIMEOUT_SEC
                    )
                    if not start_result.ok:
                        logger.debug(
                            _m(
                                f"fallback: port {port.internal} failed to start: "
                                f"status={start_result.status} logs={start_result.logs}",
                                extra=get_extra_info(log_ctx),
                            )
                        )
                        failed.append(port)
                        continue

                    started = True
                    await asyncio.sleep(1.0)
                    if await self.port_tester.test_one(session, host, port, token):
                        logger.info(
                            _m(
                                f"fallback: port {port.internal} ok ({idx}/{len(ports_to_test)})",
                                extra=get_extra_info(log_ctx),
                            )
                        )
                        successful.append(port)
                    else:
                        failed.append(port)

                except TimeoutError:
                    # the later ports would hang the same way; ending here lets the round report the port check
                    logger.warning(
                        _m(
                            f"fallback: container start timed out after {self.START_TIMEOUT_SEC}s, "
                            f"{len(ports_to_test) - idx + 1} port(s) left untested",
                            extra=get_extra_info(log_ctx),
                        )
                    )
                    failed.extend(ports_to_test[idx - 1:])
                    break

                except Exception as e:
                    logger.error(
                        _m(
                            f"fallback: error testing port {port.internal}: {str(e)[:100]}",
                            extra=get_extra_info(log_ctx),
                        )
                    )
                    failed.append(port)

                finally:
                    if started:
                        try:
                            await self.runner.cleanup(ssh_client, name)
                        except Exception as e:
                            logger.warning(
                                _m(f"fallback: cleanup failed: {str(e)[:50]}", extra=get_extra_info(log_ctx))
                            )

                    if idx < len(ports_to_test):
                        await asyncio.sleep(0.5)

        logger.info(
            _m(f"fallback: {len(successful)}/{len(ports_to_test)} verified", extra=get_extra_info(log_ctx))
        )
        return successful, failed


class SemiBatchVerifier:
    """Verifies ports by publishing them via -p, in one container per chunk.

    Middle tier of the cascade: published ports are reached through the nat/DOCKER
    DNAT chain, so they survive a ufw default-deny INPUT policy — the same policy
    that silently drops BatchVerifier's --network=host listeners. Returns nothing
    verified if every chunk fails to start, letting the caller fall through.
    """

    # docker refuses the whole `run` when a single published port is already allocated, so the
    # chunk size is the blast radius of one busy port (DAH-2527: idle fillers take the same low
    # ports this tier probes). Ten matches the sequential tier's cap.
    CHUNK_SIZE = 10
    # a healthy chunk answers in under 10 s (prod p90 9 s); a host too loaded to start a container
    # took 6.5 min and pushed rented B300s past the job deadline (20 Oct), so a stalled chunk fails
    CHUNK_TIMEOUT_SEC = 60

    def __init__(self, port_tester: PortTester, runner: ContainerRunner):
        self.port_tester = port_tester
        self.runner = runner

    async def verify(
        self,
        ports: list[PortPair],
        *,
        ssh_client: SSHClientConnection,
        host: str,
        max_ports: int = 50,
        log_ctx: dict | None = None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Publish up to max_ports in chunked containers, then probe them concurrently."""
        log_ctx = log_ctx or {}
        ports_to_test = ports[:max_ports]
        chunks = [
            ports_to_test[start : start + self.CHUNK_SIZE]
            for start in range(0, len(ports_to_test), self.CHUNK_SIZE)
        ]

        logger.info(
            _m(
                f"semi-batch: publishing {len(ports_to_test)} ports in {len(chunks)} containers",
                extra=get_extra_info(log_ctx),
            )
        )

        chunk_results = await asyncio.gather(
            *(self._verify_chunk_in_time(chunk, ssh_client=ssh_client, host=host, log_ctx=log_ctx) for chunk in chunks)
        )
        successful = [port for chunk_successful, _ in chunk_results for port in chunk_successful]
        failed = [port for _, chunk_failed in chunk_results for port in chunk_failed]

        logger.info(
            _m(f"semi-batch: {len(successful)}/{len(ports_to_test)} verified", extra=get_extra_info(log_ctx))
        )
        return successful, failed

    async def _verify_chunk_in_time(
        self,
        ports: list[PortPair],
        *,
        ssh_client: SSHClientConnection,
        host: str,
        log_ctx: dict,
    ) -> tuple[list[PortPair], list[PortPair]]:
        try:
            return await asyncio.wait_for(
                self._verify_chunk(ports, ssh_client=ssh_client, host=host, log_ctx=log_ctx),
                self.CHUNK_TIMEOUT_SEC,
            )
        except TimeoutError:
            # the container kills itself after its 20 s window, so nothing is left holding the ports
            logger.warning(
                _m(
                    f"semi-batch: chunk of {len(ports)} ports timed out after {self.CHUNK_TIMEOUT_SEC}s",
                    extra=get_extra_info(log_ctx),
                )
            )
            return [], ports

    async def _verify_chunk(
        self,
        ports: list[PortPair],
        *,
        ssh_client: SSHClientConnection,
        host: str,
        log_ctx: dict,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Publish one chunk in one container; a chunk that cannot start fails only its own ports."""
        token = uuid.uuid4().hex
        container_name = f"port_test_pub_{token[:8]}"
        script = NetcatScript.batch(ports, token, 0)
        # external differs from internal when the miner advertises port_mappings
        publish_flags = " ".join(f"-p {port.external}:{port.internal}" for port in ports)

        # any error below must not escape: an exception here would zero the whole
        # verification (fatal PortCountCheck) instead of falling through to the
        # sequential fallback tier
        started = False
        try:
            # short self-kill window: a missed cleanup would otherwise hold every published
            # port (and its docker-proxy) against the next cycle's verification
            start_result = await self.runner.run(ssh_client, container_name, script, publish_flags, 20)
            if not start_result.ok:
                logger.debug(
                    _m(
                        f"semi-batch start failed for {len(ports)} ports: "
                        f"status={start_result.status} logs={start_result.logs}",
                        extra=get_extra_info(log_ctx),
                    )
                )
                return [], ports

            started = True
            async with aiohttp.ClientSession(connector=public_host_connector(limit=0)) as session:
                return await self.port_tester.test_many(session, host, ports, token, log_ctx)

        except Exception as e:
            logger.error(
                _m(f"semi-batch: error: {str(e)[:100]}", extra=get_extra_info(log_ctx))
            )
            return [], ports

        finally:
            if started:
                try:
                    await self.runner.cleanup(ssh_client, container_name)
                except Exception as e:
                    logger.warning(
                        _m(f"semi-batch: cleanup failed: {str(e)[:50]}", extra=get_extra_info(log_ctx))
                    )
