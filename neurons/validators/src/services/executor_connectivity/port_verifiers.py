import asyncio
import logging
import random
import uuid

import aiohttp
from asyncssh import SSHClientConnection

from core.utils import _m, get_extra_info
from services.executor_connectivity.container_runner import ContainerRunner
from services.executor_connectivity.external_probe import ExternalPortProbe
from services.executor_connectivity.models import BatchResult, PortPair
from services.executor_connectivity.netcat_script import NetcatScript
from services.executor_connectivity.port_tester import PortTester
from services.node_host import public_host_connector

logger = logging.getLogger(__name__)


# DAH-4537: ports fetched from outside first; each netcat listener answers once, so these are not
# fetched again by the validator unless the outside fetch missed them
EXTERNAL_PROBE_SAMPLE = 3


class BatchVerifier:
    """Verifies ports using --network=host in batches."""

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
    ) -> BatchResult:
        """Verify ports with retries."""
        log_ctx = log_ctx or {}
        timeout_sec = 60
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            token = uuid.uuid4().hex
            container_name = f"port_test_{token[:8]}"

            logger.info(
                _m(
                    f"testing {len(ports)} ports (attempt {attempt}/{max_attempts}, timeout={timeout_sec}s)",
                    extra=get_extra_info(log_ctx),
                )
            )

            try:
                attempt_result = await asyncio.wait_for(
                    self._attempt(ports, token, container_name, ssh_client, host, log_ctx),
                    timeout=timeout_sec
                )
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
                logger.debug(
                    _m(f"attempt {attempt} timed out after {timeout_sec}s", extra=get_extra_info(log_ctx))
                )
                await self.runner.cleanup(ssh_client, container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(
                    _m(f"attempt {attempt} failed: {e}", extra=get_extra_info(log_ctx)),
                    exc_info=True,
                )
                await self.runner.cleanup(ssh_client, container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

        # DAH-3593: the outcome is the PortCountCheck verdict ("Port verification failed"); the
        # attempts and tiers on the way there are DEBUG. An exception inside an attempt keeps
        # ERROR above: that is the validator's own failure, not the node's ports.
        logger.debug(_m(f"all {max_attempts} attempts failed", extra=get_extra_info(log_ctx)))
        return BatchResult([], ports)

    async def _attempt(
        self,
        ports: list[PortPair],
        token: str,
        name: str,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
    ) -> BatchResult:
        """Single verification attempt."""
        log_ctx = log_ctx or {}
        script = NetcatScript.batch(ports, token, 0)
        start_result = await self.runner.run(ssh_client, name, script, "host", 60)
        if not start_result.ok:
            logger.debug(
                _m(
                    f"batch start failed: status={start_result.status} logs={start_result.logs}",
                    extra=get_extra_info(log_ctx),
                )
            )
            return BatchResult([], ports)

        try:
            sample = random.sample(ports, min(EXTERNAL_PROBE_SAMPLE, len(ports))) if self.external_probe else []
            reached = await self.external_probe.reached(host, sample, token) if sample else None
            async with aiohttp.ClientSession(connector=public_host_connector(limit=0)) as session:
                remaining = [p for p in ports if p not in (reached or [])]
                successful, failed = (
                    await self.port_tester.test_many(session, host, remaining, token, log_ctx) if remaining else ([], [])
                )
                logger.info(
                    _m(f"progress: {len(successful)}/{len(ports)} verified", extra=get_extra_info(log_ctx))
                )
        finally:
            await self.runner.cleanup(ssh_client, name)

        external_blocked = reached == [] and all(p in successful for p in sample)
        if external_blocked:
            logger.warning(
                _m(
                    f"ports {[p.external for p in sample]} answer the validator but not the outside prober",
                    extra=get_extra_info(log_ctx),
                )
            )
        return BatchResult((reached or []) + successful, failed, external_blocked)


class FallbackVerifier:
    """Verifies ports sequentially using -p publish."""

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
                    start_result = await self.runner.run(ssh_client, name, script, network_flag, 10)
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
            *(self._verify_chunk(chunk, ssh_client=ssh_client, host=host, log_ctx=log_ctx) for chunk in chunks)
        )
        successful = [port for chunk_successful, _ in chunk_results for port in chunk_successful]
        failed = [port for _, chunk_failed in chunk_results for port in chunk_failed]

        logger.info(
            _m(f"semi-batch: {len(successful)}/{len(ports_to_test)} verified", extra=get_extra_info(log_ctx))
        )
        return successful, failed

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
