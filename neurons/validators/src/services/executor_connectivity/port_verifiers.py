import asyncio
import logging
import uuid

import aiohttp

from core.utils import _m, get_extra_info
from services.executor_connectivity.container_runner import ContainerRunner
from services.executor_connectivity.models import PortPair
from services.executor_connectivity.netcat_script import NetcatScript
from services.executor_connectivity.port_tester import PortTester

logger = logging.getLogger(__name__)


class BatchVerifier:
    """Verifies ports using --network=host in batches."""

    def __init__(self, port_tester: PortTester, runner: ContainerRunner):
        self.port_tester = port_tester
        self.runner = runner

    async def verify(
        self,
        ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Verify ports with retries."""
        ctx = log_ctx or {}
        max_attempts = 2
        timeout_sec = 60

        for attempt in range(1, max_attempts + 1):
            token = uuid.uuid4().hex
            container_name = f"port_test_{token[:8]}"

            logger.info(
                _m(
                    f"testing {len(ports)} ports (attempt {attempt}/{max_attempts}, timeout={timeout_sec}s)",
                    extra=get_extra_info(ctx),
                )
            )

            try:
                successful, failed = await asyncio.wait_for(
                    self._attempt(ports, token, container_name, ssh_client, host, ctx),
                    timeout=timeout_sec
                )

                if successful:
                    logger.info(
                        _m(f"complete: {len(successful)}/{len(ports)} verified", extra=get_extra_info(ctx))
                    )
                    return successful, failed

                if attempt < max_attempts:
                    logger.warning(
                        _m(f"attempt {attempt} failed, retrying in 2s", extra=get_extra_info(ctx))
                    )
                    await asyncio.sleep(2)

            except asyncio.TimeoutError:
                logger.error(
                    _m(f"attempt {attempt} timed out after {timeout_sec}s", extra=get_extra_info(ctx))
                )
                await self.runner.cleanup(ssh_client, container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(
                    _m(f"attempt {attempt} failed: {e}", extra=get_extra_info(ctx)),
                    exc_info=True,
                )
                await self.runner.cleanup(ssh_client, container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

        logger.error(_m(f"all {max_attempts} attempts failed", extra=get_extra_info(ctx)))
        return [], ports

    async def _attempt(
        self,
        ports: list[PortPair],
        token: str,
        name: str,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Single verification attempt."""
        ctx = log_ctx or {}
        script = NetcatScript.batch(ports, token, 0)
        start_result = await self.runner.run(ssh_client, name, script, "host", 60)
        if not start_result.ok:
            logger.warning(
                _m(
                    f"batch start failed: status={start_result.status} logs={start_result.logs}",
                    extra=get_extra_info(ctx),
                )
            )
            return [], ports

        try:
            async with aiohttp.ClientSession() as session:
                successful, failed = await self.port_tester.test_many(session, host, ports, token)
                logger.info(
                    _m(f"progress: {len(successful)}/{len(ports)} verified", extra=get_extra_info(ctx))
                )
        finally:
            await self.runner.cleanup(ssh_client, name)

        return successful, failed


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
        ctx = log_ctx or {}
        ports_to_test = ports[:max_ports]
        logger.info(
            _m(f"fallback: testing {len(ports_to_test)} ports sequentially", extra=get_extra_info(ctx))
        )

        successful, failed = [], []

        async with aiohttp.ClientSession() as session:
            for idx, port in enumerate(ports_to_test, 1):
                token = uuid.uuid4().hex
                name = f"port_test_seq_{token[:8]}"
                script = NetcatScript.single(port, token)
                network_flag = f"-p {port.internal}:{port.internal}"

                started = False
                try:
                    start_result = await self.runner.run(ssh_client, name, script, network_flag, 10)
                    if not start_result.ok:
                        logger.warning(
                            _m(
                                f"fallback: port {port.internal} failed to start: "
                                f"status={start_result.status} logs={start_result.logs}",
                                extra=get_extra_info(ctx),
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
                                extra=get_extra_info(ctx),
                            )
                        )
                        successful.append(port)
                    else:
                        failed.append(port)

                except Exception as e:
                    logger.error(
                        _m(
                            f"fallback: error testing port {port.internal}: {str(e)[:100]}",
                            extra=get_extra_info(ctx),
                        )
                    )
                    failed.append(port)

                finally:
                    if started:
                        try:
                            await self.runner.cleanup(ssh_client, name)
                        except Exception as e:
                            logger.warning(
                                _m(f"fallback: cleanup failed: {str(e)[:50]}", extra=get_extra_info(ctx))
                            )

                    if idx < len(ports_to_test):
                        await asyncio.sleep(0.5)

        logger.info(
            _m(f"fallback: {len(successful)}/{len(ports_to_test)} verified", extra=get_extra_info(ctx))
        )
        return successful, failed
