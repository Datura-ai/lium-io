import asyncio
import logging
import uuid

import aiohttp

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
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Verify ports with retries."""
        max_attempts = 2
        timeout_sec = 60

        for attempt in range(1, max_attempts + 1):
            token = uuid.uuid4().hex
            container_name = f"port_test_{token[:8]}"

            logger.info(
                "testing %s ports (attempt %s/%s, timeout=%ss)",
                len(ports),
                attempt,
                max_attempts,
                timeout_sec,
            )

            try:
                successful, failed = await asyncio.wait_for(
                    self._attempt(ports, token, container_name, ssh_client, host),
                    timeout=timeout_sec
                )

                if successful:
                    logger.info("complete: %s/%s verified", len(successful), len(ports))
                    return successful, failed

                if attempt < max_attempts:
                    logger.warning("attempt %s failed, retrying in 2s", attempt)
                    await asyncio.sleep(2)

            except asyncio.TimeoutError:
                logger.error("attempt %s timed out after %ss", attempt, timeout_sec)
                await self.runner.cleanup(ssh_client, container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error("attempt %s failed: %s", attempt, e, exc_info=True)
                await self.runner.cleanup(ssh_client, container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

        logger.error("all %s attempts failed", max_attempts)
        return [], ports

    async def _attempt(
        self,
        ports: list[PortPair],
        token: str,
        name: str,
        ssh_client,
        host: str,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Single verification attempt."""
        script = NetcatScript.batch(ports, token, 0)
        start_result = await self.runner.run(ssh_client, name, script, "host", 60)
        if not start_result.ok:
            logger.warning(
                "batch start failed: status=%s logs=%s",
                start_result.status,
                start_result.logs,
            )
            return [], ports

        try:
            async with aiohttp.ClientSession() as session:
                successful, failed = await self.port_tester.test_many(session, host, ports, token)
                logger.info("progress: %s/%s verified", len(successful), len(ports))
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
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Test ports one by one."""
        ports_to_test = ports[:max_ports]
        logger.info("fallback: testing %s ports sequentially", len(ports_to_test))

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
                            "fallback: port %s failed to start: status=%s logs=%s",
                            port.internal,
                            start_result.status,
                            start_result.logs,
                        )
                        failed.append(port)
                        continue

                    started = True
                    await asyncio.sleep(1.0)
                    if await self.port_tester.test_one(session, host, port, token):
                        logger.info(
                            "fallback: port %s ok (%s/%s)",
                            port.internal,
                            idx,
                            len(ports_to_test),
                        )
                        successful.append(port)
                    else:
                        failed.append(port)

                except Exception as e:
                    logger.error(
                        "fallback: error testing port %s: %s",
                        port.internal,
                        str(e)[:100],
                    )
                    failed.append(port)

                finally:
                    if started:
                        try:
                            await self.runner.cleanup(ssh_client, name)
                        except Exception as e:
                            logger.warning("fallback: cleanup failed: %s", str(e)[:50])

                    if idx < len(ports_to_test):
                        await asyncio.sleep(0.5)

        logger.info("fallback: %s/%s verified", len(successful), len(ports_to_test))
        return successful, failed
