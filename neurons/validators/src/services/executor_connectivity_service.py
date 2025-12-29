import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from uuid import UUID

import aiohttp
import asyncssh
from asyncssh import SSHClientConnection
from datura.requests.miner_requests import ExecutorSSHInfo
from pydantic import BaseModel

from core.config import settings
from core.utils import _m
from daos.port_mapping_dao import PortMappingDao
from models.port_mapping import PortMapping
from services.const import (
    BATCH_PORT_CONCURRENCY,
    BATCH_PORT_VERIFICATION_SIZE,
    DOCKER_DIND_IMAGE,
    PREFERRED_POD_PORTS,
)
from services.redis_service import RedisService

logger = logging.getLogger(__name__)


# ============================================================================
# VALUE OBJECTS
# ============================================================================

@dataclass(frozen=True)
class PortPair:
    """Immutable port mapping."""
    internal: int
    external: int


class DockerConnectionCheckResult(BaseModel):
    success: bool
    log_text: str | None = None
    sysbox_runtime: bool
    verified_port_count: int = 0


# ============================================================================
# DOCKER COMMAND BUILDER - All docker commands in one place
# ============================================================================

class DockerCommand:
    """Builds safe docker command strings."""

    @staticmethod
    def run_alpine(name: str, script: str, network_mode: str, timeout: int) -> str:
        """Build docker run command for Alpine netcat container."""
        heredoc = "__NC_EOF__"
        network_flag = f"--network={network_mode}" if network_mode == "host" else network_mode
        return (
            f"/usr/bin/docker run -d --rm --name {name} "
            f"{network_flag} docker.io/library/alpine:3.19 sh -c "
            f"'cat << \"{heredoc}\" > /tmp/nc.sh\n{script}\n{heredoc}\n"
            f"timeout {timeout} sh /tmp/nc.sh'"
        )

    @staticmethod
    def run_dind(name: str, port: int, public_key: str, sysbox: bool) -> str:
        """Build docker run command for DinD container."""
        runtime = "--runtime=sysbox-runc " if sysbox else ""
        ssh_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
        return f"/usr/bin/docker run -d {runtime} --name {name} --gpus all -p {port}:22 {DOCKER_DIND_IMAGE} {ssh_cmd}"

    @staticmethod
    def remove(name: str) -> str:
        """Build docker rm command."""
        return f"/usr/bin/docker rm -f {name} 2>/dev/null || true"

    @staticmethod
    def ps_filter(name_pattern: str) -> str:
        """Build docker ps command with filter."""
        return f'/usr/bin/docker ps -a --filter "name={name_pattern}" --format "{{{{.Names}}}}"'

    @staticmethod
    def inspect_status(container_id: str) -> str:
        """Build docker inspect command for status."""
        return f"/usr/bin/docker ps -a --filter id={container_id} --format '{{{{.Status}}}}|||{{{{.State}}}}' 2>&1"

    @staticmethod
    def logs(container_id: str) -> str:
        """Build docker logs command."""
        return f"/usr/bin/docker logs {container_id} 2>&1 | head -20"

    @staticmethod
    def inspect_exit_code(container_id: str) -> str:
        """Build docker inspect for exit code."""
        return f"/usr/bin/docker inspect {container_id} --format '{{{{.State.ExitCode}}}}' 2>&1"

    @staticmethod
    def volume_prune() -> str:
        """Build docker volume prune command."""
        return "/usr/bin/docker volume prune -af"


# ============================================================================
# NETCAT SCRIPT BUILDER
# ============================================================================

class NetcatScript:
    """Builds netcat test scripts."""

    @staticmethod
    def batch(ports: list[PortPair], token: str, batch_idx: int) -> str:
        """Build script for batch port testing."""
        port_list = ' '.join([str(p.internal) for p in ports])
        return f'''
echo "Batch {batch_idx}: starting ports {port_list}" >&2
for port in {port_list}; do
    (
        echo "Binding port $port" >&2
        body="{token}:$port"
        printf "HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: ${{#body}}\\r\\nConnection: close\\r\\n\\r\\n$body" | nc -l -p $port
        nc_status=$?
        [ $nc_status -eq 0 ] && echo "Port $port served" >&2 || echo "Port $port failed (nc exit $nc_status)" >&2
    ) &
done
wait
echo "Batch {batch_idx}: completed" >&2
sleep 60
'''

    @staticmethod
    def single(port: PortPair, token: str) -> str:
        """Build script for single port testing."""
        return f'''
echo "Testing port {port.internal}" >&2
body="{token}:{port.internal}"
printf "HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: ${{#body}}\\r\\nConnection: close\\r\\n\\r\\n$body" | nc -l -p {port.internal}
'''


# ============================================================================
# ALPINE CONTAINER MANAGER
# ============================================================================

class AlpineContainer:
    """Manages Alpine netcat test containers."""

    def __init__(self, ssh: SSHClientConnection, extra: dict):
        self.ssh = ssh
        self.extra = extra

    async def start_and_verify(self, name: str, script: str, network_mode: str, timeout: int) -> bool:
        """Start Alpine container with netcat script and verify it's running."""
        cmd = DockerCommand.run_alpine(name, script, network_mode, timeout)
        logger.debug(_m(f"starting container: {name}", self.extra))

        result = await self.ssh.run(cmd)
        if result.exit_status != 0:
            logger.error(_m(f"container start failed: {result.stderr.strip()}", self.extra))
            return False

        container_id = result.stdout.strip()
        logger.debug(_m(f"container started: {container_id[:12]}", self.extra))
        await asyncio.sleep(1.0)

        # Verify running
        result = await self.ssh.run(DockerCommand.inspect_status(container_id))
        if "Up" in result.stdout:
            logger.info(_m(f"container running: {result.stdout.strip()}", self.extra))
            return True

        # Failed - get diagnostics
        logger.error(_m(f"container not running: {result.stdout.strip()}", self.extra))
        logs_result = await self.ssh.run(DockerCommand.logs(container_id))
        logger.error(_m(f"container failed logs={logs_result.stdout.strip()[:500]}", self.extra))
        return False

    async def cleanup(self, name: str):
        """Remove container."""
        await self.ssh.run(DockerCommand.remove(name))


# ============================================================================
# PORT TESTER - HTTP Testing
# ============================================================================

class PortTester:
    """Tests port connectivity via HTTP."""

    def __init__(self, session: aiohttp.ClientSession, host: str, extra: dict):
        self.session = session
        self.host = host
        self.extra = extra

    async def test_one(self, port: PortPair, token: str) -> bool:
        """Test single port."""
        url = f"http://{self.host}:{port.external}/"
        expected = f"{token}:{port.internal}"

        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                text = await resp.text()
                if text.strip() == expected:
                    logger.debug(_m(f"port {port.internal} ok", self.extra))
                    return True
                logger.warning(_m(f"port {port.internal} wrong response: {text[:50]}", self.extra))
                return False
        except Exception as e:
            logger.warning(_m(f"port {port.internal} failed: {str(e)[:100]}", self.extra))
            return False

    async def test_many(self, ports: list[PortPair], token: str) -> tuple[list[PortPair], list[PortPair]]:
        """Test multiple ports concurrently."""
        results = await asyncio.gather(*[self.test_one(p, token) for p in ports])
        successful = [p for p, ok in zip(ports, results) if ok]
        failed = [p for p, ok in zip(ports, results) if not ok]
        return successful, failed


# ============================================================================
# PORT SELECTOR
# ============================================================================

class PortSelector:
    """Selects which ports to verify."""

    @staticmethod
    def select(executor_info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select available ports with preference for common ports."""
        if executor_info.port_mappings:
            return PortSelector._from_mappings(executor_info, size, rented)
        return PortSelector._from_range(executor_info, size, rented)

    @staticmethod
    def _from_mappings(info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select from existing mappings."""
        mappings = json.loads(info.port_mappings)
        available = [
            PortPair(i, e) for i, e in mappings
            if i != info.ssh_port and e != info.ssh_port and e not in rented
        ]
        preferred = [p for p in available if p.internal in PREFERRED_POD_PORTS or p.external in PREFERRED_POD_PORTS]
        remaining = [p for p in available if p not in preferred]

        result = preferred[:size]
        if len(result) < size and remaining:
            result.extend(random.sample(remaining, min(size - len(result), len(remaining))))
        return result[:size]

    @staticmethod
    def _from_range(info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select from port range."""
        if info.port_range:
            if "-" in info.port_range:
                min_p, max_p = map(int, (part.strip() for part in info.port_range.split("-")))
                ports = list(range(min_p, max_p + 1))
            else:
                ports = list(map(int, (part.strip() for part in info.port_range.split(","))))
        else:
            ports = list(range(20000, 65535))

        ports = [p for p in ports if p != info.ssh_port and p not in rented]
        if not ports:
            return []

        preferred = [p for p in PREFERRED_POD_PORTS if p in ports]
        remaining = [p for p in ports if p not in PREFERRED_POD_PORTS]

        selected = preferred[:size]
        if len(selected) < size and remaining:
            selected.extend(random.sample(remaining, min(size - len(selected), len(remaining))))
        return [PortPair(p, p) for p in selected[:size]]


# ============================================================================
# BATCH VERIFIER - Host Network Mode
# ============================================================================

class BatchVerifier:
    """Verifies ports using --network=host in batches."""

    def __init__(self, alpine: AlpineContainer, host: str, extra: dict):
        self.alpine = alpine
        self.host = host
        self.extra = extra

    async def verify(self, ports: list[PortPair]) -> tuple[list[PortPair], list[PortPair]]:
        """Verify ports with retries."""
        max_attempts = 2
        batches = (len(ports) + BATCH_PORT_CONCURRENCY - 1) // BATCH_PORT_CONCURRENCY
        timeout_sec = 60 * max(1, batches)

        for attempt in range(1, max_attempts + 1):
            token = uuid.uuid4().hex
            container_name = f"port_test_{token[:8]}"

            logger.info(_m(
                f"testing {len(ports)} ports (attempt {attempt}/{max_attempts}, timeout={timeout_sec}s)",
                self.extra
            ))

            try:
                successful, failed = await asyncio.wait_for(
                    self._attempt(ports, token, container_name),
                    timeout=timeout_sec
                )

                if successful:
                    logger.info(_m(f"complete: {len(successful)}/{len(ports)} verified", self.extra))
                    return successful, failed

                if attempt < max_attempts:
                    logger.warning(_m(f"attempt {attempt} failed, retrying in 2s", self.extra))
                    await asyncio.sleep(2)

            except asyncio.TimeoutError:
                logger.error(_m(f"attempt {attempt} timed out after {timeout_sec}s", self.extra))
                await self.alpine.cleanup(container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(_m(f"attempt {attempt} failed: {e}", self.extra), exc_info=True)
                await self.alpine.cleanup(container_name)
                if attempt < max_attempts:
                    await asyncio.sleep(2)

        logger.error(_m(f"all {max_attempts} attempts failed", self.extra))
        return [], ports

    async def _attempt(self, ports: list[PortPair], token: str, base_name: str) -> tuple[list[PortPair], list[PortPair]]:
        """Single verification attempt."""
        successful, failed = [], []

        for batch_idx in range(0, len(ports), BATCH_PORT_CONCURRENCY):
            batch = ports[batch_idx:batch_idx + BATCH_PORT_CONCURRENCY]
            batch_num = batch_idx // BATCH_PORT_CONCURRENCY
            container_name = f"{base_name}_b{batch_num}"

            script = NetcatScript.batch(batch, token, batch_num)
            if not await self.alpine.start_and_verify(container_name, script, "host", 60):
                failed.extend(batch)
                continue

            try:
                async with aiohttp.ClientSession() as session:
                    tester = PortTester(session, self.host, self.extra)
                    batch_ok, batch_fail = await tester.test_many(batch, token)
                    successful.extend(batch_ok)
                    failed.extend(batch_fail)
                    logger.info(_m(f"progress: {len(successful)}/{len(ports)} verified", self.extra))
            finally:
                await self.alpine.cleanup(container_name)

        return successful, failed


# ============================================================================
# FALLBACK VERIFIER - Publish Mode
# ============================================================================

class FallbackVerifier:
    """Verifies ports sequentially using -p publish."""

    def __init__(self, alpine: AlpineContainer, host: str, extra: dict):
        self.alpine = alpine
        self.host = host
        self.extra = extra

    async def verify(self, ports: list[PortPair], max_ports: int = 10) -> tuple[list[PortPair], list[PortPair]]:
        """Test ports one by one."""
        ports_to_test = ports[:max_ports]
        logger.info(_m(f"fallback: testing {len(ports_to_test)} ports sequentially", self.extra))

        successful, failed = [], []

        async with aiohttp.ClientSession() as session:
            tester = PortTester(session, self.host, self.extra)

            for idx, port in enumerate(ports_to_test, 1):
                token = uuid.uuid4().hex
                name = f"port_test_seq_{token[:8]}"
                script = NetcatScript.single(port, token)
                network_flag = f"-p {port.internal}:{port.internal}"

                started = False
                try:
                    started = await self.alpine.start_and_verify(name, script, network_flag, 10)
                    if not started:
                        logger.warning(_m(f"fallback: port {port.internal} failed to start", self.extra))
                        failed.append(port)
                        continue

                    await asyncio.sleep(1.0)
                    if await tester.test_one(port, token):
                        logger.info(_m(f"fallback: port {port.internal} ok ({idx}/{len(ports_to_test)})", self.extra))
                        successful.append(port)
                    else:
                        failed.append(port)

                except Exception as e:
                    logger.error(_m(f"fallback: error testing port {port.internal}: {str(e)[:100]}", self.extra))
                    failed.append(port)

                finally:
                    if started:
                        try:
                            await self.alpine.cleanup(name)
                        except Exception as e:
                            logger.warning(_m(f"fallback: cleanup failed: {str(e)[:50]}", self.extra))

                    if idx < len(ports_to_test):
                        await asyncio.sleep(0.5)

        logger.info(_m(f"fallback: {len(successful)}/{len(ports_to_test)} verified", self.extra))
        return successful, failed


# ============================================================================
# DIND VERIFIER
# ============================================================================

class DindVerifier:
    """Verifies Docker-in-Docker capability."""

    def __init__(self, ssh: SSHClientConnection, host: str, extra: dict):
        self.ssh = ssh
        self.host = host
        self.extra = extra

    async def verify(
        self, port: PortPair, miner_hotkey: str, private_key: str, public_key: str, sysbox: bool
    ) -> DockerConnectionCheckResult:
        """Verify DinD on port."""
        extra = {**self.extra, "internal_port": port.internal, "external_port": port.external}
        name = f"container_{miner_hotkey}_{port.external}"

        try:
            logger.info(_m(f"dind: start port={port.internal}", extra))

            cmd = DockerCommand.run_dind(name, port.internal, public_key, sysbox)
            logger.debug(_m(f"run: {cmd[:100]}...", extra))

            result = await self.ssh.run(cmd)
            if result.exit_status != 0:
                logger.error(_m(f"dind creation failed: {result.stderr.strip()} port={port.internal}", extra))
                await self.ssh.run(DockerCommand.remove(name))
                return DockerConnectionCheckResult(
                    success=False, log_text=f"dind: check failed port={port.internal}", sysbox_runtime=sysbox
                )

            logger.info(_m("dind: docker created", extra))
            await asyncio.sleep(5)

            # Test SSH
            pkey = asyncssh.import_private_key(private_key)
            async with asyncssh.connect(
                host=self.host, port=port.external, username="root", client_keys=[pkey], known_hosts=None
            ) as ssh:
                logger.info(_m("dind: ssh connected", extra))

                # Test sysbox
                if sysbox:
                    result = await ssh.run("docker pull hello-world")
                    sysbox_ok = result.exit_status == 0
                    logger.info(_m(f"dind: sysbox {'ok' if sysbox_ok else 'fail'}", extra))
                    if not sysbox_ok:
                        logger.debug(_m(f"sysbox failed: {result.stderr.strip()}", extra))
                        sysbox = False

            await self.ssh.run(DockerCommand.remove(name))
            logger.info(_m(f"dind: check ok port={port.internal}", extra))

            return DockerConnectionCheckResult(
                success=True, log_text=f"dind: check ok port={port.internal}", sysbox_runtime=sysbox
            )

        except Exception as e:
            logger.error(_m(f"dind failed: {str(e)} port={port.internal}", extra), exc_info=True)
            await self.ssh.run(DockerCommand.remove(name))
            return DockerConnectionCheckResult(
                success=False, log_text=f"dind: check failed port={port.internal}", sysbox_runtime=sysbox
            )


# ============================================================================
# MAIN SERVICE - Thin Orchestrator
# ============================================================================

class ExecutorConnectivityService:
    """Orchestrates port verification workflow."""

    def __init__(self, redis_service: "RedisService", port_mapping_dao: PortMappingDao):
        self.redis = redis_service
        self.dao = port_mapping_dao

    async def verify_ports(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        sysbox_runtime: bool = False,
        rented_ports: list[int] | None = None,
        rented_pod_names: list[str] | None = None,
    ) -> DockerConnectionCheckResult:
        """Verify executor port connectivity and DinD capability."""
        extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "ssh_username": executor_info.ssh_username,
            "ssh_port": executor_info.ssh_port,
            "version": settings.VERSION,
            "sysbox_runtime": sysbox_runtime,
        }

        try:
            t1 = time.monotonic()

            # Setup
            alpine = AlpineContainer(ssh_client, extra)

            # Cleanup old containers
            await self._cleanup_old_containers(ssh_client, executor_info, extra)

            # Select ports - use rented_ports from parameter if provided
            if rented_ports is not None:
                rented = set(rented_ports)
            else:
                rented = await self.dao.get_busy_external_ports(UUID(executor_info.uuid))
            ports = PortSelector.select(executor_info, BATCH_PORT_VERIFICATION_SIZE, rented)

            if not ports:
                return DockerConnectionCheckResult(
                    success=False,
                    log_text="No port available for docker container",
                    sysbox_runtime=sysbox_runtime,
                    verified_port_count=0
                )

            logger.debug(_m(f"checking {len(ports)} port mappings", extra))

            # Verify ports (batch with fallback)
            batch_verifier = BatchVerifier(alpine, executor_info.address, extra)
            successful, failed = await batch_verifier.verify(ports)

            if not successful:
                logger.warning(_m("batch verification failed, trying fallback", extra))
                fallback_verifier = FallbackVerifier(alpine, executor_info.address, extra)
                successful, failed = await fallback_verifier.verify(ports, max_ports=10)

            # Verify DinD
            dind_port = successful.pop(0) if successful else random.choice(ports)
            dind_verifier = DindVerifier(ssh_client, executor_info.address, extra)
            dind_result = await dind_verifier.verify(
                dind_port, miner_hotkey, private_key, public_key, sysbox_runtime
            )

            if dind_result.success:
                successful.append(dind_port)
                sysbox_runtime = dind_result.sysbox_runtime
            else:
                failed.append(dind_port)
                sysbox_runtime = False

            if not successful:
                return DockerConnectionCheckResult(
                    success=False,
                    log_text="No working ports found",
                    sysbox_runtime=sysbox_runtime,
                    verified_port_count=0
                )

            # Save results
            await self._save_results(executor_info, miner_hotkey, successful, failed, extra)

            # Build message
            total = len(successful) + len(failed)
            pct = (len(successful) / total * 100) if total > 0 else 0
            dind_status = "ok" if dind_result.success else "failed"
            batch_count = len(successful) - (1 if dind_result.success else 0)
            batch_status = "ok" if batch_count > 0 else "failed"

            ok_sample = sorted([p.internal for p in successful])[:5]
            fail_sample = sorted([p.internal for p in failed])[:5]

            msg = (
                f"verification complete total_time={time.monotonic() - t1:.2f}s {pct:.0f}% available, "
                f"dind={dind_status} batch={batch_status} ok={len(successful)}{ok_sample}"
            )
            if failed:
                msg += f" fail={len(failed)}{fail_sample}"

            logger.info(_m(msg, extra))
            return DockerConnectionCheckResult(
                success=True,
                log_text=msg,
                sysbox_runtime=sysbox_runtime,
                verified_port_count=len(successful)
            )
        except Exception as e:
            logger.error(_m(f"verification failed: {str(e)} executor={executor_info.address}", extra), exc_info=True)
            return DockerConnectionCheckResult(
                success=False,
                log_text=f"Verification failed: {str(e)}",
                sysbox_runtime=sysbox_runtime,
                verified_port_count=0
            )

    async def _cleanup_old_containers(self, ssh: SSHClientConnection, info: ExecutorSSHInfo, extra: dict):
        """Remove old containers not in active pods."""
        try:
            pod_names = []
            rented = await self.redis.get_rented_machine(info)
            if rented:
                pod_names = [p.get("name", "") for p in rented.get("containers", [])]

            result = await ssh.run(DockerCommand.ps_filter("^/container_"))
            if not result.stdout.strip():
                return

            names = [n for n in result.stdout.strip().split("\n") if n and n not in pod_names]
            if names:
                logger.info(_m(f"cleanup: found {len(names)} containers", extra))
                await ssh.run(f"/usr/bin/docker rm {' '.join(names)} -f")
                await ssh.run(DockerCommand.volume_prune())
                logger.info(_m(f"cleanup: removed {len(names)} containers", extra))
        except Exception as e:
            logger.error(_m(f"cleanup failed: {e}", extra), exc_info=True)

    async def _save_results(
        self, info: ExecutorSSHInfo, hotkey: str, successful: list[PortPair], failed: list[PortPair], extra: dict
    ):
        """Save verification results to database."""
        try:
            records = [
                PortMapping(
                    miner_hotkey=hotkey,
                    executor_id=UUID(info.uuid),
                    internal_port=p.internal,
                    external_port=p.external,
                    is_successful=True,
                )
                for p in successful
            ]
            records.extend([
                PortMapping(
                    miner_hotkey=hotkey,
                    executor_id=UUID(info.uuid),
                    internal_port=p.internal,
                    external_port=p.external,
                    is_successful=False,
                )
                for p in failed
            ])

            if records:
                await self.dao.upsert_port_results(records)
                cleaned = await self.dao.clean_ports(records[0].executor_id, 24 * 60)
                logger.info(_m(f"saved {len(records)} ports to db, {cleaned=}", extra))
        except Exception as e:
            logger.error(_m(f"save to db failed: {e}", extra), exc_info=True)
