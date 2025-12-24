import asyncio
import json
import logging
import random
import time
import uuid
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
    BATCH_PORT_VERIFICATION_SIZE,
    DOCKER_DIND_IMAGE,
    PREFERRED_POD_PORTS,
    BATCH_PORT_CONCURRENCY,
)
from services.redis_service import RedisService

logger = logging.getLogger(__name__)


class DockerConnectionCheckResult(BaseModel):
    success: bool
    log_text: str | None = None
    sysbox_runtime: bool


class ExecutorConnectivityService:
    def __init__(self, redis_service: "RedisService", port_mapping_dao: PortMappingDao):
        self.redis_service = redis_service
        self.port_mapping_dao = port_mapping_dao

    async def verify_ports(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        sysbox_runtime: bool = False,
    ) -> DockerConnectionCheckResult:
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

        """Verify multiple ports concurrently."""
        try:
            t1 = time.monotonic()
            await self.cleanup_docker_containers(ssh_client, executor_info, extra)
            rented_external_ports = await self.port_mapping_dao.get_busy_external_ports(UUID(executor_info.uuid))
            port_maps = self.get_available_port_maps(executor_info, BATCH_PORT_VERIFICATION_SIZE, rented_external_ports)
            if not port_maps:
                return DockerConnectionCheckResult(
                    success=False, log_text="No port available for docker container", sysbox_runtime=sysbox_runtime,
                )

            logger.debug(_m(f"checking {len(port_maps)} port mappings", extra))

            successful_ports, failed_ports = await self.verify_ports_bulk(ssh_client, port_maps, executor_info, extra)
            dind_port = successful_ports.pop(0) if successful_ports else random.choice(port_maps)
            dind_result = await self.verify_port_dind(
                ssh_client,
                miner_hotkey,
                executor_info,
                private_key,
                public_key,
                dind_port[0],
                dind_port[1],
                sysbox_runtime,
                extra,
            )

            # Add dind port pair
            if dind_result.success:
                successful_ports.append(dind_port)
                sysbox_runtime = dind_result.sysbox_runtime
            else:
                failed_ports.append(dind_port)
                sysbox_runtime = False

            # Calculate statistics
            total_checked = len(successful_ports) + len(failed_ports)
            success_percentage = (len(successful_ports) / total_checked * 100) if total_checked > 0 else 0

            # Log verification summary
            dind_status = "ok" if dind_result.success else "failed"
            batch_successful_count = len(successful_ports) - (1 if dind_result.success else 0)
            batch_status = "ok" if batch_successful_count > 0 else "failed"

            if not successful_ports:
                failure_msg = "No working ports found"
                return DockerConnectionCheckResult(success=False, log_text=failure_msg, sysbox_runtime=sysbox_runtime,)

            # Save successful ports
            await self.save_to_db(executor_info, miner_hotkey, successful_ports, failed_ports, extra)

            # Create detailed success message
            successful_internal_ports = [port_pair[0] for port_pair in successful_ports]
            failed_internal_ports = [port_pair[0] for port_pair in failed_ports]

            success_sample = sorted(successful_internal_ports)[:5]
            failed_sample = sorted(failed_internal_ports)[:5]

            total_time = time.monotonic() - t1
            success_msg = f"verification complete {total_time=:.2f}s {success_percentage:.0f}% available, dind={dind_status} batch={batch_status} ok={len(successful_ports)}{success_sample}"
            if failed_ports:
                success_msg += f" fail={len(failed_ports)}{failed_sample}"
            logger.info(_m(success_msg, extra))

            return DockerConnectionCheckResult(success=True, log_text=success_msg, sysbox_runtime=sysbox_runtime,)
        except Exception as e:
            logger.error(_m(f"verification failed: {str(e)} executor={executor_info.address}", extra), exc_info=True)

            return DockerConnectionCheckResult(
                success=False, log_text=f"Verification failed: {str(e)}", sysbox_runtime=sysbox_runtime,
            )

    def _build_netcat_script(self, port_maps: list[tuple[int, int]], token: str) -> str:
        """Build bash script that uses netcat to listen on multiple ports with unique token."""
        # Split ports into batches
        port_batches = []
        for i in range(0, len(port_maps), BATCH_PORT_CONCURRENCY):
            batch = port_maps[i:i + BATCH_PORT_CONCURRENCY]
            batch_ports = ' '.join([str(internal_port) for internal_port, _ in batch])
            port_batches.append(batch_ports)

        # Build netcat commands for each batch
        batch_commands = []
        for idx, batch_ports in enumerate(port_batches):
            batch_cmd = f'''
echo "Batch {idx}: starting ports {batch_ports}" >&2
for port in {batch_ports}; do
    (
        echo "Binding port $port" >&2
        body="{token}:$port"
        printf "HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: ${{#body}}\\r\\nConnection: close\\r\\n\\r\\n$body" | nc -l -p $port
        echo "Port $port served" >&2
    ) &
done
wait
echo "Batch {idx}: completed" >&2
'''
            batch_commands.append(batch_cmd)

        return '\n'.join(batch_commands) + '\necho "All batches completed" >&2'

    async def _start_port_test_container(
        self, ssh_client: SSHClientConnection, container_name: str, nc_script: str, extra: dict
    ) -> bool:
        """Start Alpine container with netcat script and verify it's running.

        Returns:
            True if container started successfully, False otherwise
        """
        command = (
            f"/usr/bin/docker run -d --rm --name {container_name} "
            f"--network=host docker.io/library/alpine:3.19 sh -c '{nc_script}'"
        )

        logger.debug(_m(f"starting container: {container_name}", extra))
        logger.debug(_m(f"nc_script: {nc_script[:200]}...", extra))

        result = await ssh_client.run(command)
        if result.exit_status != 0:
            logger.error(_m(f"container start failed: {result.stderr.strip()}", extra))
            return False

        # Give container time to start
        await asyncio.sleep(0.5)

        # Verify container is still running
        check_cmd = f"/usr/bin/docker ps --filter name={container_name} --format '{{{{.Status}}}}'"
        check_result = await ssh_client.run(check_cmd)

        if check_result.stdout.strip():
            logger.info(_m(f"container status: {check_result.stdout.strip()}", extra))
            return True

        # Container not running - get logs
        logger.error(_m(f"container not running! checking logs...", extra))
        logs_cmd = f"/usr/bin/docker logs {container_name} 2>&1 || echo 'no logs'"
        logs_result = await ssh_client.run(logs_cmd)
        logger.error(_m(f"container logs: {logs_result.stdout.strip()}", extra))
        return False

    async def _test_ports_in_batches(
        self,
        port_maps: list[tuple[int, int]],
        executor_host: str,
        token: str,
        extra: dict,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Test all ports in batches and return results.

        Returns:
            Tuple of (successful_ports, failed_ports)
        """
        successful_ports = []
        failed_ports = []

        async with aiohttp.ClientSession() as session:
            for batch_idx in range(0, len(port_maps), BATCH_PORT_CONCURRENCY):
                batch = port_maps[batch_idx:batch_idx + BATCH_PORT_CONCURRENCY]

                # Small delay between batches to let container bind ports
                if batch_idx > 0:
                    await asyncio.sleep(0.1)

                logger.debug(_m(f"testing batch {batch_idx // BATCH_PORT_CONCURRENCY}: {len(batch)} ports", extra))

                # Test all ports in this batch concurrently
                tasks = [
                    self._test_single_port_with_session(session, executor_host, int_port, ext_port, token, extra)
                    for int_port, ext_port in batch
                ]
                results = await asyncio.gather(*tasks)

                # Separate successful and failed ports
                for (int_port, ext_port), success in zip(batch, results):
                    if success:
                        successful_ports.append((int_port, ext_port))
                    else:
                        failed_ports.append((int_port, ext_port))

                # Log progress
                if (batch_idx + len(batch)) % BATCH_PORT_CONCURRENCY == 0 or (batch_idx + len(batch)) >= len(port_maps):
                    logger.info(_m(f"progress: {len(successful_ports)}/{len(port_maps)} verified", extra))

        return successful_ports, failed_ports

    async def _cleanup_port_test_container(self, ssh_client: SSHClientConnection, container_name: str):
        """Remove port test container."""
        cleanup_cmd = f"/usr/bin/docker rm -f {container_name} 2>/dev/null || true"
        await ssh_client.run(cleanup_cmd)

    async def _verify_ports_bulk_attempt(
        self,
        ssh_client: SSHClientConnection,
        port_maps: list[tuple[int, int]],
        executor_host: str,
        token: str,
        container_name: str,
        extra: dict,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Single attempt to verify ports. Returns (successful_ports, failed_ports)."""
        # Build netcat script
        nc_script = self._build_netcat_script(port_maps, token)

        # Start container
        if not await self._start_port_test_container(ssh_client, container_name, nc_script, extra):
            return [], port_maps

        # Test all ports
        successful_ports, failed_ports = await self._test_ports_in_batches(
            port_maps, executor_host, token, extra
        )

        # Cleanup
        await self._cleanup_port_test_container(ssh_client, container_name)

        return successful_ports, failed_ports

    async def verify_ports_bulk(
        self,
        ssh_client: SSHClientConnection,
        port_maps: list[tuple[int, int]],
        executor_info: ExecutorSSHInfo,
        extra: dict = {},
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Test ports using Alpine container with netcat and unique UUID token.

        This method prevents conflicts when multiple validators test the same executor
        by using a unique token per verification session. Retries once if first attempt fails.
        Has a 60-second timeout per attempt to prevent hanging.
        """
        if not port_maps:
            return [], []

        max_attempts = 2
        timeout_seconds = 60  # 1 minute timeout per attempt

        for attempt in range(1, max_attempts + 1):
            # Generate unique token and container name for each attempt
            token = uuid.uuid4().hex
            container_name = f"port_test_{token[:8]}"

            logger.info(_m(
                f"testing {len(port_maps)} ports in batches of {BATCH_PORT_CONCURRENCY} with token {token[:8]} (attempt {attempt}/{max_attempts}, timeout={timeout_seconds}s)",
                extra
            ))

            try:
                # Wrap in timeout to prevent hanging
                successful_ports, failed_ports = await asyncio.wait_for(
                    self._verify_ports_bulk_attempt(
                        ssh_client, port_maps, executor_info.address, token, container_name, extra
                    ),
                    timeout=timeout_seconds
                )

                # If we got any successful ports, return immediately
                if successful_ports:
                    logger.info(_m(f"complete: {len(successful_ports)}/{len(port_maps)} ports verified", extra))
                    return successful_ports, failed_ports

                # First attempt failed completely - retry after delay
                if attempt < max_attempts:
                    retry_delay = 2
                    logger.warning(_m(f"attempt {attempt} failed, retrying in {retry_delay}s...", extra))
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(_m(f"all {max_attempts} attempts failed", extra))
                    return [], port_maps

            except asyncio.TimeoutError:
                logger.error(_m(f"port verification attempt {attempt} timed out after {timeout_seconds}s", extra))
                await self._cleanup_port_test_container(ssh_client, container_name)

                # Retry after delay if not last attempt
                if attempt < max_attempts:
                    retry_delay = 2
                    logger.warning(_m(f"timeout on attempt {attempt}, retrying in {retry_delay}s...", extra))
                    await asyncio.sleep(retry_delay)
                else:
                    return [], port_maps

            except Exception as e:
                logger.error(_m(f"port verification attempt {attempt} failed: {e}", extra), exc_info=True)
                await self._cleanup_port_test_container(ssh_client, container_name)

                # Retry after delay if not last attempt
                if attempt < max_attempts:
                    retry_delay = 2
                    logger.warning(_m(f"exception on attempt {attempt}, retrying in {retry_delay}s...", extra))
                    await asyncio.sleep(retry_delay)
                else:
                    return [], port_maps

        return [], port_maps

    async def _test_single_port_with_session(
        self,
        session: aiohttp.ClientSession,
        host: str,
        internal_port: int,
        external_port: int,
        token: str,
        extra: dict = {},
    ) -> bool:
        """Test a single port expecting token:port response via HTTP using provided session."""
        url = f"http://{host}:{external_port}/"
        expected = f"{token}:{internal_port}"

        try:
            # Use 3 second timeout instead of 40 - we just need to know if port is open
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                text = await resp.text()
                if text.strip() == expected:
                    logger.debug(_m(f"port {internal_port} ok", extra))
                    return True
                else:
                    logger.warning(_m(f"port {internal_port} wrong response: {text[:50]}", extra))
                    return False
        except Exception as e:
            logger.warning(_m(f"port {internal_port} failed: {str(e)[:100]}", extra))
            return False

    async def save_to_db(
        self,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        successful_ports: list[tuple[int, int]],
        failed_ports: list[tuple[int, int]],
        extra: dict = {},
    ):
        """Save successful port verification results to database."""
        try:
            # Prepare database records for successful ports only
            db_records = [
                PortMapping(
                    miner_hotkey=miner_hotkey,
                    executor_id=UUID(executor_info.uuid),
                    internal_port=internal_port,
                    external_port=external_port,
                    is_successful=True,
                )
                for internal_port, external_port in successful_ports
            ]
            for internal_port, external_port in failed_ports:
                db_records.append(
                    PortMapping(
                        miner_hotkey=miner_hotkey,
                        executor_id=UUID(executor_info.uuid),
                        internal_port=internal_port,
                        external_port=external_port,
                        is_successful=False,
                    )
                )

            if db_records:
                await self.port_mapping_dao.upsert_port_results(db_records)
                cleaned_ports = await self.port_mapping_dao.clean_ports(db_records[0].executor_id, 24*60)
                logger.info(_m(f"saved {len(db_records)} ports to db, {cleaned_ports=}", extra))

        except Exception as e:
            logger.error(_m(f"save to db failed: {e}", extra), exc_info=True)

    async def cleanup_docker_containers(self, ssh_client: SSHClientConnection, executor_info: ExecutorSSHInfo, extra: dict = {}):
        try:
            # get all pod names from rented machine
            pod_names = []
            rented_machine = await self.redis_service.get_rented_machine(executor_info)
            if rented_machine:
                pod_names = [pod.get("name", "") for pod in rented_machine.get("containers", [])]
            
            command = '/usr/bin/docker ps -a --filter "name=^/container_" --format "{{.Names}}"'

            result = await ssh_client.run(command)
            container_names = []

            if result.stdout.strip():
                container_names.extend(result.stdout.strip().split("\n"))

            container_names = [container for container in container_names if container not in pod_names]
            if container_names:
                container_names_str = " ".join(container_names)
                logger.info(_m(f"cleanup: found {len(container_names)} containers {container_names_str}", extra))
                command = f"/usr/bin/docker rm {container_names_str} -f"
                await ssh_client.run(command)
                command = "/usr/bin/docker volume prune -af"
                await ssh_client.run(command)
                logger.info(_m(f"cleanup: removed {len(container_names)} containers", extra))
        except Exception as e:
            logger.error(_m(f"cleanup docker containers failed: {e}", extra), exc_info=True)

    def get_available_port_maps(
        self, executor_info: ExecutorSSHInfo, batch_size: int = 1000, rented_external_ports: set[int] | None = None
    ) -> list[tuple[int, int]]:
        """Get a list of available port maps for batch verification. with priority for PREFERRED_POD_PORTS"""
        if rented_external_ports is None:
            rented_external_ports = set()

        if executor_info.port_mappings:
            port_mappings: list[tuple[int, int]] = json.loads(executor_info.port_mappings)
            port_mappings = [
                (internal_port, external_port)
                for internal_port, external_port in port_mappings
                if internal_port != executor_info.ssh_port
                and external_port != executor_info.ssh_port
                and external_port not in rented_external_ports
            ]

            # Prioritize preferred ports from existing port mappings
            preferred_mappings = [
                mapping
                for mapping in port_mappings
                if mapping[0] in PREFERRED_POD_PORTS or mapping[1] in PREFERRED_POD_PORTS
            ]
            remaining_mappings = [mapping for mapping in port_mappings if mapping not in preferred_mappings]

            # Combine preferred first, then sample from remaining
            result = preferred_mappings[:]
            if len(result) < batch_size and remaining_mappings:
                additional_needed = batch_size - len(result)
                additional_ports = random.sample(remaining_mappings, min(additional_needed, len(remaining_mappings)))
                result.extend(additional_ports)

            return result[:batch_size]

        # Generate ports from range
        if executor_info.port_range:
            if "-" in executor_info.port_range:
                min_port, max_port = map(int, (part.strip() for part in executor_info.port_range.split("-")))
                ports = list(range(min_port, max_port + 1))
            else:
                ports = list(map(int, (part.strip() for part in executor_info.port_range.split(","))))
        else:
            # Default range if port_range is empty
            ports = list(range(20000, 65535))

        ports = [port for port in ports if port != executor_info.ssh_port and port not in rented_external_ports]

        if not ports:
            return []

        # Prioritize preferred ports first
        preferred_ports = [port for port in PREFERRED_POD_PORTS if port in ports]
        remaining_ports = [port for port in ports if port not in PREFERRED_POD_PORTS]

        # Start with preferred ports
        selected_ports = preferred_ports[:]

        # Add remaining ports if needed
        if len(selected_ports) < batch_size and remaining_ports:
            additional_needed = batch_size - len(selected_ports)
            additional_ports = random.sample(remaining_ports, min(additional_needed, len(remaining_ports)))
            selected_ports.extend(additional_ports)

        return [(port, port) for port in selected_ports[:batch_size]]

    async def verify_port_dind(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        internal_port: int,
        external_port: int,
        sysbox_runtime: bool = False,
        extra: dict = {},
    ) -> DockerConnectionCheckResult:
        extra.update(
            {"internal_port": internal_port, "external_port": external_port,}
        )

        container_name = f"container_{miner_hotkey}_{external_port}"

        try:
            logger.info(_m(f"dind: start docker port={internal_port}", extra))

            docker_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
            command = (
                f"/usr/bin/docker run -d "
                f'{"--runtime=sysbox-runc " if sysbox_runtime else ""}'
                f"--name {container_name} --gpus all "
                f"-p {internal_port}:22 "
                f"{DOCKER_DIND_IMAGE} "
                f"{docker_cmd}"
            )

            logger.debug(_m(f"run: {command[:100]}...", extra))

            result = await ssh_client.run(command)
            if result.exit_status != 0:
                error_message = result.stderr.strip() if result.stderr else "No error message"
                logger.error(_m(f"dind docker creation failed: {error_message} port={internal_port}", extra))

                try:
                    command = f"/usr/bin/docker rm {container_name} -f"
                    await ssh_client.run(command)
                except Exception:
                    pass

                failure_msg = f"dind: check failed port={internal_port}"
                return DockerConnectionCheckResult(success=False, log_text=failure_msg, sysbox_runtime=sysbox_runtime,)

            logger.info(_m(f"dind: docker created", extra))

            await asyncio.sleep(5)

            pkey = asyncssh.import_private_key(private_key)
            async with asyncssh.connect(
                host=executor_info.address, port=external_port, username="root", client_keys=[pkey], known_hosts=None,
            ) as container_ssh_client:
                logger.info(_m(f"dind: ssh connected", extra))

                if sysbox_runtime:
                    command = "docker pull hello-world"
                    result = await container_ssh_client.run(command)
                    sysbox_success = result.exit_status == 0
                    status = "ok" if sysbox_success else "fail"
                    logger.info(_m(f"dind: sysbox test {status}", extra))

                    if not sysbox_success:
                        error_message = result.stderr.strip() if result.stderr else "No error message"
                        logger.debug(_m(f"sysbox test failed: {error_message}", extra))
                        sysbox_runtime = False

            command = f"/usr/bin/docker rm {container_name} -f"
            await ssh_client.run(command)

            success_msg = f"dind: check ok port={internal_port}"
            logger.info(_m(success_msg, extra))

            return DockerConnectionCheckResult(success=True, log_text=success_msg, sysbox_runtime=sysbox_runtime,)
        except Exception as e:
            logger.error(_m(f"dind check failed: {str(e)} port={internal_port}", extra), exc_info=True)

            try:
                command = f"/usr/bin/docker rm {container_name} -f"
                await ssh_client.run(command)
            except Exception:
                pass

            failure_msg = f"dind: check failed port={internal_port}"
            return DockerConnectionCheckResult(success=False, log_text=failure_msg, sysbox_runtime=sysbox_runtime,)
