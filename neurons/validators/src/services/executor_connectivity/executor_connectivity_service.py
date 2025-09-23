import asyncio
import json
import logging
import random
from typing import Any

import aiohttp
import asyncssh
from asyncssh import SSHClientConnection
from datura.requests.miner_requests import ExecutorSSHInfo
from pydantic import BaseModel

from core.config import settings
from services.const import (
    BATCH_PORT_VERIFICATION_SIZE,
    DOCKER_DIND_IMAGE,
)
from services.redis_service import (
    AVAILABLE_PORT_MAPS_PREFIX,
)

# Constants
BATCH_VERIFIER_CONTAINER_PREFIX = "batch_verifier"
BATCH_VERIFIER_IMAGE = "arhangel662/batch-port-verifier:latest"

logger = logging.getLogger(__name__)


class DockerConnectionCheckResult(BaseModel):
    success: bool
    log_text: str | None = None
    sysbox_runtime: bool


class ExecutorConnectivityService:
    def __init__(self, redis_service):
        self.redis_service = redis_service

    async def batch_verify_ports(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        sysbox_runtime: bool = False,
    ) -> DockerConnectionCheckResult:
        """Verify multiple ports concurrently."""
        try:
            await self.cleanup_docker_containers(ssh_client)

            port_maps = self.get_available_port_maps(executor_info, BATCH_PORT_VERIFICATION_SIZE)
            if not port_maps:
                return DockerConnectionCheckResult(
                    success=False,
                    log_text="No port available for docker container",
                    sysbox_runtime=sysbox_runtime,
                )

            # Debug: show port mappings summary
            logger.debug(f"Checking {len(port_maps)} port mappings")

            # Start dind and batch checks
            dind_ports = port_maps[0]
            dind_task = self.verify_single_port(
                ssh_client,
                job_batch_id,
                miner_hotkey,
                executor_info,
                private_key,
                public_key,
                dind_ports[0],
                dind_ports[1],
                sysbox_runtime,
            )
            other_ports_task = self.verify_other_ports(ssh_client, port_maps[1:], executor_info)

            # Wait for both checks to complete
            dind_result, (batch_successful_ports, batch_failed_ports) = await asyncio.gather(
                dind_task, other_ports_task
            )

            # Collect successful and failed port pairs
            successful_ports = []  # for Redis: list[tuple[int, int]]
            failed_ports = []  # for statistics: list[tuple[int, int]]

            # Add dind port pair
            if dind_result.success:
                successful_ports.append(dind_ports)
                sysbox_runtime = dind_result.sysbox_runtime
            else:
                failed_ports.append(dind_ports)

            # Add batch results (already processed port pairs)
            successful_ports.extend(batch_successful_ports)
            failed_ports.extend(batch_failed_ports)

            # Calculate statistics
            total_checked = len(successful_ports) + len(failed_ports)
            success_percentage = (
                (len(successful_ports) / total_checked * 100) if total_checked > 0 else 0
            )

            # Log verification summary
            dind_status = "ok" if dind_result.success else "failed"
            batch_successful_count = len(successful_ports) - (1 if dind_result.success else 0)
            batch_status = "ok" if batch_successful_count > 0 else "failed"
            logger.info(
                f"summary: dind={dind_status}, batch={batch_status}, successful ports: {len(successful_ports)}"
            )

            if not successful_ports:
                failure_msg = "No working ports found"
                return DockerConnectionCheckResult(
                    success=False,
                    log_text=failure_msg,
                    sysbox_runtime=sysbox_runtime,
                )

            # Save successful ports to Redis
            await self.save_to_redis(executor_info, miner_hotkey, successful_ports)

            # Create detailed success message
            successful_internal_ports = [port_pair[0] for port_pair in successful_ports]
            failed_internal_ports = [port_pair[0] for port_pair in failed_ports]

            success_sample = sorted(successful_internal_ports)[:5]
            failed_sample = sorted(failed_internal_ports)[:5]

            success_msg = f"Port verification completed successfully {success_percentage:.0f}% ports available. "
            success_msg += f"{len(successful_ports)} success ports: {success_sample}"

            if failed_ports:
                success_msg += f", {len(failed_ports)} failed ports: {failed_sample}"

            return DockerConnectionCheckResult(
                success=True,
                log_text=success_msg,
                sysbox_runtime=sysbox_runtime,
            )
        except Exception as e:
            # Log overall verification failure
            logger.error(
                f"error: Port verification failed - {str(e)} (executor={executor_info.address})"
            )

            return DockerConnectionCheckResult(
                success=False,
                log_text=f"Verification failed: {str(e)}",
                sysbox_runtime=sysbox_runtime,
            )

    async def verify_other_ports(
        self,
        ssh_client: SSHClientConnection,
        port_maps: list[tuple[int, int]],
        executor_info: ExecutorSSHInfo,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Check multiple ports using dockerized batch port verifier service.

        Returns:
            tuple: (successful_port_pairs, failed_port_pairs)
        """
        if not port_maps:
            return [], []

        # Use first port as API_PORT for checker container
        api_port = port_maps[0][1]  # external port
        ports_to_check = port_maps[1:]
        container_name = f"{BATCH_VERIFIER_CONTAINER_PREFIX}_{api_port}"

        # Log batch check start
        logger.info(
            f"batch-check: Starting batch verifier on port {api_port}, checking {len(ports_to_check)} ports..."
        )

        try:
            # Start Docker container
            command = (
                f"/usr/bin/docker run -d --name {container_name} --network=host "
                f"-e API_PORT={api_port} {BATCH_VERIFIER_IMAGE}"
            )

            # Debug: log command
            logger.debug(f"DEBUG: Running command: {command[:100]}...")

            result = await ssh_client.run(command)
            if result.exit_status != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(
                    f"error: Batch container start failed - {error_msg} (api_port={api_port})"
                )
                return [], []

            # Log container started
            logger.info(f"batch-check: Container started on port {api_port}, waiting for health...")

            # Wait for health endpoint
            if not await self._wait_for_health(executor_info.address, api_port):
                logger.error(
                    f"error: Batch service health check failed - Service did not become healthy (api_port={api_port})"
                )
                return [], []

            # Log health ready
            logger.info(f"batch-check: Service healthy on port {api_port}, sending request...")

            # Log request sending
            logger.info(
                f"batch-check: Port verification request sent for {len(ports_to_check)} ports..."
            )

            # Send port check request
            results = await self._send_port_check_request(
                executor_info.address, api_port, ports_to_check
            )

            # Process results into port pairs
            successful_ports = []
            failed_ports = []

            # Add api_port to successful ports (it was used for the service)
            api_port_pair = port_maps[0]
            successful_ports.append(api_port_pair)

            # Process other ports based on results
            for port_pair in ports_to_check:
                internal_port = port_pair[0]
                # results uses string keys
                if results.get(str(internal_port), False):
                    successful_ports.append(port_pair)
                else:
                    failed_ports.append(port_pair)

            # Count successful ports
            successful_count = len(successful_ports) - 1  # exclude api_port from count

            # Log batch results
            logger.info(
                f"batch-check: Results received - successful: {successful_count}/{len(ports_to_check)} + api_port"
            )

            # Log batch completion
            status = "SUCCESS" if successful_count > 0 else "failed"
            logger.info(
                f"batch-check: Batch verification completed: {status} ({len(ports_to_check)} ports checked)"
            )

            return successful_ports, failed_ports

        except Exception as e:
            logger.error(f"error: Batch port verification failed - {str(e)} (api_port={api_port})")
            return [], []
        finally:
            # Cleanup container
            try:
                cleanup_command = f"/usr/bin/docker rm -f {container_name}"
                await ssh_client.run(cleanup_command)
            except Exception as e:
                logger.debug(f"Container cleanup warning: {e}")

    async def _wait_for_health(self, external_ip: str, api_port: int) -> bool:
        """Wait for batch port verifier service to become healthy."""
        health_url = f"http://{external_ip}:{api_port}/health"
        timeout = 10  # seconds
        start_time = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession() as session:
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    async with session.get(
                        health_url, timeout=aiohttp.ClientTimeout(total=2)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("status") == "ok":
                                return True
                except Exception:
                    pass  # Continue retrying

                await asyncio.sleep(0.5)

        return False

    async def _send_port_check_request(
        self, external_ip: str, api_port: int, port_maps: list[tuple[int, int]]
    ) -> dict[int, bool]:
        """Send HTTP request to check ports."""
        check_url = f"http://{external_ip}:{api_port}/check-ports"

        # Prepare request payload
        payload = {"external_ip": external_ip, "ports": port_maps}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    check_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", {})
                    else:
                        logger.error(f"Port check request failed with status {response.status}")
                        return [], []
        except Exception as e:
            logger.error(f"Error sending port check request: {e}")
            return {}

    async def save_to_redis(
        self, executor_info: ExecutorSSHInfo, miner_hotkey: str, successful_ports: list[Any]
    ):
        key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_hotkey}:{executor_info.uuid}"
        MAX_REDIS_SAVE = 10
        for internal_port, external_port in successful_ports[:MAX_REDIS_SAVE]:
            port_map = f"{internal_port},{external_port}"

            # delete all the same port_maps in the list
            await self.redis_service.lrem(key=key, element=port_map)

            # insert port_map in the list
            await self.redis_service.lpush(key, port_map)

            # keep the latest 10 port maps
            port_maps = await self.redis_service.lrange(key)
            if len(port_maps) > 10:
                await self.redis_service.rpop(key)

    async def cleanup_docker_containers(self, ssh_client: SSHClientConnection):
        # Clean container_ prefixed containers
        command = '/usr/bin/docker ps -a --filter "name=^/container_" --format "{{.Names}}"'
        result = await ssh_client.run(command)
        container_names = []

        if result.stdout.strip():
            container_names.extend(result.stdout.strip().split("\n"))

        # Clean port_checker prefixed containers
        command = f'/usr/bin/docker ps -a --filter "name=^/{BATCH_VERIFIER_CONTAINER_PREFIX}_" --format "{{.Names}}"'
        result = await ssh_client.run(command)

        if result.stdout.strip():
            container_names.extend(result.stdout.strip().split("\n"))

        # Log cleanup start
        logger.info(
            f"CLEANUP: Starting cleanup of old containers... found: {len(container_names)} - {container_names}"
        )

        if container_names:
            container_names_str = " ".join(container_names)

            command = f"/usr/bin/docker rm {container_names_str} -f"
            await ssh_client.run(command)

            command = "/usr/bin/docker volume prune -af"
            await ssh_client.run(command)

        # Log cleanup completion
        logger.info(f"CLEANUP: Cleanup completed, removed: {len(container_names)} containers")

    def get_available_port_maps(
        self,
        executor_info: ExecutorSSHInfo,
        batch_size: int = 1000,
    ) -> list[tuple[int, int]]:
        """Get a list of available port maps for batch verification."""
        if executor_info.port_mappings:
            port_mappings: list[tuple[int, int]] = json.loads(executor_info.port_mappings)
            port_mappings = [
                (internal_port, external_port)
                for internal_port, external_port in port_mappings
                if internal_port != executor_info.ssh_port
                and external_port != executor_info.ssh_port
            ]

            # Return up to batch_size port mappings
            return random.sample(port_mappings, 1000)

        # Generate ports from range
        if executor_info.port_range:
            if "-" in executor_info.port_range:
                min_port, max_port = map(
                    int, (part.strip() for part in executor_info.port_range.split("-"))
                )
                ports = list(range(min_port, max_port + 1))
            else:
                ports = list(
                    map(int, (part.strip() for part in executor_info.port_range.split(",")))
                )
        else:
            # Default range if port_range is empty
            ports = list(range(40000, 65536))

        ports = [port for port in ports if port != executor_info.ssh_port]

        if not ports:
            return []

        # Select random ports for batch verification
        selected_ports = random.sample(ports, min(batch_size, len(ports)))
        return [(port, port) for port in selected_ports]

    async def verify_single_port(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        internal_port: int,
        external_port: int,
        sysbox_runtime: bool = False,
    ) -> DockerConnectionCheckResult:
        default_extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "ssh_username": executor_info.ssh_username,
            "ssh_port": executor_info.ssh_port,
            "version": settings.VERSION,
            "sysbox_runtime": sysbox_runtime,
            "internal_port": internal_port,
            "external_port": external_port,
        }

        container_name = f"container_{miner_hotkey}_{external_port}"

        try:
            # Log dind check start
            logger.info(f"dind-check: Starting container on port {internal_port}...")

            docker_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
            command = (
                f'/usr/bin/docker run -d '
                f'{"--runtime=sysbox-runc " if sysbox_runtime else ""}'
                f'--name {container_name} --gpus all '
                f'-p {internal_port}:22 '
                f'{DOCKER_DIND_IMAGE} '
                f'{docker_cmd}'
            )

            # Debug: log command (truncated)
            logger.debug(f"DEBUG: Running command: {command[:100]}...")

            result = await ssh_client.run(command)
            if result.exit_status != 0:
                error_message = (
                    result.stderr.strip() if result.stderr else "No error message available"
                )

                # Log container creation failure
                logger.error(
                    f"error: dind container creation failed - {error_message} (port={internal_port})"
                )

                try:
                    command = f"/usr/bin/docker rm {container_name} -f"
                    await ssh_client.run(command)
                except Exception:
                    pass

                failure_msg = (
                    f"dind-check: Container check completed: failed (port {internal_port})"
                )
                return DockerConnectionCheckResult(
                    success=False,
                    log_text=failure_msg,
                    sysbox_runtime=sysbox_runtime,
                )

            # Log container created, waiting for SSH
            logger.info(
                f"dind-check: Container created on port {internal_port}, waiting for SSH..."
            )

            await asyncio.sleep(5)

            pkey = asyncssh.import_private_key(private_key)
            async with asyncssh.connect(
                host=executor_info.address,
                port=external_port,
                username="root",
                client_keys=[pkey],
                known_hosts=None,
            ) as container_ssh_client:
                # Log SSH connection established
                logger.info(f"dind-check: SSH connection established on port {internal_port}")

                if sysbox_runtime:
                    command = "docker pull hello-world"
                    result = await container_ssh_client.run(command)
                    sysbox_success = result.exit_status == 0

                    # Log sysbox test result
                    status = "ok" if sysbox_success else "failed"
                    logger.info(f"dind-check: Sysbox runtime test: {status}")

                    if not sysbox_success:
                        error_message = (
                            result.stderr.strip() if result.stderr else "No error message available"
                        )
                        logger.debug(f"error: Sysbox runtime test failed - {error_message}")
                        sysbox_runtime = False

            command = f"/usr/bin/docker rm {container_name} -f"
            await ssh_client.run(command)

            # Log dind check completion
            success_msg = f"dind-check: Container check completed: SUCCESS (port {internal_port})"
            logger.info(success_msg)

            return DockerConnectionCheckResult(
                success=True,
                log_text=success_msg,
                sysbox_runtime=sysbox_runtime,
            )
        except Exception as e:
            # Log exception with better formatting
            logger.error(f"error: dind check failed - {str(e)} (port={internal_port})")

            try:
                command = f"/usr/bin/docker rm {container_name} -f"
                await ssh_client.run(command)
            except Exception:
                pass

            failure_msg = f"dind-check: Container check completed: failed (port {internal_port})"
            return DockerConnectionCheckResult(
                success=False,
                log_text=failure_msg,
                sysbox_runtime=sysbox_runtime,
            )
