import asyncio
import json
import logging
import random
from typing import Any
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
)
from services.redis_service import (
    AVAILABLE_PORT_MAPS_PREFIX,
    RedisService,
)

# Constants
BATCH_VERIFIER_CONTAINER_PREFIX = "container_batch_verifier"
BATCH_VERIFIER_IMAGE = "daturaai/batch-port-verifier:0.0.0"

logger = logging.getLogger(__name__)


class DockerConnectionCheckResult(BaseModel):
    success: bool
    log_text: str | None = None
    sysbox_runtime: bool


class ExecutorConnectivityService:
    def __init__(self, redis_service: "RedisService", port_mapping_dao: PortMappingDao):
        self.redis_service = redis_service
        self.port_mapping_dao = port_mapping_dao

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
            await self.cleanup_docker_containers(ssh_client, extra)

            port_maps = self.get_available_port_maps(executor_info, BATCH_PORT_VERIFICATION_SIZE)
            if not port_maps:
                return DockerConnectionCheckResult(
                    success=False,
                    log_text="No port available for docker container",
                    sysbox_runtime=sysbox_runtime,
                )

            # Debug: show port mappings summary
            logger.debug(_m(f"Checking {len(port_maps)} port mappings", extra))

            successful_ports, failed_ports = await self.verify_other_ports(ssh_client, port_maps, executor_info, extra)
            dind_port = successful_ports.pop(0) if successful_ports else port_maps[0]
            dind_result = await self.verify_single_port(
                ssh_client,
                job_batch_id,
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
            success_percentage = (
                (len(successful_ports) / total_checked * 100) if total_checked > 0 else 0
            )

            # Log verification summary
            dind_status = "ok" if dind_result.success else "failed"
            batch_successful_count = len(successful_ports) - (1 if dind_result.success else 0)
            batch_status = "ok" if batch_successful_count > 0 else "failed"

            if not successful_ports:
                failure_msg = "No working ports found"
                return DockerConnectionCheckResult(
                    success=False,
                    log_text=failure_msg,
                    sysbox_runtime=sysbox_runtime,
                )

            # Save successful ports
            redis_task = self.save_to_redis(executor_info, miner_hotkey, successful_ports, extra)
            db_task = self.save_to_db(executor_info, miner_hotkey, successful_ports, failed_ports, extra)
            await asyncio.gather(redis_task, db_task)


            # Create detailed success message
            successful_internal_ports = [port_pair[0] for port_pair in successful_ports]
            failed_internal_ports = [port_pair[0] for port_pair in failed_ports]

            success_sample = sorted(successful_internal_ports)[:5]
            failed_sample = sorted(failed_internal_ports)[:5]

            success_msg = f"Port verification completed successfully {success_percentage:.0f}% ports available. "
            success_msg += f"{dind_status}, {batch_status}, {len(successful_ports)} success ports: {success_sample}"

            if failed_ports:
                success_msg += f", {len(failed_ports)} failed ports: {failed_sample}"
            logger.info(_m(success_msg, extra))

            return DockerConnectionCheckResult(
                success=True,
                log_text=success_msg,
                sysbox_runtime=sysbox_runtime,
            )
        except Exception as e:
            # Log overall verification failure
            logger.error(
                _m(f"error: Port verification failed - {str(e)} (executor={executor_info.address})", extra), exc_info=True
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
            extra: dict = {}
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Check multiple ports using dockerized batch port verifier service.

        Returns:
            tuple: (successful_port_pairs, failed_port_pairs)
        """
        if not port_maps:
            return [], []

        # Use first port as API_PORT for checker container
        api_internal, api_external = port_maps[0]  # external port
        ports_to_check = port_maps[1:]
        container_name = f"{BATCH_VERIFIER_CONTAINER_PREFIX}_{api_external}"

        # Log batch check start
        logger.info(
            _m(f"batch-check: Starting batch verifier on port {api_internal}:{api_external}, checking {len(ports_to_check)} ports...", extra)
        )

        try:
            # Start Docker container
            command = (
                f"/usr/bin/docker run -d --name {container_name} --network=host "
                f"-e API_PORT={api_internal} {BATCH_VERIFIER_IMAGE}"
            )

            # Debug: log command
            logger.debug(_m(f"DEBUG: Running command: {command[:100]}...", extra))

            result = await ssh_client.run(command)
            if result.exit_status != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(
                    _m(f"error: Batch container start failed - {error_msg} (api_port={api_external})", extra), exc_info=True
                )
                return [], []

            # Log container started
            logger.info(_m(f"batch-check: Container started on port {api_external}, waiting for health...", extra))

            # Wait for health endpoint
            if not await self._wait_for_health(executor_info.address, api_external, extra):
                logger.error(
                    _m(f"error: Batch service health check failed - Service did not become healthy (api_port={api_external})", extra), exc_info=True
                )
                return [], []

            # Log health ready
            logger.info(_m(f"batch-check: Service healthy on port {api_external}, sending request...", extra))

            # Log request sending
            logger.info(
                _m(f"batch-check: Port verification request sent for {len(ports_to_check)} ports...", extra)
            )

            # Send port check request
            results = await self._send_port_check_request(
                executor_info.address, api_external, ports_to_check, extra
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
                _m(f"batch-check: Results: {successful_count}/{len(ports_to_check)} + api_port", extra)
            )

            # Log batch completion
            status = "SUCCESS" if successful_count > 0 else "failed"
            logger.info(
                _m(f"batch-check: Batch verification completed: {status} ({len(ports_to_check)} ports checked)", extra)
            )

            return successful_ports, failed_ports

        except Exception as e:
            logger.error(_m(f"error: Batch port verification failed - {str(e)} (api_port={api_external})", extra), exc_info=True)
            return [], []
        finally:
            # Cleanup container
            try:
                cleanup_command = f"/usr/bin/docker rm -f {container_name}"
                await ssh_client.run(cleanup_command)
            except Exception as e:
                logger.debug(_m(f"Container cleanup warning: {e}", extra))

    async def _wait_for_health(self, external_ip: str, api_port: int, extra: dict = {}) -> bool:
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
                                logger.debug(_m(f"Health check successful for {external_ip}:{api_port}", extra))
                                return True
                except Exception:
                    pass  # Continue retrying

                await asyncio.sleep(0.5)

        return False

    async def _send_port_check_request(
        self, external_ip: str, api_port: int, port_maps: list[tuple[int, int]], extra: dict = {}
    ) -> dict[str, bool]:
        """Send HTTP request to check ports."""
        check_url = f"http://{external_ip}:{api_port}/check-ports"
        if not port_maps:
            logger.info(_m("No ports to check, skipping request...", extra))
            return {}

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
                        logger.error(_m(f"Port check request failed with status {response.status}", extra))
                        return {}
        except Exception as e:
            logger.error(_m(f"Error sending port check request: {e}", extra), exc_info=True)
            return {}

    async def save_to_redis(
        self, executor_info: ExecutorSSHInfo, miner_hotkey: str, successful_ports: list[Any], extra: dict = {}
    ):
        key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_hotkey}:{executor_info.uuid}"
        MAX_REDIS_SAVE = 10
        MAX_REDIS_KEEP = 10
        for internal_port, external_port in successful_ports[:MAX_REDIS_SAVE]:
            port_map = f"{internal_port},{external_port}"

            # delete all the same port_maps in the list
            await self.redis_service.lrem(key=key, element=port_map)

            # insert port_map in the list
            await self.redis_service.lpush(key, port_map)

            # keep the latest 10 port maps
            port_maps = await self.redis_service.lrange(key)
            if len(port_maps) > MAX_REDIS_KEEP:
                await self.redis_service.rpop(key)


    async def save_to_db(
        self,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        successful_ports: list[tuple[int, int]],
        failed_ports: list[tuple[int, int]],
        extra: dict = {}
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
                    is_successful=True
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
                        is_successful=False
                    )
                )

            if db_records:
                await self.port_mapping_dao.upsert_port_results(db_records)
                await self.port_mapping_dao.clean_ports(db_records[0].executor_id)
                logger.info(_m(f"Saved {len(db_records)} successful ports to database", extra))

        except Exception as e:
            logger.error(_m(f"Error saving ports to database: {e}", extra), exc_info=True)
            # Redis still works as fallback

    async def cleanup_docker_containers(self, ssh_client: SSHClientConnection, extra: dict = {}):
        # Clean container_ prefixed containers
        command = '/usr/bin/docker ps -a --filter "name=^/container_" --format "{{.Names}}"'
        result = await ssh_client.run(command)
        container_names = []

        if result.stdout.strip():
            container_names.extend(result.stdout.strip().split("\n"))

        # Log cleanup start
        logger.info(
            _m(f"CLEANUP: Starting cleanup of old containers... found: {len(container_names)} - {container_names}", extra)
        )

        if container_names:
            container_names_str = " ".join(container_names)

            command = f"/usr/bin/docker rm {container_names_str} -f"
            await ssh_client.run(command)

            command = "/usr/bin/docker volume prune -af"
            await ssh_client.run(command)

        # Log cleanup completion
        logger.info(_m(f"CLEANUP: Cleanup completed, removed: {len(container_names)} containers", extra))

    def get_available_port_maps(
        self,
        executor_info: ExecutorSSHInfo,
        batch_size: int = 1000,
    ) -> list[tuple[int, int]]:
        """Get a list of available port maps for batch verification. with priority for PREFERED_POD_PORTS"""
        if executor_info.port_mappings:
            port_mappings: list[tuple[int, int]] = json.loads(executor_info.port_mappings)
            port_mappings = [
                (internal_port, external_port)
                for internal_port, external_port in port_mappings
                if internal_port != executor_info.ssh_port
                and external_port != executor_info.ssh_port
            ]

            # Prioritize preferred ports from existing port mappings
            preferred_mappings = [
                mapping for mapping in port_mappings
                if mapping[0] in PREFERRED_POD_PORTS or mapping[1] in PREFERRED_POD_PORTS
            ]
            remaining_mappings = [
                mapping for mapping in port_mappings
                if mapping not in preferred_mappings
            ]

            # Combine preferred first, then sample from remaining
            result = preferred_mappings[:]
            if len(result) < batch_size and remaining_mappings:
                additional_needed = batch_size - len(result)
                additional_ports = random.sample(
                    remaining_mappings,
                    min(additional_needed, len(remaining_mappings))
                )
                result.extend(additional_ports)

            return result[:batch_size]

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

        # Prioritize preferred ports first
        preferred_ports = [port for port in PREFERRED_POD_PORTS if port in ports]
        remaining_ports = [port for port in ports if port not in PREFERRED_POD_PORTS]

        # Start with preferred ports
        selected_ports = preferred_ports[:]

        # Add remaining ports if needed
        if len(selected_ports) < batch_size and remaining_ports:
            additional_needed = batch_size - len(selected_ports)
            additional_ports = random.sample(
                remaining_ports,
                min(additional_needed, len(remaining_ports))
            )
            selected_ports.extend(additional_ports)

        return [(port, port) for port in selected_ports[:batch_size]]

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
        extra: dict = {}
    ) -> DockerConnectionCheckResult:
        extra.update( {
            "internal_port": internal_port,
            "external_port": external_port,
        })

        container_name = f"container_{miner_hotkey}_{external_port}"

        try:
            # Log dind check start
            logger.info(_m(f"dind-check: Starting container on port {internal_port}...", extra))

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
            logger.debug(_m(f"DEBUG: Running command: {command[:100]}...", extra))

            result = await ssh_client.run(command)
            if result.exit_status != 0:
                error_message = (
                    result.stderr.strip() if result.stderr else "No error message available"
                )

                # Log container creation failure
                logger.error(
                    _m(f"error: dind container creation failed - {error_message} (port={internal_port})", extra)
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
                _m(f"dind-check: Container created on port {internal_port}, waiting for SSH...", extra)
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
                logger.info(_m(f"dind-check: SSH connection established on port {internal_port}", extra))

                if sysbox_runtime:
                    command = "docker pull hello-world"
                    result = await container_ssh_client.run(command)
                    sysbox_success = result.exit_status == 0

                    # Log sysbox test result
                    status = "ok" if sysbox_success else "failed"
                    logger.info(_m(f"dind-check: Sysbox runtime test: {status}", extra))

                    if not sysbox_success:
                        error_message = (
                            result.stderr.strip() if result.stderr else "No error message available"
                        )
                        logger.debug(_m(f"error: Sysbox runtime test failed - {error_message}", extra))
                        sysbox_runtime = False

            command = f"/usr/bin/docker rm {container_name} -f"
            await ssh_client.run(command)

            # Log dind check completion
            success_msg = f"dind-check: Container check completed: SUCCESS (port {internal_port})"
            logger.info(_m(success_msg, extra))

            return DockerConnectionCheckResult(
                success=True,
                log_text=success_msg,
                sysbox_runtime=sysbox_runtime,
            )
        except Exception as e:
            # Log exception with better formatting
            logger.error(_m(f"error: dind check failed - {str(e)} (port={internal_port})", extra), exc_info=True)

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
