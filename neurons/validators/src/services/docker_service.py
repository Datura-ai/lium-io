import asyncio
import random
from datetime import datetime
import logging
import time
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4, UUID
import shlex
import secrets
import aiohttp
import asyncssh
import bittensor
import redis.exceptions
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerBaseRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    AddSshPublicKeyRequest,
    RemoveSshPublicKeysRequest,
    ContainerCreated,
    ContainerDeleted,
    ContainerStarted,
    ContainerStopped,
    SshPubKeyAdded,
    SshPubKeyRemoved,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    FailedContainerErrorTypes,
    ExternalVolumeInfo,
    InstallJupyterServerRequest,
    JupyterServerInstalled,
    JupyterInstallationFailed,
    CustomOptions,
    ContainerWarningCode,
    PayloadPortMapping,
    WorkloadKind,
)
from protocol.vc_protocol.compute_requests import RentedMachine

from core.utils import _m, get_extra_info, retry_ssh_command
from services.const import (
    FILLER_CONTAINER_PREFIX,
    POD_CONTAINER_PREFIX,
    PREFERRED_POD_PORTS,
    MIN_PORT_COUNT,
)
from services.redis_service import (
    STREAMING_LOG_CHANNEL,
    RedisService,
)
from services.attestation_service import AttestationService, AttestationError
from services.nvidia_devices import build_gpu_flags
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)

REPOSITORIES = [
    "daturaai/compute-subnet-executor:latest",
    "daturaai/compute-subnet-executor-runner:latest",
    "nickfedor/watchtower",
    "daturaai/pytorch",
    "daturaai/ubuntu",
]

LOG_STREAM_INTERVAL = 5  # 5 seconds
IN_CONTAINER_SSH_BOOTSTRAP_PATH = "/tmp/lium-ssh-bootstrap.sh"

DOCKER_VOLUME_PLUGINS = {
    "s3fs": "mochoa/s3fs-volume-plugin"
}

# DAH-1991: tolerate concurrent health_check_* / container_* on the executor.
# Probe TTL is short (~30s); same-command retry within a 90s budget covers the
# documented race without regenerating port mappings.
_PORT_ALLOCATED_PHRASES = ("port is already allocated", "address already in use", "failed to bind host port")
_PORT_ALLOCATED_RETRY_BUDGET_SEC = 90
_PORT_ALLOCATED_RETRY_SLEEP_SEC = 5
_LOCAL_VOLUME_TIMEOUT_THRESHOLD_GB = 100
_LOCAL_VOLUME_TIMEOUT_BASE_SEC = 30
_LOCAL_VOLUME_TIMEOUT_GB_PER_SEC = 10
_LOCAL_VOLUME_TIMEOUT_MAX_SEC = 180
_FILLER_EXTERNAL_PORT_OFFSET = 20


class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        attestation_service: Annotated[AttestationService, Depends(AttestationService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.attestation_service = attestation_service
        self.lock = asyncio.Lock()
        self.logs_queue: list[dict] = []
        self.log_task: asyncio.Task | None = None
        self.is_realtime_logging = False

    @staticmethod
    def get_container_name(payload: ContainerBaseRequest) -> str:
        if payload.workload_kind == WorkloadKind.FILLER:
            return f"{FILLER_CONTAINER_PREFIX}{payload.pod_id}"
        return f"{POD_CONTAINER_PREFIX}{payload.pod_id}"

    def _ssh_bootstrap_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "assets" / "sshd_bootstrap.sh"

    async def _run_docker_create_with_port_retry(
        self,
        ssh_client,
        command: str,
        container_name: str,
        log_tag: str,
        default_extra: dict,
        timeout: int,
    ) -> None:
        """Run `docker run` with same-command retry on port-allocated races.

        DAH-1991: backend-spawned `health_check_*` probes (TTL ~30s) can land
        on a port we already accepted into `port_maps` during the 20-60s gap
        inside `create_container` (driven by `docker pull`,
        `clean_existing_containers(sleep=10)`, and volume creation). Wait
        through the probe's natural lifetime by retrying the same command on
        a 90s budget. Non-port-allocated errors propagate immediately.

        DAH-2018: Docker reserves the container name during command parse,
        before port-bind. A port-bind failure therefore leaves a Created-state
        container holding `pod_<id>`, and the next same-command attempt would
        otherwise collide with "container name already in use". Between
        attempts (after the backoff sleep, just before the next `docker run`)
        we issue `docker rm -f <container_name>` so the rm→run window stays
        tight. Cleanup failures are warning-logged but do not abort the loop.
        """
        deadline = time.monotonic() + _PORT_ALLOCATED_RETRY_BUDGET_SEC
        attempt = 0
        while True:
            try:
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text="Creating docker container",
                    log_extra=default_extra,
                    timeout=timeout,
                )
                return
            except Exception as e:
                matched_phrase = next((p for p in _PORT_ALLOCATED_PHRASES if p in str(e)), None)
                if matched_phrase is None or time.monotonic() >= deadline:
                    raise
                attempt += 1
                logger.info(
                    _m(
                        "PORT_ALREADY_ALLOCATED_RETRY",
                        extra=get_extra_info({
                            **default_extra,
                            "attempt": attempt,
                            "remaining_sec": int(deadline - time.monotonic()),
                            "sleep_seconds": _PORT_ALLOCATED_RETRY_SLEEP_SEC,
                            "matched_phrase": matched_phrase,
                        }),
                    )
                )
                await asyncio.sleep(_PORT_ALLOCATED_RETRY_SLEEP_SEC)
                # DAH-2018: drop the Created-state container Docker reserved
                # during the prior `docker run` parse, so the next same-command
                # attempt cannot collide with "container name already in use".
                try:
                    rm_cmd = f"/usr/bin/docker rm -f {shlex.quote(container_name)}"
                    await ssh_client.run(rm_cmd)
                except asyncio.CancelledError:
                    raise
                except Exception as rm_exc:
                    logger.warning(
                        _m(
                            "PORT_RETRY_STALE_RM_FAILED",
                            extra=get_extra_info({
                                **default_extra,
                                "container_name": container_name,
                                "rm_error": str(rm_exc),
                            }),
                        )
                    )

    async def _prepare_known_hosts_policy(
        self,
        executor: ExecutorSSHInfo,
        miner_hotkey: str | None,
        log_context: dict,
    ) -> asyncssh.SSHKnownHosts | None:
        try:
            known_hosts, _, _ = await self.attestation_service.prepare_host_policy(
                executor, 
            )
            return known_hosts
        except AttestationError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "Unable to prepare known_hosts policy",
                    extra=get_extra_info({**log_context, "error": str(exc)}),
                )
            )
            return None

    async def generate_portMappings(
        self,
        miner_hotkey: str,
        executor_id: str,
        pod_id: UUID,
        internal_ports: list[int] | None = None,
        initial_port_count: int | None = None,
        enable_jupyter: bool | None = False,
        available_ports_raw: list[PayloadPortMapping] | None = None,
        pod_mapping_raw: list[PayloadPortMapping] | None = None,
        workload_kind: WorkloadKind | None = None,
    ) -> tuple[list[tuple[int, int, int]], tuple[int, int] | None]:
        executor_uuid = UUID(executor_id)

        try:
            # Use distributed lock to prevent race conditions when allocating ports
            async with self.redis_service.acquire_executor_lock(executor_id):
                # Use port data from backend
                if available_ports_raw is not None and pod_mapping_raw is not None:
                    available_ports, pod_mapping = self._convert_payload_ports(available_ports_raw, pod_mapping_raw)
                    logger.info(f"Using port data from backend: {len(available_ports)} available, {len(pod_mapping)} pod mappings")
                else:
                    # No backend data provided - cannot proceed without port information
                    logger.error(f"No port data provided from backend for executor {executor_id}")
                    available_ports = {}
                    pod_mapping = {}

                if not pod_mapping and len(available_ports) < MIN_PORT_COUNT:
                    logger.warning(
                        f"Insufficient ports available ({len(available_ports)}/{MIN_PORT_COUNT}) "
                        f"for executor {executor_id}"
                    )
                    return [], None

                mappings = []
                reused_count = 0
                ssh_port = 22
                jupyter_port = 8888
                jupyter_port_map: tuple[int, int] | None = None

                user_defined = bool(internal_ports)
                docker_internal_ports = internal_ports or self._get_preferred_ports(initial_port_count)
                if ssh_port in docker_internal_ports:
                    docker_internal_ports.remove(ssh_port)
                docker_internal_ports.insert(0, ssh_port)

                if enable_jupyter:
                    if jupyter_port in docker_internal_ports:
                        docker_internal_ports.remove(jupyter_port)
                    docker_internal_ports.insert(1, jupyter_port)

                # Pre-strip every external_port that will be reused so the random.choice
                # / min branch below cannot pick a port still owned by pod_mapping.
                for port in docker_internal_ports:
                    if port in pod_mapping:
                        available_ports.pop(pod_mapping[port]["external_port"], None)

                for port in docker_internal_ports:
                    if port in pod_mapping:
                        port_mapping = pod_mapping[port]
                        mappings.append((port, port_mapping["internal_port"], port_mapping["external_port"]))
                        reused_count += 1
                        continue

                    if not len(available_ports):
                        break

                    filler_external_port = port + _FILLER_EXTERNAL_PORT_OFFSET
                    if (
                        workload_kind == WorkloadKind.FILLER
                        and user_defined
                        and port not in {ssh_port, jupyter_port}
                        and filler_external_port in available_ports
                    ):
                        docker_port = port
                        external_port = filler_external_port
                    elif port in available_ports:
                        docker_port = port
                        external_port = port
                    elif port == ssh_port or port == jupyter_port:
                        docker_port = port
                        external_port = max(available_ports.keys())
                    else:
                        external_port = random.choice(list(available_ports.keys())) if user_defined else min(available_ports.keys())
                        docker_port = port if user_defined else external_port

                    port_mapping = available_ports.pop(external_port)
                    mappings.append((docker_port, port_mapping["internal_port"], external_port))

                allocated_count = len(mappings) - reused_count
                logger.info(
                    f"Generated {len(mappings)} port mappings for pod {pod_id}: "
                    f"reused={reused_count}, allocated={allocated_count}, executor={executor_id}"
                )

                if enable_jupyter:
                    mapping = self._find_mapping_by_docker_port(mappings, jupyter_port)
                    if mapping:
                        jupyter_port_map = (mapping[0], mapping[2])

                # Port reservation now handled by backend

                return mappings, jupyter_port_map

        except (redis.exceptions.LockError, redis.exceptions.LockNotOwnedError) as e:
            logger.error(
                f"Failed to acquire or maintain lock for executor {executor_id} during port mapping generation: {e}",
                exc_info=True
            )
            # Return empty result to signal failure - caller should handle this case
            return [], None

    def _find_mapping_by_docker_port(self, mappings: list[tuple[int, int, int]], docker_port: int) -> tuple[int, int, int] | None:
        """Find a port mapping by docker port number."""
        return next((m for m in mappings if m[0] == docker_port), None)

    def _convert_payload_ports(
        self,
        available_ports_raw: list[PayloadPortMapping],
        pod_mapping_raw: list[PayloadPortMapping],
    ) -> tuple[dict[int, dict], dict[int, dict]]:
        """
        Convert payload port mappings to the format expected by generate_portMappings.

        Returns:
            - available_ports: dict[external_port, port_info_dict]
            - pod_mapping: dict[docker_port, port_info_dict]
        """
        available_ports: dict[int, dict] = {}
        for p in available_ports_raw:
            # Create a minimal port info dict with required fields
            port_info = {
                "internal_port": p.internal_port,
                "external_port": p.external_port,
                "docker_port": p.docker_port,
            }
            available_ports[p.external_port] = port_info

        pod_mapping: dict[int, dict] = {}
        for p in pod_mapping_raw:
            port_info = {
                "internal_port": p.internal_port,
                "external_port": p.external_port,
                "docker_port": p.docker_port,
            }
            # Use docker_port as key if available, otherwise fallback to external_port
            key = p.docker_port if p.docker_port is not None else p.external_port
            pod_mapping[key] = port_info

        return available_ports, pod_mapping

    @staticmethod
    def _build_docker_login_command(username: str, password: str) -> str:
        return (
            f"echo {shlex.quote(password)} | "
            f"/usr/bin/docker login --username {shlex.quote(username)} --password-stdin"
        )

    async def execute_and_stream_logs(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        command: str,
        log_tag: str,
        log_text: str,
        log_extra: dict = {},
        timeout: int = 0,
        raise_exception: bool = True,
    ) -> tuple[bool, str]:
        logger.info(
            _m(
                log_text,
                extra=get_extra_info({
                    **log_extra,
                    "command": command,
                    "timeout_seconds": timeout,
                }),
            ),
        )

        await self.stream_log(log_text, "success", log_tag)

        status = True
        error = ''
        try:
            async with ssh_client.create_process(command) as process:
                if timeout != 0:
                    status, error = await asyncio.wait_for(self._stream_process_output(process, log_tag), timeout=timeout)
                else:
                    status, error = await self._stream_process_output(process, log_tag)
        except asyncio.TimeoutError:
            status = False
            error = "Process timed out"
            await self.stream_log(error, "error", log_tag)
            logger.warning(
                _m(
                    "Docker command timed out",
                    extra=get_extra_info({
                        **log_extra,
                        "command": command,
                        "timeout_seconds": timeout,
                        "log_text": log_text,
                    }),
                )
            )

        if not status and raise_exception:
            raise Exception(f"Failed {log_text}. command: {command} error: {error}")

        return status, error

    async def _stream_process_output(self, process, log_tag):
        status = True
        error = ''

        async for line in process.stdout:
            await self.stream_log(line.strip(), "success", log_tag)

        async for line in process.stderr:
            status = False
            error += line.strip() + "\n"
            await self.stream_log(line.strip(), "error", log_tag)

        return status, error

    async def handle_stream_logs(
        self,
        miner_hotkey,
        executor_id,
        pod_id,
    ):
        default_extra = {
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_id,
            "pod_id": pod_id,
        }

        self.is_realtime_logging = True

        while True:
            await asyncio.sleep(LOG_STREAM_INTERVAL)

            async with self.lock:
                logs_to_process = self.logs_queue[:]
                self.logs_queue.clear()

            if logs_to_process:
                try:
                    await self.redis_service.publish(
                        STREAMING_LOG_CHANNEL,
                        {
                            "logs": logs_to_process,
                            "miner_hotkey": miner_hotkey,
                            "executor_uuid": executor_id,
                            "pod_id": pod_id,
                        },
                    )

                    logger.info(
                        _m(
                            f"Successfully published {len(logs_to_process)} logs",
                            extra=get_extra_info(default_extra),
                        )
                    )

                except Exception as e:
                    logger.error(
                        _m(
                            "Error publishing log stream",
                            extra=get_extra_info({**default_extra, "error": str(e)}),
                        ),
                        exc_info=True,
                    )

            if not self.is_realtime_logging:
                break

        logger.info(
            _m(
                "Exit handle_stream_logs",
                extra=get_extra_info(default_extra),
            )
        )

    async def finish_stream_logs(self):
        self.is_realtime_logging = False
        if self.log_task:
            await self.log_task

    async def check_container_running(
        self, ssh_client: asyncssh.SSHClientConnection, container_name: str, timeout: int = 10
    ):
        """Check if the container is running"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = await ssh_client.run(f"/usr/bin/docker ps -q -f name={container_name}")
            if result.stdout.strip():
                return True
            await asyncio.sleep(1)
        return False

    async def wait_for_port_check_containers(
        self,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        keypair: bittensor.Keypair,
        private_key: str,
        max_retries: int = 2,
        retry_delay: int = 60,
        ssh_client: asyncssh.SSHClientConnection | None = None,
    ) -> tuple[bool, str]:
        """Wait for port check containers to finish before creating rental containers.

        Matches two prefix patterns:
        - 'container_{miner_hotkey}_*' — validator DinD/port-check probes (hotkey-scoped,
          preserved for cross-miner isolation on shared physical hosts)
        - 'health_check_*' — backend executor_health_check probes (hotkey-agnostic,
          backend creates these without a hotkey segment — see DAH-1991)

        DAH-2018: when the caller already holds an open SSH connection, pass it
        in via ``ssh_client`` to avoid the cost (and TOCTOU widening) of a
        second connect — the late re-check inside ``create_container`` runs
        right before ``docker run`` and reuses the existing session.

        Args:
            executor_info: Executor SSH connection info (ignored when
                ``ssh_client`` is provided).
            miner_hotkey: The miner's hotkey to check containers for
            keypair: Bittensor keypair for decrypting private key (ignored when
                ``ssh_client`` is provided).
            private_key: Encrypted SSH private key (ignored when ``ssh_client``
                is provided).
            max_retries: Maximum number of times to check (default 2)
            retry_delay: Seconds to wait between checks (default 60)
            ssh_client: Optional pre-opened SSH session to reuse.

        Returns:
            Tuple of (success: bool, message: str)
            - (True, "No port check containers found") - Can proceed immediately
            - (True, "Port check containers cleared after X attempts") - Waited and cleared
            - (False, "Port check containers still exist after max retries") - Failed to clear
        """
        container_prefix = f"container_{miner_hotkey}_"
        health_check_prefix = "health_check_"

        async def _run_checks(client: asyncssh.SSHClientConnection) -> tuple[bool, str]:
            for attempt in range(max_retries + 1):
                # docker ps OR-s multiple --filter name= flags
                command = (
                    '/usr/bin/docker ps --format "{{.Names}}" '
                    f'--filter "name=^{container_prefix}" '
                    f'--filter "name=^{health_check_prefix}"'
                )
                result = await client.run(command)

                if not result.stdout or not result.stdout.strip():
                    if attempt == 0:
                        return True, "No port check containers found"
                    else:
                        return True, f"Port check containers cleared after {attempt} attempt(s)"

                # Found port check containers
                container_names = result.stdout.strip()

                if attempt < max_retries:
                    logger.info(
                        f"Port check containers exist ({container_names}), "
                        f"waiting {retry_delay}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    # Max retries reached, containers still exist - force cleanup
                    logger.warning(
                        f"Port check containers still running after {max_retries} retries, "
                        f"forcing cleanup: {container_names}"
                    )

                    # Force remove containers matching either prefix.
                    remove_cmd = (
                        "docker ps -q "
                        f"--filter 'name=^{container_prefix}' "
                        f"--filter 'name=^{health_check_prefix}' "
                        "| xargs -r docker rm -f"
                    )
                    await client.run(remove_cmd)

                    logger.info("Forced removal of stale port check containers completed")
                    return True, f"Port check containers forcefully removed after {max_retries} retries"

            # Should never reach here, but just in case
            return False, "Unexpected error in wait_for_port_check_containers"

        if ssh_client is not None:
            try:
                return await _run_checks(ssh_client)
            except Exception as e:
                logger.error(f"Error checking for port check containers: {e}")
                return True, "Unable to check for port check containers, proceeding"

        # No reusable session — open a dedicated SSH connection.
        decrypted_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(decrypted_key)
        try:
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as new_client:
                return await _run_checks(new_client)
        except Exception as e:
            logger.error(f"Error connecting to check for port check containers: {e}")
            # If we can't connect, assume it's safe to proceed
            return True, "Unable to check for port check containers, proceeding"

    async def clean_existing_containers(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        pod_name: str,
        sleep: int = 0,
        clear_volume: bool = True,
        active_container_names: list[str] | None = None,
        active_volume_names: list[str] | None = None,
    ):
        command = f'/usr/bin/docker ps -a --format "{{{{.Names}}}}"'
        result = await ssh_client.run(command)
        if result.stdout.strip():
            # wait until the docker connection check is finished.
            await asyncio.sleep(sleep)

            active_set = set(active_container_names) if active_container_names else set()
            active_volume_set = set(active_volume_names) if active_volume_names else set()
            pod_containers = [
                name for name in result.stdout.strip().split("\n")
                if name == pod_name
                or name.startswith(POD_CONTAINER_PREFIX)
                or name.startswith(FILLER_CONTAINER_PREFIX)
            ]
            stale_containers = []
            for name in pod_containers:
                if name in active_set:
                    continue
                stale_containers.append(name)
            container_names = " ".join(shlex.quote(name) for name in stale_containers)
            if not container_names:
                return

            logger.info(
                _m(
                    "Cleaning existing docker containers",
                    extra=get_extra_info({
                        **default_extra,
                        "container_names": container_names,
                        "active_containers": list(active_set),
                    }),
                ),
            )

            command = f'/usr/bin/docker rm -fv {container_names}'
            await retry_ssh_command(ssh_client, command, 'clean_existing_containers')

            if clear_volume:
                volumes_to_remove = []
                for name in stale_containers:
                    volume_id = name
                    for prefix in (POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX):
                        if name.startswith(prefix):
                            volume_id = name.removeprefix(prefix)
                            break
                    volume_name = f"volume_{volume_id}"
                    if volume_name not in active_volume_set:
                        volumes_to_remove.append(volume_name)
                if volumes_to_remove:
                    volumes = " ".join(shlex.quote(volume) for volume in volumes_to_remove)
                    command = f'/usr/bin/docker volume rm {volumes} 2>/dev/null || true'
                    await retry_ssh_command(ssh_client, command, 'clean_existing_containers')

    async def clean_stale_vloopback_volumes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        skip_volume_names: list[str] | set[str] | None = None,
    ) -> None:
        skip_set = {name for name in (skip_volume_names or []) if name}
        list_volumes_cmd = '/usr/bin/docker volume ls --format "{{.Name}} {{.Driver}}"'
        mounted_volumes_cmd = (
            "/usr/bin/docker ps -a -q | xargs -r /usr/bin/docker inspect --format "
            "'{{range .Mounts}}{{if eq .Type \"volume\"}}{{.Name}}{{\"\\n\"}}{{end}}{{end}}'"
        )

        try:
            volume_result = await ssh_client.run(list_volumes_cmd)
            if getattr(volume_result, "exit_status", 0) != 0:
                logger.warning(
                    _m(
                        "Unable to list vloopback volumes",
                        extra=get_extra_info({
                            **default_extra,
                            "stderr": getattr(volume_result, "stderr", ""),
                        }),
                    )
                )
                return

            vloopback_volumes = set()
            for line in (volume_result.stdout or "").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                name, driver = parts
                if not (
                    name.startswith("volume_")
                    and (driver == "vloopback" or driver.startswith("vloopback:"))
                ):
                    continue
                vloopback_volumes.add(name)
            if not vloopback_volumes:
                return

            mounted_result = await ssh_client.run(mounted_volumes_cmd)
            if getattr(mounted_result, "exit_status", 0) != 0:
                logger.warning(
                    _m(
                        "Unable to inspect mounted Docker volumes",
                        extra=get_extra_info({
                            **default_extra,
                            "stderr": getattr(mounted_result, "stderr", ""),
                        }),
                    )
                )
                return

            mounted_volumes = {
                name.strip() for name in (mounted_result.stdout or "").splitlines() if name.strip()
            }
            stale_volumes = sorted(vloopback_volumes - mounted_volumes - skip_set)
            if not stale_volumes:
                return

            logger.info(
                _m(
                    "Cleaning stale vloopback Docker volumes",
                    extra=get_extra_info({
                        **default_extra,
                        "stale_volumes": stale_volumes,
                        "skipped_volumes": sorted(skip_set),
                    }),
                )
            )
            volumes = " ".join(shlex.quote(volume) for volume in stale_volumes)
            await retry_ssh_command(
                ssh_client,
                f"/usr/bin/docker volume rm {volumes} 2>/dev/null || true",
                "clean_stale_vloopback_volumes",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "Failed to clean stale vloopback volumes",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )

    async def cleanup_failed_container_creation(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        container_name: str,
        volume_name: str | None = None,
        remove_volume: bool = False,
    ) -> None:
        try:
            container = shlex.quote(container_name)
            await retry_ssh_command(
                ssh_client,
                f"/usr/bin/docker rm -fv {container} 2>/dev/null || true",
                "cleanup_failed_container_creation",
            )

            if remove_volume and volume_name:
                volume = shlex.quote(volume_name)
                await retry_ssh_command(
                    ssh_client,
                    f"/usr/bin/docker volume rm {volume} 2>/dev/null || true",
                    "cleanup_failed_container_creation",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "Failed to clean up failed container creation artifacts",
                    extra=get_extra_info({
                        **default_extra,
                        "container_name": container_name,
                        "volume_name": volume_name,
                        "remove_volume": remove_volume,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )

    async def install_open_ssh_server_and_start_ssh_service(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        log_tag: str,
        log_extra: dict,
    ) -> bool:
        local_script_path = self._ssh_bootstrap_script_path()
        container_path = IN_CONTAINER_SSH_BOOTSTRAP_PATH
        success = True

        try:
            script_content = local_script_path.read_text()
        except Exception as exc:
            await self.stream_log("Failed to read SSH bootstrap script", "error", log_tag)
            logger.warning(
                _m(
                    "Failed to read SSH bootstrap script",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "local_script_path": str(local_script_path),
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return False

        try:
            container_name_quoted = shlex.quote(container_name)
            container_path_quoted = shlex.quote(container_path)
            heredoc = f"__LIUM_SSHD_BOOTSTRAP_{uuid4().hex}__"
            create_script_command = (
                f"/usr/bin/docker exec -i {container_name_quoted} sh -c "
                f"\"cat > {container_path_quoted} && chmod +x {container_path_quoted}\" "
                f"<< '{heredoc}'\n"
                f"{script_content}\n"
                f"{heredoc}"
            )
            logger.info(
                _m(
                    "Creating SSH bootstrap script inside container",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "local_script_path": str(local_script_path),
                        "container_path": container_path,
                    }),
                ),
            )
            create_result = await ssh_client.run(create_script_command)
            if create_result.exit_status != 0:
                await self.stream_log("Failed to create SSH bootstrap script in container", "error", log_tag)
                logger.warning(
                    _m(
                        "Failed to create SSH bootstrap script in container",
                        extra=get_extra_info({
                            **log_extra,
                            "container_name": container_name,
                            "container_path": container_path,
                            "exit_status": create_result.exit_status,
                            "stdout": create_result.stdout,
                            "stderr": create_result.stderr,
                        }),
                    )
                )
                return False
        except Exception as exc:
            await self.stream_log("Failed to create SSH bootstrap script in container", "error", log_tag)
            logger.warning(
                _m(
                    "Failed to create SSH bootstrap script in container",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "container_path": container_path,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return False

        command = f"/usr/bin/docker exec {container_name_quoted} sh {container_path_quoted}"
        status, _ = await self.execute_and_stream_logs(
            ssh_client=ssh_client,
            command=command,
            log_tag=log_tag,
            log_text="Bootstrapping SSH daemon and watchdog",
            log_extra=log_extra,
            raise_exception=False,
        )
        success = success and status

        if not success:
            logger.warning(
                _m(
                    "SSH bootstrap script finished with errors",
                    extra=get_extra_info({**log_extra, "container_name": container_name}),
                )
            )

        return success

    async def create_s3fs_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        log_extra: dict,
        volume_info: ExternalVolumeInfo,
        log_tag: str,
    ):
        responses = []
        # install docker volume plugin
        command = "/usr/bin/docker plugin install mochoa/s3fs-volume-plugin --alias s3fs --grant-all-permissions --disable"
        responses.append(await ssh_client.run(command))

        # disable volume plugin
        command = "/usr/bin/docker plugin disable s3fs -f"
        responses.append(await ssh_client.run(command))

        # set credentials
        command = f"/usr/bin/docker plugin set s3fs AWSACCESSKEYID={volume_info.iam_user_access_key} AWSSECRETACCESSKEY={volume_info.iam_user_secret_key}"
        responses.append(await ssh_client.run(command))

        # set allow_other option
        command = '/usr/bin/docker plugin set s3fs DEFAULT_S3FSOPTS="allow_other"'
        responses.append(await ssh_client.run(command))

        # enable volume plugin
        command = "/usr/bin/docker plugin enable s3fs"
        responses.append(await ssh_client.run(command))

        # create volume
        command = f"/usr/bin/docker volume create -d s3fs {volume_info.name}"
        result = await self.execute_and_stream_logs(
            ssh_client=ssh_client,
            command=command,
            log_tag=log_tag,
            log_text="Creating docker volume",
            log_extra=log_extra,
            raise_exception=False,
        )
        is_success, message = result
        if not is_success:
            responses_text = message
            for i, r in enumerate(responses):
                responses_text += f"|Step {i}: exit={r.exit_status}, stdout={r.stdout}, stderr={r.stderr}"
            logger.warning(_m(f"s3fs_volume failed. {responses_text}",extra=get_extra_info({**log_extra})))
        else:
            logger.info(_m("s3fs_volume success", extra=get_extra_info({**log_extra})))

        return result

    async def disable_s3fs_volume_plugin(
        self,
        ssh_client: asyncssh.SSHClientConnection,
    ):
        # disable volume plugin
        command = f"/usr/bin/docker plugin disable s3fs -f"
        await ssh_client.run(command)

    async def run_jupyter(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        jupyter_token: str,
        jupyter_port: int,
        log_tag: str,
        log_extra: dict,
        local_volume: str | None = None,
        local_volume_path: str = '/root',
    ):
        if local_volume:
            temp_container_name = f"temp_jupyter_copy_{uuid4()}"
            try:
                command = (
                    f"/usr/bin/docker run -d --rm -v {local_volume}:/mnt "
                    f"--name {temp_container_name} --entrypoint sh "
                    f"daturaai/compute-subnet-executor:latest -c 'cp /root/app/run_jupyter.sh /mnt/'"
                )
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text="Creating temporary container for script copy",
                    log_extra=log_extra,
                    raise_exception=True,
                )
            finally:
                command = f"/usr/bin/docker rm -f {temp_container_name}"
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text="Removing temporary container",
                    log_extra=log_extra,
                    raise_exception=False,
                )

            command = (
                f"/usr/bin/docker exec {container_name} "
                f"sh -c 'chmod +x {local_volume_path}/run_jupyter.sh'"
            )
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Making run_jupyter.sh executable",
                log_extra=log_extra,
                raise_exception=True,
            )

            command = (
                f"/usr/bin/docker exec {container_name} sh -c "
                f"'{local_volume_path}/run_jupyter.sh --password={jupyter_token} --port={jupyter_port}'"
            )
            status, error = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Running jupyter from volume",
                log_extra=log_extra,
                raise_exception=False,
            )
        else:
            command = f"/usr/bin/docker cp /root/app/run_jupyter.sh {container_name}:/root/run_jupyter.sh"
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Copying run_jupyter.sh to container",
                log_extra=log_extra,
                raise_exception=True
            )
            command = f"/usr/bin/docker exec {container_name} sh -c 'chmod +x /root/run_jupyter.sh'"
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="chmod +x /root/run_jupyter.sh",
                log_extra=log_extra,
                raise_exception=True
            )
            command = f"/usr/bin/docker exec {container_name} sh -c '/root/run_jupyter.sh --password={jupyter_token} --port={jupyter_port}'"
            status, error = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Running jupyter",
                log_extra=log_extra,
                raise_exception=False
            )

        # Only raise exception for actual errors, not warnings or info messages
        if not status and error and any(keyword.lower() in error.lower() for keyword in [
            "Error", "FATAL", "CRITICAL", "Traceback", "Exception",
            "Permission denied", "Address already in use", "No such file or directory",
            "Connection refused", "Port already in use", "Failed to start"
        ]):
            raise Exception(error)

    async def get_docker_root_dir(self, ssh_client: asyncssh.SSHClientConnection):
        """Get Docker storage info using docker info command"""
        command = f"/usr/bin/docker info --format '{{{{.DockerRootDir}}}}'"
        result = await ssh_client.run(command)
        return result.stdout.strip()

    @staticmethod
    def _get_local_volume_create_timeout(limit: int | None, requested_timeout: int) -> int:
        if requested_timeout == 0:
            return requested_timeout
        if not limit or limit <= _LOCAL_VOLUME_TIMEOUT_THRESHOLD_GB:
            return requested_timeout

        scaled_timeout = _LOCAL_VOLUME_TIMEOUT_BASE_SEC + (
            (limit + _LOCAL_VOLUME_TIMEOUT_GB_PER_SEC - 1)
            // _LOCAL_VOLUME_TIMEOUT_GB_PER_SEC
        )
        return max(requested_timeout, min(scaled_timeout, _LOCAL_VOLUME_TIMEOUT_MAX_SEC))

    async def create_local_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        local_volume: str,
        log_tag: str,
        log_text: str,
        log_extra: dict,
        limit: int | None = None,
        timeout: int = 10,
    ):
        requested_timeout = timeout
        if limit:
            # install loopback plugin
            loopback_plugin_name = "vloopback"

            docker_root_dir = await self.get_docker_root_dir(ssh_client)
            logger.info(_m(f"Docker data root: {docker_root_dir}", extra=get_extra_info(log_extra)))

            command = f'/usr/bin/docker plugin install ashald/docker-volume-loopback --alias {loopback_plugin_name} --grant-all-permissions DATA_DIR="{docker_root_dir}/loopback"'
            await ssh_client.run(command)

            command = f'/usr/bin/docker volume create -d {loopback_plugin_name} {local_volume} -o size={limit}g'
        else:
            loopback_plugin_name = None
            command = f"/usr/bin/docker volume create {local_volume}"

        timeout = self._get_local_volume_create_timeout(limit, timeout)
        volume_log_extra = {
            **log_extra,
            "local_volume": local_volume,
            "volume_limit_gb": limit,
            "requested_timeout_seconds": requested_timeout,
            "effective_timeout_seconds": timeout,
            "timeout_scaled": timeout != requested_timeout,
            "loopback_plugin": loopback_plugin_name,
        }
        logger.info(
            _m(
                "Preparing local Docker volume creation",
                extra=get_extra_info(volume_log_extra),
            )
        )
        await self.execute_and_stream_logs(
            ssh_client=ssh_client,
            command=command,
            log_tag=log_tag,
            log_text=log_text,
            log_extra=volume_log_extra,
            timeout=timeout,
        )

    async def create_container(
        self,
        payload: ContainerCreateRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        warnings = []
        local_volume = payload.local_volume
        external_volume_info = payload.external_volume_info

        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "workload_kind": payload.workload_kind.value,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
            "docker_image": payload.docker_image,
            "local_volume": local_volume,
            "edit_pod": True if local_volume else False,
            "external_volume": external_volume_info.name if external_volume_info else None,
            "enable_jupyter": payload.enable_jupyter,
        }

        # Deploy container profiler
        profilers = []
        if payload.timestamp:
            profilers.append({"name": "Requested from backend", "timestamp": payload.timestamp})
            prev_timestamp = payload.timestamp
        else:
            prev_timestamp = int(datetime.utcnow().timestamp() * 1000)
        profilers.append({"name": "Started in subnet", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
        prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

        logger.info(
            _m(
                "Edit Docker Container" if local_volume else "Create Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        log_tag = "container_creation"

        try:
            custom_options = CustomOptions.sanitize(payload.custom_options)
            # generate port maps
            port_maps, jupyter_port_map = await self.generate_portMappings(
                payload.miner_hotkey,
                payload.executor_id,
                UUID(payload.pod_id),
                custom_options.internal_ports,
                custom_options.initial_port_count,
                payload.enable_jupyter,
                payload.available_ports,
                payload.pod_mapping,
                payload.workload_kind,
            )

            # Add profiler for port mappings generation
            profilers.append({"name": "Port mappings generated", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
            prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

            if not port_maps:
                log_text = _m(
                    "No port mappings found",
                    extra=get_extra_info(default_extra),
                )
                logger.error(log_text)

                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.NoPortMappings,
                )

            default_extra = {
                **default_extra,
                "jupyter_port_map": jupyter_port_map,
            }

            if payload.enable_jupyter and not jupyter_port_map:
                log_text = _m(
                    "No Jupyter port mapping found",
                    extra=get_extra_info(default_extra),
                )
                logger.error(log_text)

                # Port release now handled by backend

                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.NoJupyterPortMapping,
                )

            if not payload.user_public_keys:
                log_text = _m(
                    "No public keys",
                    extra=get_extra_info(default_extra),
                )
                logger.error(log_text)

                # Port release now handled by backend

                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.NoSshKeys,
                )

            # add executor in pending status dict
            await self.redis_service.add_pending_pod(payload.miner_hotkey, payload.executor_id, payload.pod_id)

            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            pkey = asyncssh.import_private_key(private_key)

            known_hosts_policy: asyncssh.SSHKnownHosts | None = None
            try:
                known_hosts_policy = await self._prepare_known_hosts_policy(
                    executor_info,
                    payload.miner_hotkey,
                    default_extra,
                )
            except AttestationError as exc:
                log_text = _m(
                    "Attestation failed",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                )
                logger.error(log_text)
                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.UnknownError,
                )

            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                # Add profiler for ssh connection
                profilers.append({"name": "SSH connection established", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                # set real-time logging
                self.log_task = asyncio.create_task(
                    self.handle_stream_logs(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        pod_id=payload.pod_id,
                    )
                )
                # command = f"/usr/bin/docker logout"
                # await self.execute_and_stream_logs(
                #     ssh_client=ssh_client,
                #     command=command,
                #     log_tag=log_tag,
                #     log_text=f"Logging out of Docker registry",
                #     log_extra=default_extra,
                # )
                if payload.docker_username and payload.docker_password:
                    command = self._build_docker_login_command(
                        payload.docker_username, payload.docker_password
                    )
                    await self.execute_and_stream_logs(
                        ssh_client=ssh_client,
                        command=command,
                        log_tag=log_tag,
                        log_text=f"Logging in to Docker registry as {payload.docker_image}",
                        log_extra=default_extra,
                        raise_exception=False
                    )

                # Add profiler for docker login
                profilers.append({"name": "Docker login step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                command = f"/usr/bin/docker pull {payload.docker_image}"
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text=f"Pulling docker image {payload.docker_image}",
                    log_extra=default_extra,
                )

                # Add profiler for docker pull
                profilers.append({"name": "Docker pull step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                port_flags = " ".join(
                    [
                        f"-p {internal_port}:{docker_port}"
                        for docker_port, internal_port, _ in port_maps
                    ]
                )

                # Get the container path from the first volume
                local_volume_path = custom_options.volumes[0].split(':')[-1] if custom_options.volumes else '/root'
                entrypoint_flag = (
                    f"--entrypoint {custom_options.entrypoint}"
                    if custom_options
                    and custom_options.entrypoint
                    and custom_options.entrypoint.strip()
                    else ""
                )
                shm_size_flag = (
                    f"--shm-size {custom_options.shm_size}"
                    if custom_options and custom_options.shm_size
                    else ""
                )
                env_flags = (
                    " ".join(
                        [
                            f"-e '{key}={value}'"
                            for key, value in custom_options.environment.items()
                            if key and value and key.strip() and value.strip()
                        ]
                        + ["-e NVIDIA_DRIVER_CAPABILITIES=all"]
                    )
                    if custom_options and custom_options.environment
                    else "-e NVIDIA_DRIVER_CAPABILITIES=all"
                )
                startup_commands = (
                    f"{custom_options.startup_commands}"
                    if custom_options
                    and custom_options.startup_commands
                    and custom_options.startup_commands.strip()
                    else ""
                )

                container_name = self.get_container_name(payload)
                created_local_volume = False
                protected_volume_names = set(payload.active_volume_names or [])
                if local_volume:
                    protected_volume_names.add(local_volume)

                await self.clean_existing_containers(
                    ssh_client=ssh_client,
                    default_extra=default_extra,
                    pod_name=container_name,
                    sleep=10,
                    clear_volume=False if local_volume else True,
                    active_container_names=payload.active_container_names,
                    active_volume_names=payload.active_volume_names,
                )

                await self.clean_stale_vloopback_volumes(
                    ssh_client=ssh_client,
                    default_extra=default_extra,
                    skip_volume_names=protected_volume_names,
                )

                # Add profiler for docker volume creation
                profilers.append({"name": "Container cleaning step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                if not local_volume:
                    # create docker volume
                    local_volume = f"volume_{payload.pod_id}"
                    await self.create_local_volume(
                        ssh_client=ssh_client,
                        local_volume=local_volume,
                        log_tag=log_tag,
                        log_text=f"Creating docker volume {local_volume}",
                        log_extra=default_extra,
                        limit=payload.volume_limit_gb,
                    )
                    created_local_volume = True

                volume_flag = f"-v {local_volume}:{local_volume_path}"

                if external_volume_info:
                    success, msg = await self.create_s3fs_volume(
                        ssh_client=ssh_client,
                        log_extra=default_extra,
                        volume_info=external_volume_info,
                        log_tag=log_tag,
                    )
                    if success:
                        # Add profiler for docker volume creation
                        profilers.append({"name": "Docker volume creation step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                        prev_timestamp = int(datetime.utcnow().timestamp() * 1000)
                        # Important: disable sysbox when using s3fs volume because s3fs volume is not supported by sysbox
                        payload.is_sysbox = False

                        volume_flag += f" -v {external_volume_info.name}:/mnt"
                    else:
                        warnings.append(ContainerWarningCode.ExternalVolumeFailed)
                        profilers.append({"name": "Docker volume creation step failed", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                        await self.stream_log("S3 volume setup failed", "error", log_tag)

                # Network permission flags (permission to create a network interface inside the container)
                net_perm_flags = (
                    "--cap-add=NET_ADMIN "
                    "--sysctl net.ipv4.conf.all.src_valid_mark=1 "
                    "--device /dev/net/tun "
                )

                # GPU flags. --gpus injects userspace libs (libnvidia-ml.so, nvidia-smi);
                # explicit --device entries persist the device cgroup across systemd
                # daemon-reload (cgroup v2 + systemd cgroup driver wipe the transient
                # nvidia hook program; HostConfig.Devices is reapplied by Docker).
                gpu_flags = await build_gpu_flags(ssh_client, payload.gpu_uuids) + " "

                # CPU and memory restriction flags
                # --cpus flag isn't working inside cvm. skip to use it when tdx_quote is present
                # TODO: remove this when cvm is fixed
                if executor_info.tdx_quote: 
                    cpu_flag = ""
                else:
                    cpu_flag = f"--cpus {payload.cpu_count} " if payload.cpu_count else ""
                memory_flag = f"--memory {payload.memory_gb}g " if payload.memory_gb else ""
                
                storage_flag = f"--storage-opt size={payload.storage_limit_gb}g " if payload.storage_limit_gb else ""

                command = (
                    f'/usr/bin/docker run -d '
                    f'{"--runtime=sysbox-runc " if payload.is_sysbox else ""}'
                    f'{net_perm_flags} '  # Network permission flags
                    f'{port_flags} '
                    f'{volume_flag} '
                    f'{entrypoint_flag} '
                    f'{env_flags} '
                    f'{shm_size_flag} '
                    f'{gpu_flags} '  # GPU restriction flags
                    f'{cpu_flag} '  # CPU restriction flags
                    f'{memory_flag} '  # Memory restriction flags
                    f'{storage_flag} '  # Storage restriction flags
                    f'--restart unless-stopped '
                    f'--name {container_name} '
                    f'{payload.docker_image} '
                    f'{startup_commands}'
                )

                timeout = 120
                logger.info(f"Running command: {command} with timeout={timeout}")

                # DAH-2018: re-check for backend health_check_* / validator
                # port-test containers immediately before `docker run`. The
                # early check in miner_service runs before the image pull, but
                # the backend's RentalVerificationCheck can spin up a
                # health_check container during the pull window and grab a
                # host port from the same verified-port pool the rental
                # allocated. Reuse the open ssh_client so we don't pay the
                # cost of a second connect (and don't widen the TOCTOU gap).
                # Tighter budget than the early call: by this point HC should
                # be near completion, and the port-allocated retry loop +
                # `docker rm -f` are the backstop for any residual race.
                wait_ok, wait_msg = await self.wait_for_port_check_containers(
                    executor_info=executor_info,
                    miner_hotkey=payload.miner_hotkey,
                    keypair=keypair,
                    private_key=private_key,
                    max_retries=1,
                    retry_delay=30,
                    ssh_client=ssh_client,
                )
                logger.info(
                    _m(
                        f"Port check container pre-run wait result: {wait_msg}",
                        extra=get_extra_info({**default_extra, "ok": wait_ok}),
                    )
                )

                try:
                    await self._run_docker_create_with_port_retry(
                        ssh_client=ssh_client,
                        command=command,
                        container_name=container_name,
                        log_tag=log_tag,
                        default_extra=default_extra,
                        timeout=timeout,
                    )

                    logger.info(f"Container creation step finished")

                    # check if the container is running correctly
                    if not await self.check_container_running(ssh_client, container_name):
                        # Capture the failure reason and check whether it points to our
                        # --device flags (DAH-1987). State.Error covers cgroup / device
                        # failures; logs --tail covers entrypoint failures.
                        failure_reason = ""
                        try:
                            inspect = await ssh_client.run(
                                f"/usr/bin/docker inspect -f '{{{{.State.Error}}}}' {container_name}"
                            )
                            logs_tail = await ssh_client.run(
                                f"/usr/bin/docker logs --tail 50 {container_name} 2>&1 || true"
                            )
                            failure_reason = (inspect.stdout or "") + "\n" + (logs_tail.stdout or "")
                        except Exception:
                            failure_reason = "(failure_reason capture failed)"

                        nvidia_signal = any(
                            marker in failure_reason.lower()
                            for marker in ("/dev/nvidia", "device cgroup", "no such device", "operation not permitted")
                        )
                        log_extra = get_extra_info({
                            **default_extra,
                            "container_name": container_name,
                            "gpu_flags": gpu_flags,
                            "failure_reason": failure_reason[:2000],
                        })
                        if nvidia_signal:
                            logger.error(_m(
                                "docker run failed with NVIDIA-device-related error — "
                                "possible regression from build_gpu_flags --device flag set",
                                extra=log_extra,
                            ))
                        else:
                            logger.error(_m("docker run failed", extra=log_extra))

                        raise Exception("Run docker run command but container is not running")
                except Exception:
                    await self.cleanup_failed_container_creation(
                        ssh_client=ssh_client,
                        default_extra=default_extra,
                        container_name=container_name,
                        volume_name=local_volume,
                        remove_volume=created_local_volume,
                    )
                    raise

                # Add profiler for docker container creation
                profilers.append({"name": "Docker container creation step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                logger.info(
                    _m(
                        "Created Docker Container",
                        extra=get_extra_info({**default_extra, "container_name": container_name}),
                    ),
                )

                await self.stream_log("Created Docker Container", "success", log_tag)

                # skip installing ssh service for daturaai images
                # if payload.docker_image.startswith("daturaai/"):
                #     logger.info(
                #         _m(
                #             "Skipping checking install and start ssh service for daturaai images",
                #             extra=get_extra_info({**default_extra, "container_name": container_name}),
                #         ),
                #     )
                # else:
                try:
                    await self.install_open_ssh_server_and_start_ssh_service(
                        ssh_client=ssh_client,
                        container_name=container_name,
                        log_tag=log_tag,
                        log_extra=default_extra,
                    )

                    jupyter_url = None
                    if payload.enable_jupyter and jupyter_port_map:
                        jupyter_token = secrets.token_hex(16)
                        await self.run_jupyter(
                            ssh_client=ssh_client,
                            container_name=container_name,
                            jupyter_token=jupyter_token,
                            jupyter_port=jupyter_port_map[0],
                            log_tag=log_tag,
                            log_extra=default_extra,
                            local_volume=local_volume,
                            local_volume_path=local_volume_path,
                        )
                        jupyter_url = f"http://{executor_info.address}:{jupyter_port_map[1]}/lab?token={jupyter_token}"

                    # Add profiler for ssh service installation
                    profilers.append({"name": "SSH service installation step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                    prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                    # add rest of public keys
                    for public_key in payload.user_public_keys:
                        command = f"/usr/bin/docker exec {container_name} sh -c 'echo \"{public_key}\" >> ~/.ssh/authorized_keys'"
                        await ssh_client.run(command)

                    # add environment variables
                    if custom_options and custom_options.environment:
                        for k, v in custom_options.environment.items():
                            if k and v and k.strip() and str(v).strip():
                                env_line = f"{k}={v}"
                                # Execute each variable addition separately for better error handling
                                script = f'printf "%s\\n" {shlex.quote(env_line)} >> /etc/environment'
                                command = f"/usr/bin/docker exec {container_name} sh -c {shlex.quote(script)}"
                                try:
                                    await ssh_client.run(command)
                                except Exception as e:
                                    print(f"Failed to set environment variable {k}: {e}")

                    # Add profiler for adding public keys
                    profilers.append({"name": "Adding public keys step finished", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})
                    prev_timestamp = int(datetime.utcnow().timestamp() * 1000)

                    await self.finish_stream_logs()

                    await self.redis_service.add_rented_pod(executor_info, payload.pod_id, container_name)
                except Exception:
                    await self.cleanup_failed_container_creation(
                        ssh_client=ssh_client,
                        default_extra=default_extra,
                        container_name=container_name,
                        volume_name=local_volume,
                        remove_volume=created_local_volume,
                    )
                    raise

                # Add profiler for ssh service installation
                profilers.append({"name": "Finished in subnet.", "duration": int(datetime.utcnow().timestamp() * 1000) - prev_timestamp})

                if payload.workload_kind == WorkloadKind.FILLER:
                    await self.redis_service.remove_pending_pod(
                        payload.miner_hotkey,
                        payload.executor_id,
                        payload.pod_id,
                    )

                return ContainerCreated(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    container_name=container_name,
                    volume_name=local_volume,
                    port_maps=[
                        (docker_port, external_port) for docker_port, _, external_port in port_maps
                    ],
                    profilers=profilers,
                    backup_log_id=payload.backup_log_id,
                    restore_path=payload.restore_path,
                    jupyter_url=jupyter_url,
                    warnings=warnings,
                    storage_limit_gb=payload.storage_limit_gb,
                    volume_limit_gb=payload.volume_limit_gb,
                    local_volume_path=local_volume_path,
                )
        except Exception as e:
            log_text = _m(
                "Failed create_container",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            await self.finish_stream_logs()
            await self.redis_service.remove_pending_pod(payload.miner_hotkey, payload.executor_id, payload.pod_id)

            # Port release now handled by backend

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    async def stream_log(self, log_msg:str, log_status: str, log_tag: str):
        async with self.lock:
            self.logs_queue.append(
                {
                    "log_text": log_msg,
                    "log_status": log_status,
                    "log_tag": log_tag,
                }
            )

    async def stop_container(
        self,
        payload: ContainerStopRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Stop Docker Container", extra=get_extra_info({**default_extra, "payload": str(payload)})
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStopFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=known_hosts_policy,
        ) as ssh_client:
            await ssh_client.run(f"/usr/bin/docker stop {payload.container_name}")

            logger.info(
                _m(
                    "Stopped Docker Container",
                    extra=get_extra_info(
                        {**default_extra, "container_name": payload.container_name}
                    ),
                ),
            )

    async def start_container(
        self,
        payload: ContainerStartRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Restart Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStartFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=known_hosts_policy,
        ) as ssh_client:
            await ssh_client.run(f"/usr/bin/docker start {payload.container_name}")
            ssh_bootstrap_ok = await self.install_open_ssh_server_and_start_ssh_service(
                ssh_client=ssh_client,
                container_name=payload.container_name,
                log_tag=f"start_container_{payload.pod_id}",
                log_extra=default_extra,
            )
            if not ssh_bootstrap_ok:
                logger.warning(
                    _m(
                        "Docker container started but SSH bootstrap did not complete cleanly",
                        extra=get_extra_info(
                            {**default_extra, "container_name": payload.container_name}
                        ),
                    )
                )
            logger.info(
                _m(
                    "Started Docker Container",
                    extra=get_extra_info(
                        {**default_extra, "container_name": payload.container_name}
                    ),
                ),
            )

    async def delete_container(
        self,
        payload: ContainerDeleteRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "pod_id": payload.pod_id,
            "workload_kind": payload.workload_kind.value,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Deleting Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                # await ssh_client.run(f"docker stop {payload.container_name}")
                command = f"/usr/bin/docker rm -fv {payload.container_name}"
                await retry_ssh_command(ssh_client, command, "delete_container", 3, 5)

                command = f"/usr/bin/docker image prune -f"
                await retry_ssh_command(ssh_client, command, "delete_container", 3, 5)

                if payload.local_volume:
                    command = f"/usr/bin/docker volume rm {payload.local_volume}"
                    await ssh_client.run(command)

                if payload.external_volume:
                    command = f"/usr/bin/docker volume rm {payload.external_volume}"
                    await ssh_client.run(command)
                    await self.disable_s3fs_volume_plugin(ssh_client)

                logger.info(
                    _m(
                        "Remove rented machine from redis",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "container_name": payload.container_name,
                                "local_volume": payload.local_volume,
                                "external_volume": payload.external_volume,
                            }
                        ),
                    ),
                )

                await self.redis_service.remove_rented_machine(executor_info, payload.container_name)

                # Port release now handled by backend

                logger.info(
                    _m(
                        "Deleted Docker Container",
                        extra=get_extra_info({**default_extra, "payload": str(payload)}),
                    ),
                )

                return ContainerDeleted(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                )
        except Exception as e:
            log_text = _m(
                "Unknown Error delete_container",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    async def install_jupyter_server(
        self,
        payload: InstallJupyterServerRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "pod_id": payload.pod_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Install Jupyter server on pod",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                jupyter_token = secrets.token_hex(16)
                jupyter_port = payload.jupyter_port_map[0]
                local_volume = payload.local_volume
                local_volume_path = payload.local_volume_path
                await self.run_jupyter(
                    ssh_client=ssh_client,
                    container_name=payload.container_name,
                    jupyter_token=jupyter_token,
                    jupyter_port=jupyter_port,
                    log_tag="jupyter",
                    log_extra=default_extra,
                    local_volume=local_volume,
                    local_volume_path=local_volume_path,
                )

                logger.info(
                    _m(
                        "Jupyter server installed",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name,
                            "jupyter_token": jupyter_token,
                            "jupyter_port": jupyter_port,
                        }),
                    ),
                )

                return JupyterServerInstalled(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    jupyter_url=f"http://{executor_info.address}:{payload.jupyter_port_map[1]}/lab?token={jupyter_token}",
                )
        except Exception as e:
            log_text = _m(
                "Failed install jupyter server",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return JupyterInstallationFailed(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
            )

    async def remove_ssh_keys(
        self,
        payload: RemoveSshPublicKeysRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "pod_id": payload.pod_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Remove ssh key(s) from pod",
                extra=get_extra_info({**default_extra}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                if not payload.user_public_keys:
                    log_text = _m(
                        "ssh key Remove error: no public key",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name,
                            "error": "No public keys",
                        }),
                    )
                    logger.error(log_text)

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        pod_id=payload.pod_id,
                        workload_kind=payload.workload_kind,
                        msg=str(log_text),
                        error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                        error_code=FailedContainerErrorCodes.NoSshKeys,
                    )

                # Remove each public key from authorized_keys
                for pubkey in payload.user_public_keys:
                    # Remove the public key from authorized_keys
                    # Properly escape slashes and pluses in pubkey for sed
                    # Use Python's shlex.quote to safely quote the pubkey for shell usage
                    import shlex

                    # Remove the public key from authorized_keys by matching the exact line
                    # This approach is safer and more reliable than trying to escape characters for sed
                    quoted_pubkey = shlex.quote(pubkey)
                    remove_cmd = (
                        f"/usr/bin/docker exec -i {payload.container_name} "
                        f"sh -c \"grep -vxF {quoted_pubkey} /root/.ssh/authorized_keys > /root/.ssh/authorized_keys.tmp && "
                        f"mv /root/.ssh/authorized_keys.tmp /root/.ssh/authorized_keys\""
                    )
                    await ssh_client.run(remove_cmd)

                logger.info(
                    _m(
                        "Removed ssh key(s) from the container",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name,
                            "removed_keys": payload.user_public_keys,
                        }),
                    ),
                )

                return SshPubKeyRemoved(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    user_public_keys=payload.user_public_keys,
                )
        except Exception as e:
            log_text = _m(
                "Unknown Error remove_ssh_keys",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    async def add_ssh_key(
        self,
        payload: AddSshPublicKeyRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Add ssh key to pod",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                if not payload.user_public_keys:
                    log_text = _m(
                        "ssh key Add error: no public key",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name,
                            "error": "No public keys",
                        }),
                    )
                    logger.error(log_text)

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        pod_id=payload.pod_id,
                        workload_kind=payload.workload_kind,
                        msg=str(log_text),
                        error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                        error_code=FailedContainerErrorCodes.NoSshKeys,
                    )

                for public_key in payload.user_public_keys:
                    command = f"/usr/bin/docker exec -i {payload.container_name} sh -c 'echo \"{public_key}\" >> ~/.ssh/authorized_keys'"
                    await retry_ssh_command(ssh_client, command, "add_ssh_key", 3, 5)

                logger.info(
                    _m(
                        "Added ssh key into Docker Container",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name
                        }),
                    ),
                )

                return SshPubKeyAdded(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    user_public_keys=payload.user_public_keys,
                )
        except Exception as e:
            log_text = _m(
                "Failed add_ssh_key",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    async def get_docker_hub_digests(self, repositories) -> dict[str, str]:
        """Retrieve all tags and their corresponding digests from Docker Hub."""
        all_digests = {}  # Initialize a dictionary to store all tag-digest pairs

        async with aiohttp.ClientSession() as session:
            for repo in repositories:
                try:
                    # Split repository and tag if specified
                    if ":" in repo:
                        repository, specified_tag = repo.split(":", 1)
                    else:
                        repository, specified_tag = repo, None

                    # Get authorization token
                    async with session.get(
                        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull"
                    ) as token_response:
                        token_response.raise_for_status()
                        token = await token_response.json()
                        token = token.get("token")

                    # Find all tags if no specific tag is specified
                    if specified_tag is None:
                        async with session.get(
                            f"https://index.docker.io/v2/{repository}/tags/list",
                            headers={"Authorization": f"Bearer {token}"},
                        ) as tags_response:
                            tags_response.raise_for_status()
                            tags_data = await tags_response.json()
                            all_tags = tags_data.get("tags", [])
                    else:
                        all_tags = [specified_tag]

                    # Dictionary to store tag-digest pairs for the current repository
                    tag_digests = {}
                    for tag in all_tags:
                        # Get image digest
                        async with session.head(
                            f"https://index.docker.io/v2/{repository}/manifests/{tag}",
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                            },
                        ) as manifest_response:
                            manifest_response.raise_for_status()
                            digest = manifest_response.headers.get("Docker-Content-Digest")
                            tag_digests[f"{repository}:{tag}"] = digest

                    # Update the all_digests dictionary with the current repository's tag-digest pairs
                    all_digests.update(tag_digests)

                except aiohttp.ClientError as e:
                    print(f"Error retrieving data for {repo}: {e}")

        return all_digests

    async def setup_ssh_access(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        ip_address: str,
        username: str = "root",
        port_maps: list[tuple[int, int]] = None,
    ) -> tuple[bool, str, str]:
        """Generate an SSH key pair, add the public key to the Docker container, and check SSH connection."""

        my_key = "my_key"
        private_key, public_key = self.ssh_service.generate_ssh_key(my_key)

        public_key = public_key.decode("utf-8")
        private_key = private_key.decode("utf-8")

        private_key = self.ssh_service.decrypt_payload(my_key, private_key)
        pkey = asyncssh.import_private_key(private_key)

        await asyncio.sleep(5)

        command = f"/usr/bin/docker exec {container_name} sh -c 'echo \"{public_key}\" >> /root/.ssh/authorized_keys'"

        result = await ssh_client.run(command)
        if result.exit_status != 0:
            log_text = "Error creating docker connection"
            log_status = "error"
            logger.error(log_text)

            return False, log_text, log_status

        port = 0
        for internal, external in port_maps:
            if internal == 22:
                port = external
        # Check SSH connection
        try:
            async with asyncssh.connect(
                host=ip_address,
                port=port,
                username=username,
                client_keys=[pkey],
                known_hosts=None,
            ):
                log_status = "info"
                log_text = "SSH connection successful!"
                logger.info(
                    _m(
                        log_text,
                        extra={
                            "container_name": container_name,
                            "ip_address": ip_address,
                            "port_maps": port_maps,
                        },
                    )
                )
                return True, log_text, log_status
        except Exception as e:
            log_text = "SSH connection failed"
            log_status = "error"
            logger.error(
                _m(
                    log_text,
                    extra={
                        "container_name": container_name,
                        "ip_address": ip_address,
                        "port_maps": port_maps,
                        "error": str(e),
                    },
                )
            )
            return False, log_text, log_status

    def _get_preferred_ports(self, initial_port_count: int | None) -> list[int]:
        """Calculate preferred ports based on initial_port_count.

        - None: return all PREFERRED_POD_PORTS
        - Less than PREFERRED_POD_PORTS length: return limited list
        - More than PREFERRED_POD_PORTS length: return PREFERRED_POD_PORTS + sequential extras
        """
        if initial_port_count is None:
            return PREFERRED_POD_PORTS

        if initial_port_count <= len(PREFERRED_POD_PORTS):
            return PREFERRED_POD_PORTS[:initial_port_count]

        # Need more ports than available in PREFERRED_POD_PORTS
        max_port = max(PREFERRED_POD_PORTS)
        extra_count = initial_port_count - len(PREFERRED_POD_PORTS)
        extra_ports = [max_port + i for i in range(extra_count)]

        return list(PREFERRED_POD_PORTS) + extra_ports
