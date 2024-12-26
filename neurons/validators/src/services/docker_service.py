import asyncio
import logging
import time
from typing import Annotated
from uuid import uuid4

import aiohttp
import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import (
    ContainerCreatedResult,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
)
from protocol.vc_protocol.compute_requests import RentedMachine

from core.utils import _m, get_extra_info
from services.redis_service import (
    AVAILABLE_PORT_MAPS_PREFIX,
    STREAMING_LOG_CHANNEL,
    RedisService,
)
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)

REPOSITORIES = [
    "daturaai/compute-subnet-executor:latest",
    "daturaai/compute-subnet-executor-runner:latest",
    "containrrr/watchtower:1.7.1",
    "daturaai/pytorch",
    "daturaai/ubuntu",
]

LOG_STREAM_INTERVAL = 5  # 5 seconds


class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.lock = asyncio.Lock()
        self.logs_queue: list[dict] = []
        self.log_task: asyncio.Task | None = None
        self.is_realtime_logging = False

    async def generate_portMappings(self, miner_hotkey, executor_id, internal_ports=None):
        try:
            docker_internal_ports = [22, 20000, 20001, 20002, 20003]
            if internal_ports:
                docker_internal_ports = internal_ports

            key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_hotkey}:{executor_id}"
            available_port_maps = await self.redis_service.lrange(key)

            logger.info(f"available_port_maps: {key}, {available_port_maps}")

            mappings = []
            for i, docker_port in enumerate(docker_internal_ports):
                if i < len(available_port_maps):
                    internal_port, external_port = map(
                        int, available_port_maps[i].decode().split(",")
                    )
                    mappings.append((docker_port, internal_port, external_port))
                else:
                    break
            return mappings
        except Exception as e:
            logger.error(f"Error generating port mappings: {e}", exc_info=True)
            return []

    async def execute_and_stream_logs(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        command: str,
        log_tag: str,
    ):
        result = True
        async with ssh_client.create_process(command) as process:
            async for line in process.stdout:
                async with self.lock:
                    self.logs_queue.append(
                        {
                            "log_text": line.strip(),
                            "log_status": "success",
                            "log_tag": log_tag,
                        }
                    )

            async for line in process.stderr:
                result = False
                async with self.lock:
                    self.logs_queue.append(
                        {
                            "log_text": line.strip(),
                            "log_status": "error",
                            "log_tag": log_tag,
                        }
                    )

        return result

    async def handle_stream_logs(
        self,
        miner_hotkey,
        executor_id,
    ):
        default_extra = {
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_id,
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
            result = await ssh_client.run(f"docker ps -q -f name={container_name}")
            if result.stdout.strip():
                return True
            await asyncio.sleep(1)
        return False

    async def create_container(
        self,
        payload: ContainerCreateRequest,
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
            "debug": payload.debug,
        }

        logger.info(
            _m(
                "Create Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        custom_options = payload.custom_options

        try:
            # generate port maps
            if custom_options and custom_options.internal_ports:
                port_maps = await self.generate_portMappings(
                    payload.miner_hotkey, payload.executor_id, custom_options.internal_ports
                )
            else:
                port_maps = await self.generate_portMappings(
                    payload.miner_hotkey, payload.executor_id
                )

            if not port_maps:
                log_text = "No port mappings found"
                logger.error(log_text)
                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=str(log_text),
                    error_code=FailedContainerErrorCodes.NoPortMappings,
                )

            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            pkey = asyncssh.import_private_key(private_key)

            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as ssh_client:
                logger.info(
                    _m(
                        "Pulling docker image",
                        extra=get_extra_info(
                            {**default_extra, "docker_image": payload.docker_image}
                        ),
                    ),
                )

                log_tag = "container_creation"

                # set real-time logging
                self.log_task = asyncio.create_task(
                    self.handle_stream_logs(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                    )
                )

                async with self.lock:
                    self.logs_queue.append(
                        {
                            "log_text": f"Pulling docker image {payload.docker_image}",
                            "log_status": "success",
                            "log_tag": log_tag,
                        }
                    )

                command = f"docker pull {payload.docker_image}"
                result = await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                )
                if not result:
                    log_text = _m(
                        "Docker pull failed",
                        extra=get_extra_info({default_extra}),
                    )
                    logger.error(log_text)

                    await self.finish_stream_logs()

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )

                port_flags = " ".join(
                    [
                        f"-p {internal_port}:{docker_port}"
                        for docker_port, internal_port, _ in port_maps
                    ]
                )

                # Prepare extra options
                sanitized_volumes = [
                    volume for volume
                    in (custom_options.volumes if custom_options and custom_options.volumes else [])
                    if volume.strip()
                ]
                volume_flags = (
                    " ".join([f"-v {volume}" for volume in sanitized_volumes])
                    if sanitized_volumes
                    else ""
                )
                entrypoint_flag = (
                    f"--entrypoint {custom_options.entrypoint}"
                    if custom_options
                    and custom_options.entrypoint
                    and custom_options.entrypoint.strip()
                    else ""
                )
                env_flags = (
                    " ".join(
                        [f"-e {key}={value}" for key, value in custom_options.environment.items()]
                    )
                    if custom_options and custom_options.environment
                    else ""
                )
                startup_commands = (
                    f"{custom_options.startup_commands}"
                    if custom_options
                    and custom_options.startup_commands
                    and custom_options.startup_commands.strip()
                    else ""
                )

                uuid = uuid4()

                # creat docker volume
                async with self.lock:
                    self.logs_queue.append(
                        {
                            "log_text": "Creating docker volume",
                            "log_status": "success",
                            "log_tag": log_tag,
                        }
                    )

                volume_name = f"volume_{uuid}"
                command = f"docker volume create {volume_name}"
                result = await self.execute_and_stream_logs(
                    ssh_client=ssh_client, command=command, log_tag="container_creation"
                )
                if not result:
                    log_text = _m(
                        "Docker volume creation failed",
                        extra=get_extra_info({default_extra}),
                    )
                    logger.error(log_text)

                    await self.finish_stream_logs()

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )

                logger.info(
                    _m(
                        "Created Docker Volume",
                        extra=get_extra_info({**default_extra, "volume_name": volume_name}),
                    ),
                )

                # create docker container with the port map & resource
                async with self.lock:
                    self.logs_queue.append(
                        {
                            "log_text": "Creating docker container",
                            "log_status": "success",
                            "log_tag": log_tag,
                        }
                    )

                container_name = f"container_{uuid}"

                if payload.debug:
                    command = f'docker run -d {port_flags} -v "/var/run/docker.sock:/var/run/docker.sock" {volume_flags} {entrypoint_flag} -e PUBLIC_KEY="{payload.user_public_key}" {env_flags} --mount source={volume_name},target=/root --name {container_name} {payload.docker_image} {startup_commands}'
                else:
                    command = f'docker run -d {port_flags} {volume_flags} {entrypoint_flag} -e PUBLIC_KEY="{payload.user_public_key}" {env_flags} --mount source={volume_name},target=/root --gpus all --name {container_name}  {payload.docker_image} {startup_commands}'

                result = await self.execute_and_stream_logs(
                    ssh_client=ssh_client, command=command, log_tag="container_creation"
                )
                if not result:
                    log_text = _m(
                        "Docker container creation failed",
                        extra=get_extra_info(default_extra),
                    )
                    logger.error(log_text)

                    await self.finish_stream_logs()

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )

                # check if the container is running correctly
                if not await self.check_container_running(ssh_client, container_name):
                    log_text = _m(
                        "Run docker run command but container is not running",
                        extra=get_extra_info({**default_extra, "container_name": container_name}),
                    )
                    logger.error(log_text)

                    await self.finish_stream_logs()

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.ContainerNotRunning,
                    )

                logger.info(
                    _m(
                        "Created Docker Container",
                        extra=get_extra_info({**default_extra, "container_name": container_name}),
                    ),
                )

                await self.finish_stream_logs()

                await self.redis_service.add_rented_machine(
                    RentedMachine(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        executor_ip_address=executor_info.address,
                        executor_ip_port=str(executor_info.port),
                    )
                )

                return ContainerCreatedResult(
                    container_name=container_name,
                    volume_name=volume_name,
                    port_maps=[
                        (docker_port, external_port) for docker_port, _, external_port in port_maps
                    ],
                )
        except Exception as e:
            log_text = _m(
                "Docker container creation failed",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            await self.finish_stream_logs()

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                msg=str(log_text),
                error_code=FailedContainerErrorCodes.UnknownError,
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

        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:
            await ssh_client.run(f"docker stop {payload.container_name}")

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

        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:
            await ssh_client.run(f"docker start {payload.container_name}")
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
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Delete Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:
            # await ssh_client.run(f"docker stop {payload.container_name}")
            await ssh_client.run(f"docker rm {payload.container_name} -f")
            await ssh_client.run(f"docker volume rm {payload.volume_name} -f")

            logger.info(
                _m(
                    "Deleted Docker Container",
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "container_name": payload.container_name,
                            "volume_name": payload.volume_name,
                        }
                    ),
                ),
            )

            await self.redis_service.remove_rented_machine(
                RentedMachine(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    executor_ip_address=executor_info.address,
                    executor_ip_port=str(executor_info.port),
                )
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

        command = f"docker exec {container_name} sh -c 'echo \"{public_key}\" >> /root/.ssh/authorized_keys'"

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
