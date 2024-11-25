import logging
import random
import aiohttp
from typing import Annotated
from uuid import uuid4
import asyncio

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
)
from protocol.vc_protocol.compute_requests import RentedMachine

from core.utils import _m, context, get_extra_info
from services.redis_service import RedisService
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)

REPOSITORYS = [
    "daturaai/compute-subnet-executor:latest",
    "daturaai/compute-subnet-executor-runner:latest",
    "containrrr/watchtower:1.7.1",
    "daturaai/pytorch",
    "daturaai/ubuntu",
]


class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service

    def generate_portMappings(self, range_external_ports: str) -> list[tuple[int, int]]:
        internal_ports = [22, 22140, 22141, 22142, 22143]

        mappings = []
        used_external_ports = set()
        
        # Parse the range_external_ports
        if range_external_ports:
            if '-' in range_external_ports:
                # Handle range like "100-1500"
                start, end = map(int, range_external_ports.split('-'))
                available_ports = list(range(start, end + 1))
            else:
                # Handle list like "120,122,145"
                available_ports = list(map(int, range_external_ports.split(',')))
        else:
                # If empty, use a default range for random selection
                available_ports = list(range(40000, 65536))

        for i in range(len(internal_ports)):
            while True:
                if available_ports:
                    external_port = random.choice(available_ports)
                    available_ports.remove(external_port)
                else:
                    external_port = random.randint(40000, 65536)

                if external_port not in used_external_ports:
                    used_external_ports.add(external_port)
                    break

            mappings.append((internal_ports[i], external_port))

        return mappings

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

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:
            # check docker image exists
            logger.info(
                _m(
                    "Checking if docker image exists",
                    extra=get_extra_info({**default_extra, "docker_image": payload.docker_image}),
                ),
            )
            result = await ssh_client.run(f"docker inspect --type=image {payload.docker_image}")

            # Read the output and error streams
            output = result.stdout
            error = result.stderr

            if error:
                logger.info(
                    _m(
                        "Pulling docker image",
                        extra=get_extra_info(
                            {**default_extra, "docker_image": payload.docker_image}
                        ),
                    ),
                )
                result = await ssh_client.run(f"docker pull {payload.docker_image}")

                # Read the output and error streams
                output = result.stdout
                error = result.stderr

                # Log the output and error
                if output:
                    logger.info(
                        _m(
                            "Docker pull output",
                            extra=get_extra_info({**default_extra, "output": output}),
                        ),
                    )
                if error:
                    logger.error(
                        _m(
                            "Docker pull error",
                            extra=get_extra_info({**default_extra, "error": error}),
                        ),
                    )
            else:
                logger.info(
                    _m(
                        "Docker image already exists locally",
                        extra=get_extra_info(
                            {**default_extra, "docker_image": payload.docker_image}
                        ),
                    ),
                )

            # generate port maps
            port_maps = self.generate_portMappings(executor_info.port_range)
            port_flags = " ".join([f"-p {external}:{internal}" for internal, external in port_maps])

            # creat docker volume
            uuid = uuid4()
            volume_name = f"volume_{uuid}"
            await ssh_client.run(f"docker volume create {volume_name}")

            logger.info(
                _m(
                    "Created Docker Volume",
                    extra=get_extra_info({**default_extra, "volume_name": volume_name}),
                ),
            )

            # create docker container with the port map & resource
            container_name = f"container_{uuid}"
            if payload.debug:
                await ssh_client.run(
                    f'docker run -d {port_flags} -v "/var/run/docker.sock:/var/run/docker.sock" -e PUBLIC_KEY="{payload.user_public_key}" --mount source={volume_name},target=/root --name {container_name} {payload.docker_image}'
                )
            else:
                await ssh_client.run(
                    f'docker run -d {port_flags} -e PUBLIC_KEY="{payload.user_public_key}" --mount source={volume_name},target=/root --gpus all --name {container_name} {payload.docker_image}'
                )

            logger.info(
                _m(
                    "Created Docker Container",
                    extra=get_extra_info({**default_extra, "container_name": container_name}),
                ),
            )
            success, _, _ = await self.docker_connection_check(
                ssh_client, 
                payload.miner_hotkey, 
                executor_info, 
                private_key, 
                payload.user_public_key
            )
            await self.redis_service.add_rented_machine(
                RentedMachine(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    executor_ip_address=executor_info.address,
                    executor_ip_port=str(executor_info.port),
                )
            )

            return success, ContainerCreatedResult(
                container_name=container_name,
                volume_name=volume_name,
                port_maps=port_maps,
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
                "Stop Docker Container", extra=get_extra_info({**default_extra, "payload": payload})
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
                extra=get_extra_info({**default_extra, "payload": payload}),
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
            await ssh_client.run(f"docker stop {payload.container_name}")
            await ssh_client.run(f"docker rm {payload.container_name} -f")
            await ssh_client.run(f"docker volume rm {payload.volume_name}")

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
                    if ':' in repo:
                        repository, specified_tag = repo.split(':', 1)
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
                            headers={"Authorization": f"Bearer {token}"}
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
                                "Accept": "application/vnd.docker.distribution.manifest.v2+json"
                            }
                        ) as manifest_response:
                            manifest_response.raise_for_status()
                            digest = manifest_response.headers.get("Docker-Content-Digest")
                            tag_digests[f"{repository}:{tag}"] = digest

                    # Update the all_digests dictionary with the current repository's tag-digest pairs
                    all_digests.update(tag_digests)

                except aiohttp.ClientError as e:
                    print(f"Error retrieving data for {repo}: {e}")

        return all_digests

    def get_available_port(
        self,
        port_range: str,
    ) -> int:
        if port_range:
            if '-' in port_range:
                min_port, max_port = map(int, (part.strip() for part in port_range.split('-')))
                ports = range(min_port, max_port + 1)
            else:
                ports = list(map(int, (part.strip() for part in port_range.split(','))))
        else:
            # Default range if port_range is empty
            ports = range(40000, 65536)

        return random.choice(ports)
    
    async def docker_connection_check(
            self,
            ssh_client: asyncssh.SSHClientConnection,
            miner_hotkey: str,
            executor_info: ExecutorSSHInfo,
            private_key: str,
            public_key: str,
        ):
            ssh_port = self.get_available_port(executor_info.port_range)
            executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
            default_extra = {
                "miner_hotkey": miner_hotkey,
                "executor_uuid": executor_info.uuid,
                "executor_ip_address": executor_info.address,
                "executor_port": executor_info.port,
                "ssh_username": executor_info.ssh_username,
                "ssh_port": ssh_port,
            }
            context.set(f"[_docker_connection_check][{executor_name}]")

            try:
                log_text = _m(
                    "Creating docker container",
                    extra=default_extra,
                )
                log_status = "info"
                logger.info(log_text)

                container_name = f"container_{miner_hotkey}"
                docker_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
                command = f"docker run -d --name {container_name} -p {ssh_port}:22 daturaai/compute-subnet-executor:latest {docker_cmd}"

                result = await ssh_client.run(command, timeout=20)
                if result.exit_status != 0:
                    log_text = _m(
                        "Error creating docker connection",
                        extra=get_extra_info({**default_extra, "error": str(e)}),
                    )
                    log_status = "error"
                    logger.error(log_text, exc_info=True)

                    return False, log_text, log_status

                await asyncio.sleep(2)

                pkey = asyncssh.import_private_key(private_key)
                async with asyncssh.connect(
                    host=executor_info.address,
                    port=ssh_port,
                    username=executor_info.ssh_username,
                    client_keys=[pkey],
                    known_hosts=None,
                ) as _:
                    log_text = _m(
                        "Connected into docker container",
                        extra=default_extra,
                    )
                    logger.info(log_text)

                command = f"docker rm {container_name} -f"
                await ssh_client.run(command, timeout=20)

                return True, log_text, log_status
            except Exception as e:
                log_text = _m(
                    "Error connection docker container",
                    extra=get_extra_info({**default_extra, "error": str(e)}),
                )
                log_status = "error"
                logger.error(log_text, exc_info=True)

                try:
                    command = f"docker container stop {container_name} && docker container prune -f"
                    await ssh_client.run(command, timeout=20)
                except:
                    pass

                return False, log_text, log_status
