import logging
import random
from typing import Annotated
from uuid import uuid4

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

from core.utils import _m, get_extra_info
from services.redis_service import RedisService
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)


class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service

    def generate_portMappings(self, start_external_port=40000) -> list[tuple[int, int]]:
        internal_ports = [22, 22140, 22141, 22142, 22143]

        mappings = []
        used_external_ports = set()

        for i in range(len(internal_ports)):
            while True:
                external_port = random.randint(start_external_port, start_external_port + 10000)
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
            port_maps = self.generate_portMappings()
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
            await ssh_client.run(
                f'docker run -d {port_flags} -e PUBLIC_KEY="{payload.user_public_key}" --mount source={volume_name},target=/root --gpus all --name {container_name} {payload.docker_image}'
            )

            logger.info(
                _m(
                    "Created Docker Container",
                    extra=get_extra_info({**default_extra, "container_name": container_name}),
                ),
            )

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
