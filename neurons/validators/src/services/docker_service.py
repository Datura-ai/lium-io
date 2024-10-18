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

from services.ssh_service import SSHService
from services.redis_service import RedisService

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
        logger.info(
            f"[create_container][{payload.executor_id}] Create Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, executor: {executor_info}"
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
                f"[create_container][{payload.executor_id}] Checking if docker image exists: {payload.docker_image}"
            )
            result = await ssh_client.run(f"docker inspect --type=image {payload.docker_image}")

            # Read the output and error streams
            output = result.stdout
            error = result.stderr

            if error:
                logger.info(
                    f"[create_container][{payload.executor_id}] Pulling docker image: {payload.docker_image}"
                )
                result = await ssh_client.run(f"docker pull {payload.docker_image}")

                # Read the output and error streams
                output = result.stdout
                error = result.stderr

                # Log the output and error
                if output:
                    logger.info(
                        f"[create_container][{payload.executor_id}] Docker pull output: {output}"
                    )
                if error:
                    logger.error(
                        f"[create_container][{payload.executor_id}] Docker pull error: {error}"
                    )
            else:
                logger.info(
                    f"[create_container][{payload.executor_id}] Docker image {payload.docker_image} already exists locally."
                )

            # generate port maps
            port_maps = self.generate_portMappings()
            port_flags = " ".join([f"-p {external}:{internal}" for internal, external in port_maps])
            logger.info(f"Port mappings: {port_maps}")

            # creat docker volume
            uuid = uuid4()
            volume_name = f"volume_{uuid}"
            await ssh_client.run(f"docker volume create {volume_name}")

            # creat docker container with the port map & resource
            container_name = f"container_{uuid}"
            await ssh_client.run(
                f'docker run -d {port_flags} -e PUBLIC_KEY="{payload.user_public_key}" --mount source={volume_name},target=/root --gpus all --name {container_name} {payload.docker_image}'
            )

            # await self.executor_dao.rent(payload.executor_id, payload.miner_hotkey)

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
        logger.info(
            f"[stop_container][{payload.executor_id}] Stop Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, executor: {executor_info}, container_name: {payload.container_name}"
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

            return

    async def start_container(
        self,
        payload: ContainerStartRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info(
            f"[start_container][{payload.executor_id}] Restart Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, contaienr_name: {payload.container_name}"
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

            return

    async def delete_container(
        self,
        payload: ContainerDeleteRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info(
            f"[delete_container][{payload.executor_id}] Restart Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, contaienr_name: {payload.container_name}"
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

            # await self.executor_dao.unrent(payload.executor_id, payload.miner_hotkey)

            return
