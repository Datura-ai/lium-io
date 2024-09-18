import io
import logging
import random
from typing import Annotated
from uuid import uuid4

import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from paramiko import AutoAddPolicy, Ed25519Key, SSHClient
from payload_models.payloads import (
    ContainerCreatedResult,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
)

from daos.executor import ExecutorDao
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)


class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        executor_dao: Annotated[ExecutorDao, Depends(ExecutorDao)],
    ):
        self.ssh_service = ssh_service
        self.executor_dao = executor_dao

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

    def get_ssh_client(
        self,
        ip_address: str,
        ssh_username: str,
        ssh_port: str,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info("Connect ssh")
        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = Ed25519Key.from_private_key(io.StringIO(private_key))

        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())
        ssh_client.connect(
            hostname=ip_address,
            username=ssh_username,
            look_for_keys=False,
            pkey=pkey,
            port=ssh_port,
        )

        return ssh_client

    def create_container(
        self,
        payload: ContainerCreateRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info(
            f"Create Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, escecutor: {executor_info}"
        )

        ssh_client = self.get_ssh_client(
            ip_address=executor_info.address,
            ssh_username=executor_info.ssh_username,
            ssh_port=executor_info.ssh_port,
            keypair=keypair,
            private_key=private_key,
        )

        # check docker image exists
        logger.info("Checking if docker image exists: %s", payload.docker_image)
        stdin, stdout, stderr = ssh_client.exec_command(
            f"docker inspect --type=image {payload.docker_image}"
        )

        # Read the output and error streams
        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")

        if not output:
            logger.info("Pulling docker image: %s", payload.docker_image)
            stdin, stdout, stderr = ssh_client.exec_command(f"docker pull {payload.docker_image}")

            # Read the output and error streams
            output = stdout.read().decode("utf-8")
            error = stderr.read().decode("utf-8")

            # Log the output and error
            if output:
                logger.info(f"Docker pull output: {output}")
            if error:
                logger.error(f"Docker pull error: {error}")
        else:
            logger.info(f"Docker image {payload.docker_image} already exists locally.")

        # generate port maps
        port_maps = self.generate_portMappings()
        port_flags = " ".join([f"-p {external}:{internal}" for internal, external in port_maps])
        logger.info(f"Port mappings: {port_maps}")

        # creat docker volume
        uuid = uuid4()
        logger.info("Create docker volume")
        volume_name = f"volume_{uuid}"
        ssh_client.exec_command(f"docker volume create {volume_name}")

        # creat docker container with the port map & resource
        logger.info("Create docker container")
        container_name = f"container_{uuid}"
        ssh_client.exec_command(
            f'docker run -d {port_flags} -e PUBLIC_KEY="{payload.user_public_key}" --mount source={volume_name},target=/root --gpus all --name {container_name} {payload.docker_image}'
        )

        ssh_client.close()

        self.executor_dao.rent(payload.executor_id, payload.miner_hotkey)

        return ContainerCreatedResult(
            container_name=container_name,
            volume_name=volume_name,
            port_maps=port_maps,
        )

    def stop_container(
        self,
        payload: ContainerStopRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info(
            f"Stop Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, executor: {executor_info}, container_name: {payload.container_name}"
        )

        ssh_client = self.get_ssh_client(
            ip_address=executor_info.address,
            ssh_username=executor_info.ssh_username,
            ssh_port=executor_info.ssh_port,
            keypair=keypair,
            private_key=private_key,
        )

        logger.info("stop container")
        ssh_client.exec_command(f"docker stop {payload.container_name}")

        ssh_client.close()

        return

    def start_container(
        self,
        payload: ContainerStartRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info(
            f"Restart Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, contaienr_name: {payload.container_name}"
        )

        ssh_client = self.get_ssh_client(
            ip_address=executor_info.address,
            ssh_username=executor_info.ssh_username,
            ssh_port=executor_info.ssh_port,
            keypair=keypair,
            private_key=private_key,
        )

        logger.info("stop container")
        ssh_client.exec_command(f"docker start {payload.container_name}")

        ssh_client.close()

        return

    def delete_container(
        self,
        payload: ContainerDeleteRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        logger.info(
            f"Restart Docker Container -> miner_address: {payload.miner_address}, miner_hotkey: {payload.miner_hotkey}, contaienr_name: {payload.container_name}"
        )

        ssh_client = self.get_ssh_client(
            ip_address=executor_info.address,
            ssh_username=executor_info.ssh_username,
            ssh_port=executor_info.ssh_port,
            keypair=keypair,
            private_key=private_key,
        )

        logger.info("delete container")
        ssh_client.exec_command(f"docker stop {payload.container_name}")
        ssh_client.exec_command(f"docker rm {payload.container_name} -f")

        logger.info("delete volume")
        ssh_client.exec_command(f"docker volume rm {payload.volume_name}")

        ssh_client.close()

        self.executor_dao.unrent(payload.executor_id, payload.miner_hotkey)

        return
