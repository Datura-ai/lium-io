import asyncio
import random
import time
import uuid
import resource

import click
from datura.requests.miner_requests import ExecutorSSHInfo

from core.utils import configure_logs_of_other_modules, _m, get_extra_info, get_logger
from core.validator import Validator
from clients.subtensor_client import SubtensorClient
from services.ioc import ioc
from services.miner_service import MinerService
from services.docker_service import DockerService
from services.file_encrypt_service import FileEncryptService
from payload_models.payloads import (
    MinerJobRequestPayload,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    AddSshPublicKeyRequest,
    CustomOptions,
    GetPodLogsRequestFromServer,
    AddDebugSshKeyRequest,
    InstallJupyterServerRequest,
)

configure_logs_of_other_modules()

logger = get_logger(__name__)


@click.group()
def cli():
    pass


def generate_random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))


@cli.command()
def debug_send_machine_specs_to_connector():
    """Debug sending machine specs to connector"""
    miner_service: MinerService = ioc["MinerService"]
    counter = 0

    while counter < 10:
        counter += 1
        debug_specs = {
            "gpu": {
                "count": 1,
                "details": [
                    {
                        "name": "NVIDIA RTX A5000",
                        "driver": "555.42.06",
                        "capacity": "24564",
                        "cuda": "8.6",
                        "power_limit": "230.00",
                        "graphics_speed": "435",
                        "memory_speed": "5000",
                        "pcei": "16",
                    }
                ],
            },
            "cpu": {"count": 128, "model": "AMD EPYC 7452 32-Core Processor", "clocks": []},
            "ram": {
                "available": 491930408,
                "free": 131653212,
                "total": 528012784,
                "used": 396359572,
            },
            "hard_disk": {"total": 20971520, "used": 13962880, "free": 7008640},
            "os": "Ubuntu 22.04.4 LTS",
        }
        asyncio.run(
            miner_service.publish_machine_specs(
                results=[
                    (
                        debug_specs,
                        ExecutorSSHInfo(
                            uuid=str(uuid.uuid4()),
                            address=generate_random_ip(),
                            port="8001",
                            ssh_username="test",
                            ssh_port=22,
                            python_path="test",
                            root_dir="test",
                        ),
                    )
                ],
                miner_hotkey="5Cco1xUS8kXuaCzAHAXZ36nr6mLzmY5B9ufxrfb8Q3HB6ZdN",
                miner_coldkey="5Cco1xUS8kXuaCzAHAXZ36nr6mLzmY5B9ufxrfb8Q3HB6ZdN",
            )
        )

        asyncio.run(
            miner_service.publish_machine_specs(
                results=[
                    (
                        debug_specs,
                        ExecutorSSHInfo(
                            uuid=str(uuid.uuid4()),
                            address=generate_random_ip(),
                            port="8001",
                            ssh_username="test",
                            ssh_port=22,
                            python_path="test",
                            root_dir="test",
                        ),
                    )
                ],
                miner_hotkey="5Cco1xUS8kXuaCzAHAXZ36nr6mLzmY5B9ufxrfb8Q3HB6ZdN",
                miner_coldkey="5Cco1xUS8kXuaCzAHAXZ36nr6mLzmY5B9ufxrfb8Q3HB6ZdN",
            )
        )

        time.sleep(2)


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
def request_job_to_miner(miner_hotkey: str):
    asyncio.run(_request_job_to_miner(miner_hotkey))


async def _request_job_to_miner(miner_hotkey: str):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)
        if not miner:
            raise ValueError(f"Miner with hotkey {miner_hotkey} not found")

        miner_service: MinerService = ioc["MinerService"]
        file_encrypt_service: FileEncryptService = ioc["FileEncryptService"]

        encrypted_files = file_encrypt_service.ecrypt_miner_job_files()

        result = await miner_service.request_job_to_miner(
            MinerJobRequestPayload(
                job_batch_id='job_batch_id',
                miner_hotkey=miner_hotkey,
                miner_coldkey=miner.coldkey,
                miner_address=miner.axon_info.ip,
                miner_port=miner.axon_info.port,
            ),
            encrypted_files=encrypted_files,
        )
        print('job_result:', result)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--count", type=int, prompt="Count", help="Number of job cycle")
def debug_validator(count: int):
    asyncio.run(_debug_validator(count))


async def _debug_validator(count: int):
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft_limit = 1024
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    # fetch miners
    subtensor_client = SubtensorClient.get_instance()
    miners = subtensor_client.get_miners()

    miner_service: MinerService = ioc["MinerService"]
    file_encrypt_service: FileEncryptService = ioc["FileEncryptService"]

    encrypted_files = file_encrypt_service.ecrypt_miner_job_files()

    job_batch_id = "123456789"

    for _ in range(count):
        jobs = [
            asyncio.create_task(
                miner_service.request_job_to_miner(
                    payload=MinerJobRequestPayload(
                        job_batch_id=job_batch_id,
                        miner_hotkey=miner.hotkey,
                        miner_coldkey=miner.coldkey,
                        miner_address=miner.axon_info.ip,
                        miner_port=miner.axon_info.port,
                    ),
                    encrypted_files=encrypted_files,
                )
            )
            for miner in miners
        ]

        task_info = {}

        for miner, job in zip(miners, jobs):
            task_info[job] = {
                "miner_hotkey": miner.hotkey,
                "miner_address": miner.axon_info.ip,
                "miner_port": miner.axon_info.port,
                "job_batch_id": job_batch_id,
            }

        try:
            # Run all jobs with asyncio.wait and set a timeout
            done, pending = await asyncio.wait(jobs, timeout=60 * 10 - 50)

            success = []
            errors = []
            pendings = []
            error_count = 0

            # Process completed jobs
            for task in done:
                try:
                    result = task.result()
                    if result:
                        miner_hotkey = result.get("miner_hotkey")

                        success.append({"miner_hotkey": miner_hotkey})
                    else:
                        info = task_info.get(task, {})
                        miner_hotkey = info.get("miner_hotkey", "unknown")

                        errors.append({"miner_hotkey": miner_hotkey})
                except Exception as e:
                    error_count += 1
                    logger.error(
                        _m(
                            "[sync] Error processing job result",
                            extra=get_extra_info(
                                {
                                    "error": str(e),
                                }
                            ),
                        ),
                    )

            # Handle pending jobs (those that did not complete within the timeout)
            if pending:
                for task in pending:
                    info = task_info.get(task, {})
                    miner_hotkey = info.get("miner_hotkey", "unknown")
                    task.cancel()

                    pendings.append({"miner_hotkey": miner_hotkey})

            logger.info(_m("[sync] All Jobs finished", extra={
                "success": success,
                "errors": errors,
                "pendings": pendings,
                "success_count": len(success),
                "errors_count": len(errors),
                "pendings_count": len(pendings),
                "error_count": error_count,
            }))

        except Exception as e:
            logger.error(
                _m(
                    "[sync] Unexpected error",
                    extra=get_extra_info(
                        {
                            "error": str(e),
                        }
                    ),
                ),
            )


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--docker_image", prompt="Docker Image", help="Docker Image")
@click.option("--enable-jupyter", is_flag=True, default=False, help="Enable Jupyter (default: False)")
def create_container_to_miner(miner_hotkey: str, executor_id: str, docker_image: str, enable_jupyter: bool = False):
    asyncio.run(_create_container_to_miner(miner_hotkey, executor_id, docker_image, enable_jupyter))


async def _create_container_to_miner(miner_hotkey: str, executor_id: str, docker_image: str, enable_jupyter: bool):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]

        payload = ContainerCreateRequest(
            docker_image=docker_image,
            user_public_keys=["user_public_key"],
            executor_id=executor_id,
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            enable_jupyter=enable_jupyter,
        )
        response = await miner_service.handle_container(payload)
        print('response ==>', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--container_name", prompt="Container name", help="Container name")
@click.option("--volume_name", required=False, help="Volume name when editing pod")
def delete_pod(miner_hotkey: str, executor_id: str, container_name: str, volume_name: str):
    asyncio.run(_delete_pod(miner_hotkey, executor_id, container_name, volume_name))


async def _delete_pod(miner_hotkey: str, executor_id: str, container_name: str, volume_name: str):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]

        payload = ContainerDeleteRequest(
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            executor_id=executor_id,
            container_name=container_name,
            volume_name=volume_name,
        )
        response = await miner_service.handle_container(payload)
        print('response ==>', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--container_name", prompt="Container name", help="Container name")
@click.option("--jupyter_port", type=int, prompt="Jupyter Port", help="Jupyter Port")
def install_jupyter_server(miner_hotkey: str, executor_id: str, container_name: str, jupyter_port: int):
    asyncio.run(_install_jupyter_server(miner_hotkey, executor_id, container_name, jupyter_port))


async def _install_jupyter_server(miner_hotkey: str, executor_id: str, container_name: str, jupyter_port: int):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]

        payload = InstallJupyterServerRequest(
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            executor_id=executor_id,
            container_name=container_name,
            jupyter_port_map=(jupyter_port, jupyter_port),
        )
        response = await miner_service.handle_container(payload)
        print('response ==>', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--docker_image", prompt="Docker Image", help="Docker Image")
def create_custom_container_to_miner(miner_hotkey: str, executor_id: str, docker_image: str):
    asyncio.run(_create_custom_container_to_miner(miner_hotkey, executor_id, docker_image))


async def _create_custom_container_to_miner(miner_hotkey: str, executor_id: str, docker_image: str):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]
        # mock custom options
        custom_options = CustomOptions(
            volumes=["/var/runer/docker.sock:/var/runer/docker.sock"],
            environment={"UPDATED_PUBLIC_KEY": "user_public_key"},
            entrypoint="",
            internal_ports=[22, 8002],
            startup_commands="/bin/bash -c 'apt-get update && apt-get install -y ffmpeg && pip install opencv-python'",
        )
        payload = ContainerCreateRequest(
            docker_image=docker_image,
            user_public_keys=["user_public_key"],
            executor_id=executor_id,
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            custom_options=custom_options
        )
        response = await miner_service.handle_container(payload)
        print('response ==>', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--container_name", prompt="Container name", help="Docker container name")
@click.option("--user_public_key", prompt="User Public key", help="public ssh key")
def add_sshkey_to_container(miner_hotkey: str, executor_id: str, container_name: str, user_public_key: str):
    asyncio.run(_add_sshkey_to_container(miner_hotkey, executor_id, container_name, user_public_key))


async def _add_sshkey_to_container(miner_hotkey: str, executor_id: str, container_name: str, user_public_key: str):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]

        payload = AddSshPublicKeyRequest(
            executor_id=executor_id,
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            container_name=container_name,
            user_public_keys=[user_public_key],
        )
        response = await miner_service.handle_container(payload)
        print('response ==>', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--container_name", prompt="Container name", help="Docker container name")
def get_pod_logs(miner_hotkey: str, executor_id: str, container_name: str):
    asyncio.run(_get_pod_logs(miner_hotkey, executor_id, container_name))


async def _get_pod_logs(miner_hotkey: str, executor_id: str, container_name: str):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]

        payload = GetPodLogsRequestFromServer(
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            executor_id=executor_id,
            container_name=container_name,
        )
        response = await miner_service.get_pod_logs(payload)
        print('response ==>', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_id", prompt="Executor Id", help="Executor Id")
@click.option("--public_key", prompt="Public Key", help="Public Key")
def add_debug_ssh_key(miner_hotkey: str, executor_id: str, public_key: str):
    asyncio.run(_add_debug_ssh_key(miner_hotkey, executor_id, public_key))


async def _add_debug_ssh_key(miner_hotkey: str, executor_id: str, public_key: str):
    try:
        subtensor_client = await SubtensorClient.initialize()
        miner = subtensor_client.get_miner(miner_hotkey)

        miner_service: MinerService = ioc["MinerService"]

        payload = AddDebugSshKeyRequest(
            miner_hotkey=miner_hotkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
            executor_id=executor_id,
            public_key=public_key,
        )

        response = await miner_service.add_debug_ssh_key(payload)
        print('response:', response)
    finally:
        logger.info("Shutting down subtensor client")
        await SubtensorClient.shutdown()

if __name__ == "__main__":
    cli()
