import asyncio

from services.docker_service import DockerService
from services.miner_service import MinerService
from services.ssh_service import SSHService
from services.task_service import TaskService

ioc = {}


async def initiate_services():
    ioc["SSHService"] = SSHService()
    ioc["TaskService"] = TaskService(
        ssh_service=ioc["SSHService"]
    )
    ioc["DockerService"] = DockerService(
        ssh_service=ioc["SSHService"]
    )
    ioc["MinerService"] = MinerService(
        ssh_service=ioc["SSHService"],
        task_service=ioc["TaskService"],
        docker_service=ioc["DockerService"],
    )


def sync_initiate():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initiate_services())


sync_initiate()
