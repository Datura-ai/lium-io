import asyncio

from core.db import get_db
from daos.executor import ExecutorDao
from daos.task import TaskDao
from services.docker_service import DockerService
from services.miner_service import MinerService
from services.ssh_service import SSHService
from services.task_service import TaskService

ioc = {}


async def initiate_daos():
    gen = get_db()
    session = await gen.__anext__()
    ioc["ExecutorDao"] = ExecutorDao(session=session)
    ioc["TaskDao"] = TaskDao(session=session)


async def initiate_services():
    ioc["SSHService"] = SSHService()
    ioc["TaskService"] = TaskService(
        task_dao=ioc["TaskDao"], executor_dao=ioc["ExecutorDao"], ssh_service=ioc["SSHService"]
    )
    ioc["DockerService"] = DockerService(
        ssh_service=ioc["SSHService"], executor_dao=ioc["ExecutorDao"]
    )
    ioc["MinerService"] = MinerService(
        ssh_service=ioc["SSHService"],
        task_service=ioc["TaskService"],
        docker_service=ioc["DockerService"],
        executor_dao=ioc["ExecutorDao"],
    )


def sync_initiate():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initiate_daos())
    loop.run_until_complete(initiate_services())


sync_initiate()
