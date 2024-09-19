from core.db import get_db
from daos.executor import ExecutorDao
from daos.task import TaskDao
from services.docker_service import DockerService
from services.miner_service import MinerService
from services.ssh_service import SSHService
from services.task_service import TaskService

ioc = {}


def initiate_daos():
    ioc["ExecutorDao"] = ExecutorDao(session=next(get_db()))
    ioc["TaskDao"] = TaskDao(session=next(get_db()))


def initiate_services():
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
    )


initiate_daos()
initiate_services()
