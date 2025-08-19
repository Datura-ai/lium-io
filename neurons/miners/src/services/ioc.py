from core.db import get_db

from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService, ValidatorDao
from services.executor_service import ExecutorService, ExecutorDao


ioc = {}


def initiate_daos():
    session = next(get_db())

    ioc["ValidatorDao"] = ValidatorDao(session=session)
    ioc["ExecutorDao"] = ExecutorDao(session=session)
    return ioc


def initiate_services():
    initiate_daos()

    ioc["MinerSSHService"] = MinerSSHService()
    ioc["ValidatorService"] = ValidatorService(
        validator_dao=ioc["ValidatorDao"],
    )
    ioc["ExecutorService"] = ExecutorService(
        executor_dao=ioc["ExecutorDao"],
    )
