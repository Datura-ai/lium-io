import asyncio

from services.docker_service import DockerService
from services.miner_service import MinerService
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.redis_service import RedisService
from services.file_encrypt_service import FileEncryptService
from services.matrix_validation_service import ValidationService
from services.verifyx_validation_service import VerifyXValidationService
from services.collateral_contract_service import CollateralContractService

ioc = {}


async def initiate_services():
    ioc["SSHService"] = SSHService()
    ioc["RedisService"] = RedisService()
    ioc["FileEncryptService"] = FileEncryptService(
        ssh_service=ioc["SSHService"],
    )
    ioc["ValidationService"] = ValidationService()
    ioc["VerifyXValidationService"] = VerifyXValidationService()
    ioc["CollateralContractService"] = CollateralContractService()
    ioc["TaskService"] = TaskService(
        ssh_service=ioc["SSHService"],
        redis_service=ioc["RedisService"],
        validation_service=ioc["ValidationService"],
        verifyx_validation_service=ioc["VerifyXValidationService"],
        collateral_contract_service=ioc["CollateralContractService"],
    )
    ioc["DockerService"] = DockerService(
        ssh_service=ioc["SSHService"],
        redis_service=ioc["RedisService"]
    )
    ioc["MinerService"] = MinerService(
        ssh_service=ioc["SSHService"],
        task_service=ioc["TaskService"],
        redis_service=ioc["RedisService"]
    )


def sync_initiate():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initiate_services())


sync_initiate()
