import asyncio

from clients.backend_client import BackendClient
from core.config import settings
from services.collateral_contract_service import CollateralContractService
from services.docker_service import DockerService
from services.executor_connectivity.container_runner import ContainerRunner
from services.executor_connectivity.dind_probe import DindProbe, DindVerifier
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector
from services.executor_connectivity.port_tester import PortTester
from services.executor_connectivity.port_verifiers import BatchVerifier, FallbackVerifier
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity import ContainerCleanupService
from services.executor_connectivity_service import ExecutorConnectivityService
from services.file_encrypt_service import FileEncryptService
from services.matrix_validation_service import ValidationService
from services.miner_service import MinerService
from services.redis_service import RedisService
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.verifyx_validation_service import VerifyXValidationService
from services.attestation_service import AttestationService

ioc = {}


async def initiate_services():
    # Backend clients
    keypair = settings.get_bittensor_wallet().get_hotkey()
    ioc["BackendClient"] = BackendClient(
        base_url=settings.COMPUTE_REST_API_URL or "",
        keypair=keypair,
    )

    ioc["SSHService"] = SSHService()
    ioc["RedisService"] = RedisService()
    ioc["FileEncryptService"] = FileEncryptService(
        ssh_service=ioc["SSHService"],
    )
    ioc["ValidationService"] = ValidationService()
    ioc["VerifyXValidationService"] = VerifyXValidationService()
    ioc["CollateralContractService"] = CollateralContractService()
    ioc["AttestationService"] = AttestationService()
    port_tester = PortTester()
    runner = ContainerRunner()
    ioc["ExecutorConnectivityService"] = ExecutorConnectivityService(
        orchestrator=ConnectivityOrchestrator(
            PortSelector(),
            PortProbe(
                BatchVerifier(port_tester, runner),
                FallbackVerifier(port_tester, runner),
            ),
            DindProbe(DindVerifier(ioc["SSHService"])),
        ),
        cleanup_service=ContainerCleanupService(),
    )
    ioc["TaskService"] = TaskService(
        ssh_service=ioc["SSHService"],
        redis_service=ioc["RedisService"],
        validation_service=ioc["ValidationService"],
        verifyx_validation_service=ioc["VerifyXValidationService"],
        collateral_contract_service=ioc["CollateralContractService"],
        executor_connectivity_service=ioc["ExecutorConnectivityService"],
        backend_client=ioc["BackendClient"],
        attestation_service=ioc["AttestationService"],
    )
    ioc["DockerService"] = DockerService(
        ssh_service=ioc["SSHService"],
        redis_service=ioc["RedisService"],
        attestation_service=ioc["AttestationService"],
    )
    ioc["MinerService"] = MinerService(
        ssh_service=ioc["SSHService"],
        task_service=ioc["TaskService"],
        redis_service=ioc["RedisService"],
        attestation_service=ioc["AttestationService"],
    )


def sync_initiate():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initiate_services())


sync_initiate()
