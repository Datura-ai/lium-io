import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import bittensor

from core.config import settings
from core.validator import Validator
from incentive.config import IncentiveConfig
from incentive.factory import IncentiveFactory
from services.task_service import JobResult


@pytest.fixture
def incentive_redis_service():
    service = AsyncMock()

    gpu_portions = {
        "H100": 0.3,
        "H200": 0.25,
        "A100": 0.2,
        "RTX4090": 0.15,
        "RTX3090": 0.1,
    }

    async def get_portion(gpu_model):
        return gpu_portions.get(gpu_model, 0.1)

    service.get_portion_per_gpu_type = AsyncMock(side_effect=get_portion)
    service.get_executor_uptime = AsyncMock(return_value=100)

    return service


@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX", 0.1)
    monkeypatch.setattr(settings, "PORTION_FOR_UPTIME", 0.2)
    monkeypatch.setattr(settings, "UPTIME_REQUIRED_MINUTES", 120)
    monkeypatch.setattr(settings, "BURNERS", [0, 1])
    monkeypatch.setattr(settings, "NEW_BURNERS", [100, 101])
    monkeypatch.setattr(settings, "ENABLE_NEW_BURN_LOGIC", True)
    monkeypatch.setattr(settings, "DRY_RUN", False)
    monkeypatch.setattr(settings, "incentive", IncentiveConfig(algorithm="default"))
    return settings


@pytest.fixture
def create_job_result():
    def _factory(
        score: float = 1.0,
        job_score: float = 1.0,
        gpu_model: str = "H100",
        gpu_count: int = 1,
        sysbox_runtime: bool = True,
        collateral_deposited: bool = True,
        executor_id: str | None = None,
        job_batch_id: str = "2024-01-01 00:00:00",
    ) -> JobResult:
        from datura.requests.miner_requests import ExecutorSSHInfo

        if executor_id is None:
            executor_id = f"test-executor-{uuid4()}"

        executor_info = ExecutorSSHInfo(
            uuid=executor_id,
            address="192.168.1.100",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            port_mappings="[]",
            port_range="40000-50000",
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        )

        return JobResult(
            executor_info=executor_info,
            score=score,
            job_score=job_score,
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            sysbox_runtime=sysbox_runtime,
            collateral_deposited=collateral_deposited,
            job_batch_id=job_batch_id,
            log_status="success",
            log_text="Test job completed",
        )

    return _factory


@pytest.fixture
def create_neuron_info():
    def _factory(
        uid: int,
        hotkey: str,
        coldkey: str = "test_coldkey",
    ):
        neuron = MagicMock(spec=bittensor.NeuronInfo)
        neuron.uid = uid
        neuron.hotkey = hotkey
        neuron.coldkey = coldkey

        axon_info = MagicMock()
        axon_info.ip = "192.168.1.1"
        axon_info.port = 8091
        axon_info.is_serving = True
        neuron.axon_info = axon_info

        return neuron

    return _factory


@pytest.fixture
def mock_subtensor_client(mock_settings):
    from clients.subtensor_client import SubtensorClient

    client = SubtensorClient.__new__(SubtensorClient)
    client.netuid = 1
    client.redis_service = AsyncMock()
    client.redis_service.publish = AsyncMock()
    SubtensorClient._subtensor = MagicMock()
    SubtensorClient._subtensor.set_weights = MagicMock(return_value=(True, "ok"))
    client.default_extra = {}
    client.version_key = 10000
    client.wallet = MagicMock()
    client.send_weights_to_lium = AsyncMock()
    client.get_last_mechansim_step_block = MagicMock(return_value=123)
    client.get_metagraph = MagicMock(return_value=MagicMock())
    client.get_current_block = MagicMock(return_value=100)
    client.get_time_from_block = AsyncMock(return_value="2024-01-01 00:00:00")
    client.should_set_weights = AsyncMock(return_value=False)
    client.get_miners = AsyncMock(return_value=[])

    return client


@pytest.fixture
def validator_with_mocks(incentive_redis_service, mock_subtensor_client, mock_settings):
    validator = Validator()
    validator.incentive = IncentiveFactory.create(settings.incentive, incentive_redis_service)
    validator.redis_service = incentive_redis_service
    validator.subtensor_client = mock_subtensor_client
    validator.file_encrypt_service = MagicMock()
    validator.file_encrypt_service.ecrypt_miner_job_files = MagicMock(return_value={})
    validator.miner_service = MagicMock()
    validator.miner_service.request_job_to_miner = AsyncMock()
    validator.miner_service.publish_machine_specs = AsyncMock()
    validator.backend_client = MagicMock()
    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=[])
    validator.last_job_run_blocks = 0
    return validator
