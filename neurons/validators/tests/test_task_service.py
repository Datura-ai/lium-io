import asyncio
from unittest.mock import MagicMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.task.models import JobResult
from services.task.service import TaskService


def _executor(executor_id: str = "exec-1") -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="10.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/root",
    )


def _payload(job_batch_id: str) -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id=job_batch_id,
        miner_hotkey="miner-hotkey",
        miner_coldkey="miner-coldkey",
        miner_address="127.0.0.1",
        miner_port=8000,
    )


def _encrypted_files() -> MinerJobEnryptedFiles:
    return MinerJobEnryptedFiles(
        encrypt_key="key",
        all_keys={},
        tmp_directory="/tmp",
        machine_scrape_file_name="machine.py",
    )


def _job_result(executor: ExecutorSSHInfo) -> JobResult:
    return JobResult(
        spec={},
        executor_info=executor,
        score=1,
        job_score=1,
        job_batch_id="cycle-job",
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
    )


@pytest.mark.asyncio
async def test_create_task_joins_active_validation_for_same_executor():
    service = TaskService.__new__(TaskService)
    service._active_validation_tasks_lock = asyncio.Lock()
    service._active_validation_tasks = {}

    executor = _executor()
    expected_result = _job_result(executor)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def create_task_impl(**_kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return expected_result

    service._create_task_impl = create_task_impl
    keypair = MagicMock()
    encrypted_files = _encrypted_files()
    rented_data = RentedExecutorsResponse(executors={})

    first = asyncio.create_task(
        service.create_task(
            miner_info=_payload("cycle-job"),
            executor_info=executor,
            keypair=keypair,
            private_key="private",
            public_key="public",
            encrypted_files=encrypted_files,
            rented_data=rented_data,
            default_docker_image_digests={},
        )
    )
    await started.wait()

    second = asyncio.create_task(
        service.create_task(
            miner_info=_payload("forced-job"),
            executor_info=executor,
            keypair=keypair,
            private_key="private",
            public_key="public",
            encrypted_files=encrypted_files,
            rented_data=rented_data,
            default_docker_image_digests={},
        )
    )

    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result is expected_result
    assert second_result is expected_result
    assert calls == 1
    assert service._active_validation_tasks == {}
