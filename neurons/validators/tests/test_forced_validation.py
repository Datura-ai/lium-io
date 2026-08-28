"""DAH-2090: forced validation of one executor, and the gate that keeps it off production."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from clients.compute_client import ComputeClient
from payload_models.payloads import ForcedValidationRequest
from services.task_service import JobResult

TARGET_EXECUTOR_ID = "11111111-2222-3333-4444-555555555555"
OTHER_EXECUTOR_ID = "99999999-8888-7777-6666-555555555555"


def _make_client(monkeypatch, deploy_env: str) -> ComputeClient:
    from core.config import settings

    monkeypatch.setattr(settings, "DEPLOY_ENV", deploy_env)
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock()
    client.miner_service.request_job_to_miner = AsyncMock()
    client.miner_service.publish_machine_specs = AsyncMock()
    client.subtensor_client = MagicMock()
    client.subtensor_client.get_miner = AsyncMock(return_value=MagicMock(coldkey="coldkey"))
    client.keypair = MagicMock()
    return client


def _make_request() -> ForcedValidationRequest:
    return ForcedValidationRequest(
        miner_hotkey="miner-hotkey",
        executor_id=TARGET_EXECUTOR_ID,
        miner_address="1.2.3.4",
        miner_port=8091,
    )


def _make_result(executor_id: str) -> JobResult:
    return JobResult(
        spec={},
        executor_info=ExecutorSSHInfo(
            uuid=executor_id,
            address="1.2.3.4",
            port=8091,
            ssh_username="root",
            ssh_port=22,
            python_path="",
            root_dir="",
        ),
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        job_batch_id="forced-batch",
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
        sysbox_runtime=True,
    )


@pytest.mark.asyncio
async def test_production_refuses_the_request(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")

    await client.handle_forced_validation(_make_request(), client.logging_extra)

    client.miner_service.request_job_to_miner.assert_not_awaited()
    client.miner_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_staging_validates_and_publishes_the_target_executor_only(monkeypatch) -> None:
    import clients.compute_client as module

    client = _make_client(monkeypatch, "STAGE")
    client.miner_service.request_job_to_miner.return_value = {
        "miner_hotkey": "miner-hotkey",
        "miner_coldkey": "coldkey",
        # A whole-miner run also answers with other executors; a forced run must drop them.
        "results": [_make_result(TARGET_EXECUTOR_ID), _make_result(OTHER_EXECUTOR_ID)],
    }
    monkeypatch.setattr(
        module,
        "BackendClient",
        MagicMock(return_value=MagicMock(get_all_rented_executors=AsyncMock(return_value=None))),
    )
    monkeypatch.setattr(module, "FileEncryptService", MagicMock())

    await client.handle_forced_validation(_make_request(), client.logging_extra)

    call = client.miner_service.request_job_to_miner.await_args
    assert call.kwargs["executor_id"] == TARGET_EXECUTOR_ID

    published = client.miner_service.publish_machine_specs.await_args.args[0]
    assert [result.executor_info.uuid for result in published] == [TARGET_EXECUTOR_ID]
