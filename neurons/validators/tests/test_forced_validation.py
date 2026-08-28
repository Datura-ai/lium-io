"""DAH-2090: forced validation of one executor, and the gates that keep it off production."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from clients.compute_client import ComputeClient
from payload_models.payloads import ForcedValidationRequest, MinerJobEnryptedFiles
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import MinerService
from services.task_service import JobResult

TARGET_EXECUTOR_ID = "11111111-2222-3333-4444-555555555555"
OTHER_EXECUTOR_ID = "99999999-8888-7777-6666-555555555555"
CYCLE_BATCH_ID = "2026-08-28 12:00:00"
ENCRYPTED_FILES = MinerJobEnryptedFiles(
    encrypt_key="key",
    all_keys={},
    tmp_directory="/tmp/forced",
    machine_scrape_file_name="machine_scrape",
)


def _make_miner_service(monkeypatch) -> MinerService:
    import services.miner_service as module

    service = MinerService.__new__(MinerService)
    service.forced_validation_lock = asyncio.Lock()
    service.backend_client = MagicMock(
        get_all_rented_executors=AsyncMock(return_value=RentedExecutorsResponse(executors={}))
    )
    service.file_encrypt_service = MagicMock(
        ecrypt_miner_job_files=MagicMock(return_value=ENCRYPTED_FILES)
    )
    service.request_job_to_miner = AsyncMock()
    service.publish_machine_specs = AsyncMock()

    subtensor_client = MagicMock()
    subtensor_client.get_miner = AsyncMock(
        return_value=MagicMock(coldkey="coldkey", axon_info=MagicMock(ip="1.2.3.4", port=8091))
    )
    subtensor_client.get_current_block = MagicMock(return_value=1_000_042)
    subtensor_client.get_time_from_block = AsyncMock(return_value=CYCLE_BATCH_ID)
    monkeypatch.setattr(
        module.SubtensorClient, "get_instance", MagicMock(return_value=subtensor_client)
    )
    monkeypatch.setattr(module, "fetch_default_image_digests", AsyncMock(return_value={}))
    return service


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
        job_batch_id=CYCLE_BATCH_ID,
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
        sysbox_runtime=True,
    )


@pytest.mark.asyncio
async def test_it_validates_and_publishes_the_target_executor_only(monkeypatch) -> None:
    service = _make_miner_service(monkeypatch)
    service.request_job_to_miner.return_value = {
        "miner_hotkey": "miner-hotkey",
        "miner_coldkey": "coldkey",
        # A manual-rental or failed run also answers with other executors; drop them.
        "results": [_make_result(TARGET_EXECUTOR_ID), _make_result(OTHER_EXECUTOR_ID)],
    }

    await service.validate_one_executor_now(TARGET_EXECUTOR_ID, "miner-hotkey", {})

    call = service.request_job_to_miner.await_args
    assert call.kwargs["executor_id"] == TARGET_EXECUTOR_ID
    # The backend parses job_batch_id as the cycle timestamp, so a forced run reuses the
    # running cycle's id instead of inventing one.
    assert call.kwargs["payload"].job_batch_id == CYCLE_BATCH_ID

    published = service.publish_machine_specs.await_args.args[0]
    assert [result.executor_info.uuid for result in published] == [TARGET_EXECUTOR_ID]


@pytest.mark.asyncio
async def test_unreachable_rental_data_stops_the_run(monkeypatch) -> None:
    """The scheduled cycle skips its iteration here; an empty map reads rented boxes as free."""
    service = _make_miner_service(monkeypatch)
    service.backend_client.get_all_rented_executors = AsyncMock(return_value=None)

    await service.validate_one_executor_now(TARGET_EXECUTOR_ID, "miner-hotkey", {})

    service.request_job_to_miner.assert_not_awaited()
    service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_forced_validations_do_not_run_at_the_same_time(monkeypatch) -> None:
    """ecrypt_miner_job_files wipes one fixed temp directory, so overlap destroys both runs."""
    service = _make_miner_service(monkeypatch)
    concurrent_runs = 0
    peak_concurrent_runs = 0

    async def slow_job(**kwargs):
        nonlocal concurrent_runs, peak_concurrent_runs
        concurrent_runs += 1
        peak_concurrent_runs = max(peak_concurrent_runs, concurrent_runs)
        await asyncio.sleep(0)
        concurrent_runs -= 1
        return {"miner_hotkey": "m", "miner_coldkey": "c", "results": []}

    service.request_job_to_miner = AsyncMock(side_effect=slow_job)

    await asyncio.gather(
        service.validate_one_executor_now(TARGET_EXECUTOR_ID, "miner-hotkey", {}),
        service.validate_one_executor_now(TARGET_EXECUTOR_ID, "miner-hotkey", {}),
    )

    assert peak_concurrent_runs == 1


def _make_client(monkeypatch, deploy_env: str) -> ComputeClient:
    from core.config import settings

    monkeypatch.setattr(settings, "DEPLOY_ENV", deploy_env)
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock(validate_one_executor_now=AsyncMock())
    return client


def _make_request() -> ForcedValidationRequest:
    return ForcedValidationRequest(
        miner_hotkey="miner-hotkey",
        executor_id=TARGET_EXECUTOR_ID,
        miner_address="1.2.3.4",
        miner_port=8091,
    )


@pytest.mark.asyncio
async def test_production_refuses_the_request(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")

    await client.handle_forced_validation(_make_request(), client.logging_extra)

    client.miner_service.validate_one_executor_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_staging_hands_the_request_to_the_service(monkeypatch) -> None:
    client = _make_client(monkeypatch, "STAGE")

    await client.handle_forced_validation(_make_request(), client.logging_extra)

    call = client.miner_service.validate_one_executor_now.await_args
    assert call.kwargs["executor_id"] == TARGET_EXECUTOR_ID
    assert call.kwargs["miner_hotkey"] == "miner-hotkey"
