"""DAH-2811: validate one miner on demand, instead of the whole fleet."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from clients.compute_client import ComputeClient
from payload_models.payloads import (
    ForcedMinerValidationRequest,
    MinerJobEnryptedFiles,
)
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import MinerService
from services.task_service import JobResult

MINER_HOTKEY = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
EXECUTOR_ID = "11111111-2222-3333-4444-555555555555"
CYCLE_BATCH_ID = "2026-08-31 12:00:00"
ENCRYPTED_FILES = MinerJobEnryptedFiles(
    encrypt_key="key",
    all_keys={},
    tmp_directory="/tmp/forced",
    machine_scrape_file_name="machine_scrape",
)

# The exact bytes the backend puts on the websocket. Reproduce with
# ForcedMinerValidationRequest(miner_hotkey=...).model_dump_json() in lium-io-backend.
BACKEND_MESSAGE = (
    '{"message_type":"ForcedMinerValidationRequest","miner_hotkey":"' + MINER_HOTKEY + '"}'
)


def _make_service(monkeypatch) -> MinerService:
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
    subtensor_client.get_miners = AsyncMock(
        return_value=[
            MagicMock(
                hotkey=MINER_HOTKEY,
                coldkey="coldkey",
                axon_info=MagicMock(ip="1.2.3.4", port=8091),
            )
        ]
    )
    subtensor_client.get_current_block = MagicMock(return_value=1_000_042)
    subtensor_client.get_time_from_block = AsyncMock(return_value=CYCLE_BATCH_ID)
    monkeypatch.setattr(
        module.SubtensorClient, "get_instance", MagicMock(return_value=subtensor_client)
    )
    monkeypatch.setattr(module, "fetch_default_image_digests", AsyncMock(return_value={}))
    return service


def _make_result() -> JobResult:
    return JobResult(
        spec={},
        executor_info=ExecutorSSHInfo(
            uuid=EXECUTOR_ID,
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
async def test_it_validates_only_the_named_miner_and_publishes(monkeypatch) -> None:
    service = _make_service(monkeypatch)
    service.request_job_to_miner.return_value = {
        "miner_hotkey": MINER_HOTKEY,
        "miner_coldkey": "coldkey",
        "results": [_make_result()],
    }

    await service.validate_one_miner_now(MINER_HOTKEY)

    payload = service.request_job_to_miner.await_args.kwargs["payload"]
    assert payload.miner_hotkey == MINER_HOTKEY
    assert payload.miner_address == "1.2.3.4"
    # The backend parses job_batch_id as the cycle timestamp, so a forced run reuses the
    # running cycle's id instead of inventing one.
    assert payload.job_batch_id == CYCLE_BATCH_ID
    assert service.publish_machine_specs.await_args.args[0] == [service.request_job_to_miner.return_value["results"][0]]


@pytest.mark.asyncio
async def test_a_miner_outside_the_metagraph_is_refused(monkeypatch) -> None:
    service = _make_service(monkeypatch)
    import services.miner_service as module

    module.SubtensorClient.get_instance().get_miners = AsyncMock(return_value=[])

    await service.validate_one_miner_now("5UnknownHotkey")

    service.request_job_to_miner.assert_not_awaited()
    service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreachable_rental_data_stops_the_run(monkeypatch) -> None:
    """The scheduled cycle skips its iteration here; an empty map reads rented boxes as free."""
    service = _make_service(monkeypatch)
    service.backend_client.get_all_rented_executors = AsyncMock(return_value=None)

    await service.validate_one_miner_now(MINER_HOTKEY)

    service.request_job_to_miner.assert_not_awaited()
    service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_runs_do_not_overlap(monkeypatch) -> None:
    """ecrypt_miner_job_files wipes one fixed temp directory, so overlap destroys both runs."""
    service = _make_service(monkeypatch)
    concurrent_runs = 0
    peak_concurrent_runs = 0

    async def slow_job(**kwargs):
        nonlocal concurrent_runs, peak_concurrent_runs
        concurrent_runs += 1
        peak_concurrent_runs = max(peak_concurrent_runs, concurrent_runs)
        await asyncio.sleep(0)
        concurrent_runs -= 1
        return {"miner_hotkey": MINER_HOTKEY, "miner_coldkey": "c", "results": []}

    service.request_job_to_miner = AsyncMock(side_effect=slow_job)

    await asyncio.gather(
        service.validate_one_miner_now(MINER_HOTKEY),
        service.validate_one_miner_now(MINER_HOTKEY),
    )

    assert peak_concurrent_runs == 1


def _make_client(monkeypatch, deploy_env: str) -> ComputeClient:
    from core.config import settings

    monkeypatch.setattr(settings, "DEPLOY_ENV", deploy_env)
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {}
    client.miner_service = MagicMock(validate_one_miner_now=AsyncMock())
    client.lock = asyncio.Lock()
    return client


@pytest.mark.asyncio
async def test_the_backend_message_reaches_the_service(monkeypatch) -> None:
    """handle_message tries many models in turn, so the branch order is worth pinning."""
    client = _make_client(monkeypatch, "STAGE")

    await client.handle_message(BACKEND_MESSAGE)

    client.miner_service.validate_one_miner_now.assert_awaited_once_with(MINER_HOTKEY)


@pytest.mark.asyncio
async def test_production_ignores_the_backend_message(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")

    await client.handle_message(BACKEND_MESSAGE)

    client.miner_service.validate_one_miner_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_container_message_is_not_read_as_a_forced_run(monkeypatch) -> None:
    client = _make_client(monkeypatch, "STAGE")
    client.miner_drivers = asyncio.Queue()
    client.miner_driver = MagicMock(return_value=asyncio.sleep(0))
    delete_request = (
        '{"message_type":"ContainerDeleteRequest","miner_hotkey":"h",'
        '"executor_id":"e","pod_id":"p","container_name":"c","volume_name":"v"}'
    )

    await client.handle_message(delete_request)

    client.miner_service.validate_one_miner_now.assert_not_awaited()
