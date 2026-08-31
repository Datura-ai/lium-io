"""DAH-2090: validate one miner on demand, instead of the whole fleet."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import clients.compute_client as transport
import services.forced_miner_validation as forced
from clients.compute_client import ComputeClient
from clients.subtensor_client import SubtensorClient
from payload_models.payloads import ForcedMinerValidationRequest, MinerJobEnryptedFiles

MINER_HOTKEY = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
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
    f'{{"message_type":"ForcedMinerValidationRequest","miner_hotkey":"{MINER_HOTKEY}"}}'
)


@pytest.fixture
def subtensor_client(monkeypatch) -> MagicMock:
    client = MagicMock()
    client.get_miner = AsyncMock(
        return_value=MagicMock(coldkey="coldkey", axon_info=MagicMock(ip="1.2.3.4", port=8091))
    )
    client.get_current_block = MagicMock(return_value=1_000_042)
    client.get_time_from_block = AsyncMock(return_value=CYCLE_BATCH_ID)
    monkeypatch.setattr(SubtensorClient, "get_instance", MagicMock(return_value=client))
    monkeypatch.setattr(forced, "fetch_default_image_digests", AsyncMock(return_value={}))
    return client


@pytest.fixture
def miner_service() -> MagicMock:
    return MagicMock(request_job_to_miner=AsyncMock(), publish_machine_specs=AsyncMock())


@pytest.fixture
def backend_client() -> MagicMock:
    from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

    return MagicMock(
        get_all_rented_executors=AsyncMock(return_value=RentedExecutorsResponse(executors={}))
    )


@pytest.fixture
def file_encrypt_service() -> MagicMock:
    return MagicMock(ecrypt_miner_job_files=MagicMock(return_value=ENCRYPTED_FILES))


async def _run(miner_service, backend_client, file_encrypt_service) -> None:
    await forced.validate_one_miner_now(
        miner_service, backend_client, file_encrypt_service, MINER_HOTKEY
    )


@pytest.mark.asyncio
async def test_it_validates_the_named_miner_and_publishes(
    subtensor_client, miner_service, backend_client, file_encrypt_service
) -> None:
    result = MagicMock(log_text="ok")
    miner_service.request_job_to_miner.return_value = {"results": [result]}

    await _run(miner_service, backend_client, file_encrypt_service)

    payload = miner_service.request_job_to_miner.await_args.kwargs["payload"]
    assert payload.miner_hotkey == MINER_HOTKEY
    assert payload.miner_address == "1.2.3.4"
    # The backend parses job_batch_id as the cycle timestamp, so a forced run reuses the
    # running cycle's id instead of inventing one.
    assert payload.job_batch_id == CYCLE_BATCH_ID
    assert miner_service.publish_machine_specs.await_args.args[0] == [result]


@pytest.mark.asyncio
async def test_a_miner_outside_the_metagraph_is_refused(
    subtensor_client, miner_service, backend_client, file_encrypt_service
) -> None:
    subtensor_client.get_miner = AsyncMock(side_effect=ValueError("not present"))

    await _run(miner_service, backend_client, file_encrypt_service)

    miner_service.request_job_to_miner.assert_not_awaited()
    miner_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreachable_rental_data_stops_the_run(
    subtensor_client, miner_service, backend_client, file_encrypt_service
) -> None:
    """The scheduled cycle skips its iteration here; an empty map reads rented boxes as free."""
    backend_client.get_all_rented_executors = AsyncMock(return_value=None)

    await _run(miner_service, backend_client, file_encrypt_service)

    miner_service.request_job_to_miner.assert_not_awaited()
    miner_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_runs_do_not_overlap(
    subtensor_client, miner_service, backend_client, file_encrypt_service
) -> None:
    """The encryption step wipes one fixed temp directory, and the job reads from it."""
    concurrent_runs = 0
    peak_concurrent_runs = 0

    async def slow_job(**kwargs):
        nonlocal concurrent_runs, peak_concurrent_runs
        concurrent_runs += 1
        peak_concurrent_runs = max(peak_concurrent_runs, concurrent_runs)
        await asyncio.sleep(0)
        concurrent_runs -= 1
        return {"results": []}

    miner_service.request_job_to_miner = AsyncMock(side_effect=slow_job)

    await asyncio.gather(
        _run(miner_service, backend_client, file_encrypt_service),
        _run(miner_service, backend_client, file_encrypt_service),
    )

    assert peak_concurrent_runs == 1


def _make_client(monkeypatch, deploy_env: str) -> ComputeClient:
    from core.config import settings

    monkeypatch.setattr(settings, "DEPLOY_ENV", deploy_env)
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {}
    client.lock = asyncio.Lock()
    client.miner_drivers = asyncio.Queue()
    client.miner_service = MagicMock()
    client.backend_client = MagicMock()
    client.file_encrypt_service = MagicMock()
    return client


@pytest.mark.asyncio
async def test_the_backend_message_starts_a_run_without_blocking_the_socket(monkeypatch) -> None:
    """The receive loop reads one frame at a time, so the run must not be awaited inline."""
    client = _make_client(monkeypatch, "STAGE")
    started = asyncio.Event()
    monkeypatch.setattr(
        transport, "validate_one_miner_now", AsyncMock(side_effect=lambda *a: started.set())
    )

    await client.handle_message(BACKEND_MESSAGE)

    assert client.miner_drivers.qsize() == 1
    await client.miner_drivers.get_nowait()
    assert started.is_set()


@pytest.mark.asyncio
async def test_production_ignores_the_backend_message(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")
    run = AsyncMock()
    monkeypatch.setattr(transport, "validate_one_miner_now", run)

    await client.handle_message(BACKEND_MESSAGE)

    await client.miner_drivers.get_nowait()
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_container_message_is_not_read_as_a_forced_run(monkeypatch) -> None:
    client = _make_client(monkeypatch, "STAGE")
    client.miner_driver = MagicMock(return_value=asyncio.sleep(0))
    run = AsyncMock()
    monkeypatch.setattr(transport, "validate_one_miner_now", run)
    delete_request = (
        '{"message_type":"ContainerDeleteRequest","miner_hotkey":"h",'
        '"executor_id":"e","pod_id":"p","container_name":"c","volume_name":"v"}'
    )

    await client.handle_message(delete_request)

    run.assert_not_awaited()


def test_the_backend_bytes_still_parse_as_the_model_this_side_expects() -> None:
    """The two repos agree only on this string; nothing else checks that they still match."""
    request = ForcedMinerValidationRequest.model_validate_json(BACKEND_MESSAGE)
    assert request.miner_hotkey == MINER_HOTKEY
