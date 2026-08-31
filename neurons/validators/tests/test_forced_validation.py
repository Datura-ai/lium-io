"""DAH-2090: the forced validation cycle, and the gate that keeps it off production."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.compute_client import ComputeClient
from payload_models.payloads import ForcedValidationCycleRequest
from services.miner_service import MinerService
from services.redis_service import FORCED_VALIDATION_CYCLE_KEY


def _make_client(monkeypatch, deploy_env: str) -> ComputeClient:
    from core.config import settings

    monkeypatch.setattr(settings, "DEPLOY_ENV", deploy_env)
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock(request_validation_cycle_now=AsyncMock())
    return client


@pytest.mark.asyncio
async def test_production_refuses_the_request(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")

    await client.handle_forced_validation_cycle()

    client.miner_service.request_validation_cycle_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_staging_hands_the_request_to_the_service(monkeypatch) -> None:
    client = _make_client(monkeypatch, "STAGE")

    await client.handle_forced_validation_cycle()

    client.miner_service.request_validation_cycle_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_service_leaves_the_flag_for_the_validator_process() -> None:
    """The cycle runs in another process, so the request crosses over Redis."""
    service = MinerService.__new__(MinerService)
    service.redis_service = MagicMock(set=AsyncMock())

    await service.request_validation_cycle_now()

    service.redis_service.set.assert_awaited_once_with(FORCED_VALIDATION_CYCLE_KEY, "1")


def test_the_message_only_matches_its_own_payload() -> None:
    """handle_message tries this model against every incoming message, so it must not over-match."""
    assert ForcedValidationCycleRequest.model_validate(
        {"message_type": "ForcedValidationCycleRequest"}
    )
    with pytest.raises(Exception):
        ForcedValidationCycleRequest.model_validate({"message_type": "ContainerCreateRequest"})


# The exact bytes the backend puts on the websocket: reproduce with
#   ForcedValidationCycleRequest().model_dump_json() in lium-io-backend.
BACKEND_WIRE_MESSAGE = '{"message_type":"ForcedValidationCycleRequest"}'


@pytest.mark.asyncio
async def test_the_backend_message_reaches_the_service_through_real_dispatch(monkeypatch) -> None:
    """handle_message tries many models in turn, so the branch order is worth pinning."""
    client = _make_client(monkeypatch, "STAGE")
    client.lock = asyncio.Lock()

    await client.handle_message(BACKEND_WIRE_MESSAGE)

    client.miner_service.request_validation_cycle_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_production_ignores_the_backend_message(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")
    client.lock = asyncio.Lock()

    await client.handle_message(BACKEND_WIRE_MESSAGE)

    client.miner_service.request_validation_cycle_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_another_message_is_not_captured_by_the_new_branch(monkeypatch) -> None:
    client = _make_client(monkeypatch, "STAGE")
    client.lock = asyncio.Lock()
    client.miner_drivers = asyncio.Queue()
    client.miner_driver = MagicMock(return_value=asyncio.sleep(0))
    delete_request = (
        '{"message_type":"ContainerDeleteRequest","miner_hotkey":"h",'
        '"executor_id":"e","pod_id":"p","container_name":"c","volume_name":"v"}'
    )

    await client.handle_message(delete_request)

    client.miner_service.request_validation_cycle_now.assert_not_awaited()
