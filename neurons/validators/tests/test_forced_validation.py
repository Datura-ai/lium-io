"""DAH-2090: the forced validation cycle, and the gate that keeps it off production."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from clients.compute_client import ComputeClient
from core.validator import Validator
from payload_models.payloads import ForcedValidationCycleRequest
from services.miner_service import MinerService
from services.redis_service import RedisService

# The exact bytes the backend puts on the websocket, taken from
# ForcedValidationCycleRequest().model_dump_json() in lium-io-backend.
FORCED_CYCLE_MESSAGE_FROM_BACKEND = '{"message_type":"ForcedValidationCycleRequest"}'


def _make_client(monkeypatch, deploy_env: str) -> ComputeClient:
    from core.config import settings

    monkeypatch.setattr(settings, "DEPLOY_ENV", deploy_env)
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock(request_validation_cycle_now=AsyncMock())
    client.lock = asyncio.Lock()
    return client


@pytest.fixture
def shared_redis() -> FakeServer:
    """One store, reached through two clients -- the connector's and the validator's."""
    return FakeServer()


def _redis_service(shared_redis: FakeServer) -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=shared_redis)
    service.lock = asyncio.Lock()
    return service


@pytest.mark.asyncio
async def test_the_backend_message_reaches_the_service_through_real_dispatch(monkeypatch) -> None:
    """handle_message tries many models in turn, so the branch order is worth pinning."""
    client = _make_client(monkeypatch, "STAGE")

    await client.handle_message(FORCED_CYCLE_MESSAGE_FROM_BACKEND)

    client.miner_service.request_validation_cycle_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_production_ignores_the_backend_message(monkeypatch) -> None:
    client = _make_client(monkeypatch, "PROD")

    await client.handle_message(FORCED_CYCLE_MESSAGE_FROM_BACKEND)

    client.miner_service.request_validation_cycle_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_container_message_is_not_read_as_a_forced_cycle(monkeypatch) -> None:
    client = _make_client(monkeypatch, "STAGE")
    client.miner_drivers = asyncio.Queue()
    client.miner_driver = MagicMock(return_value=asyncio.sleep(0))
    delete_request = (
        '{"message_type":"ContainerDeleteRequest","miner_hotkey":"h",'
        '"executor_id":"e","pod_id":"p","container_name":"c","volume_name":"v"}'
    )

    await client.handle_message(delete_request)

    client.miner_service.request_validation_cycle_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_connector_request_reaches_the_validator_process(shared_redis) -> None:
    """The two processes agree only through Redis, so mocks cannot prove this."""
    connector_miner_service = MinerService.__new__(MinerService)
    connector_miner_service.redis_service = _redis_service(shared_redis)
    validator_process = Validator.__new__(Validator)
    validator_process.redis_service = _redis_service(shared_redis)

    assert await validator_process.an_operator_asked_for_a_cycle_now() is False

    await connector_miner_service.request_validation_cycle_now()

    assert await validator_process.an_operator_asked_for_a_cycle_now() is True
    # A tick that gives up before starting a cycle must leave the request pending.
    assert await validator_process.an_operator_asked_for_a_cycle_now() is True

    await validator_process.redis_service.clear_forced_validation_cycle_request()

    assert await validator_process.an_operator_asked_for_a_cycle_now() is False


def test_the_backend_bytes_still_parse_as_the_model_this_side_expects() -> None:
    """The two repos agree only on this string; nothing else checks that they still match."""
    assert ForcedValidationCycleRequest.model_validate_json(FORCED_CYCLE_MESSAGE_FROM_BACKEND)
