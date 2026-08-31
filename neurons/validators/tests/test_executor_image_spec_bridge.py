import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.compute_client import ComputeClient
from protocol.vc_protocol.validator_requests import ExecutorSpecRequest
from services.redis_service import MACHINE_SPEC_CHANNEL

_OUTDATED_IMAGE = {
    "status": "OUTDATED",
    "observed_digest": "sha256:deadbeef",
    "expected_ref": "daturaai/compute-subnet-executor:latest",
    "expected_digest": "sha256:feedface",
}


async def _bridge_machine_spec(payload: dict[str, Any]) -> ExecutorSpecRequest:
    async def listen() -> AsyncIterator[dict[str, bytes]]:
        yield {"channel": MACHINE_SPEC_CHANNEL.encode(), "data": json.dumps(payload).encode()}
        raise asyncio.CancelledError

    pubsub = MagicMock(listen=listen, aclose=AsyncMock())
    client = ComputeClient.__new__(ComputeClient)
    client.keypair = MagicMock(ss58_address="validator-hotkey")
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.compute_app_uri = "wss://backend.example/validator"
    client.miner_service = MagicMock()
    client.miner_service.redis_service.subscribe = AsyncMock(return_value=pubsub)
    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()
    return client.message_queue[0]


@pytest.mark.asyncio
async def test_bridge_carries_executor_image() -> None:
    payload = {
        "specs": {"gpu": {"count": 1}},
        "score": 0.0,
        "synthetic_job_score": 0.0,
        "log_status": "error",
        "job_batch_id": "2026-08-31 08:00:00",
        "log_text": "outdated",
        "miner_hotkey": "hk",
        "miner_coldkey": "ck",
        "executor_uuid": "exec-1",
        "executor_ip": "127.0.0.1",
        "executor_port": 8000,
        "executor_ssh_port": 22,
        "price_per_gpu": 0.5,
        "collateral_deposited": True,
        "ssh_pub_keys": [],
        "executor_image": _OUTDATED_IMAGE,
    }

    spec = await _bridge_machine_spec(payload)

    assert spec.executor_image == _OUTDATED_IMAGE
