import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from clients.compute_client import ComputeClient
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    JupyterServerInstalled,
    PayloadPortMapping,
)

JUPYTER_TOKEN = "0123456789abcdef0123456789abcdef"
JUPYTER_URL = f"http://executor.example:30888/lab?token={JUPYTER_TOKEN}"


def _compute_client(handle_container_reply) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock()
    client.miner_service.handle_container = AsyncMock(return_value=handle_container_reply)
    client.get_miner_axon_info = AsyncMock(return_value=MagicMock(ip="executor.example", port=8091))
    return client


def _create_request() -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        enable_jupyter=True,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )


@pytest.mark.asyncio
async def test_a_jupyter_create_logs_no_token(caplog):
    request = _create_request()
    created = ContainerCreated(
        miner_hotkey="miner",
        executor_id=request.executor_id,
        pod_id=request.pod_id,
        container_name="container_test",
        volume_name="volume_test",
        port_maps=[(22, 20001), (8888, 30888)],
        jupyter_url=JUPYTER_URL,
    )
    client = _compute_client(created)

    with caplog.at_level(logging.INFO):
        await client.miner_driver(request)

    # what the JSON log handler ships: the message and its `extra`
    logged = [record.msg.to_full_string() for record in caplog.records if hasattr(record.msg, "to_full_string")]
    assert any("Sending back created container info to compute app" in line for line in logged)
    assert not any(JUPYTER_TOKEN in line for line in logged)
    # the message itself still carries the URL to the backend
    assert client.message_queue[0].model_dump()["jupyter_url"] == JUPYTER_URL


def test_jupyter_server_installed_str_holds_no_token():
    installed = JupyterServerInstalled(
        miner_hotkey="miner", executor_id=str(uuid4()), pod_id=str(uuid4()), jupyter_url=JUPYTER_URL
    )

    assert JUPYTER_TOKEN not in str(installed)
    assert JUPYTER_TOKEN not in repr(installed)
    assert installed.model_dump(mode="json")["jupyter_url"] == JUPYTER_URL
