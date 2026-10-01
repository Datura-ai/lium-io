import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from clients.compute_client import ComputeClient
from payload_models.payloads import ContainerCreated, ContainerCreateRequest

from core.utils import JSONFormatter

JUPYTER_TOKEN = "secret-jupyter-token"


@pytest.mark.asyncio
async def test_connector_json_log_of_a_create_reply_has_no_jupyter_token(caplog):
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {}
    client.get_miner_axon_info = AsyncMock(return_value=SimpleNamespace(ip="127.0.0.1", port=8091))
    created = ContainerCreated(
        miner_hotkey="miner-hotkey",
        executor_id="executor-id",
        pod_id="pod-id",
        container_name="pod_pod-id",
        volume_name="volume_pod-id",
        port_maps=[(22, 20300), (8888, 20299)],
        jupyter_url=f"http://127.0.0.1:20299/lab?token={JUPYTER_TOKEN}",
    )
    client.miner_service = SimpleNamespace(handle_container=AsyncMock(return_value=created))
    request = ContainerCreateRequest(
        miner_hotkey="miner-hotkey",
        executor_id="executor-id",
        pod_id="pod-id",
        docker_image="daturaai/pytorch",
        user_public_keys=["ssh-ed25519 renter-key"],
        gpu_uuids=["GPU-0"],
    )

    with caplog.at_level(logging.INFO, logger="clients.compute_client"):
        await client._drive_miner_job(request)

    json_lines = [JSONFormatter(include_validator_hotkey=False).format(record) for record in caplog.records]
    reply_lines = [line for line in json_lines if "Sending back created container info" in line]
    assert len(reply_lines) == 1
    assert "pod_pod-id" in json.loads(reply_lines[0])["extra"]["response"]
    assert not any(JUPYTER_TOKEN in line for line in json_lines)
    # the reply itself still carries the URL to the backend
    assert JUPYTER_TOKEN in client.message_queue[0].model_dump_json()
