"""create_task marks the node's SSH host key verified only when the shell opened with known_hosts pinned to it."""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from services.attestation_service import HostPolicyResult
from services.task.models import JobResult, build_msg
from services.task.service import TaskService

HOST_KEY = asyncssh.generate_private_key("ssh-ed25519").export_public_key().decode().strip()


class _Shell:
    def __init__(self, **kwargs: object) -> None:
        self.known_hosts = kwargs.get("known_hosts")

    async def __aenter__(self) -> "_Shell":
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


def _service(known_hosts: asyncssh.SSHKnownHosts | None) -> TaskService:
    service = TaskService.__new__(TaskService)
    service.redis_service = MagicMock()
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(
        return_value=HostPolicyResult(known_hosts=known_hosts)
    )
    event = build_msg(
        event="Validation finished", reason="OK", severity="info", impact="none"
    )
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=(True, [event], MagicMock(success=True)))
    service.pipeline_factory = MagicMock()
    service.pipeline_factory.build_context = AsyncMock(return_value=MagicMock(verified=None))
    service.pipeline_factory.build_pipeline = MagicMock(return_value=pipeline)
    return service


async def _run(known_hosts: asyncssh.SSHKnownHosts | None) -> JobResult:
    executor = ExecutorSSHInfo(
        uuid="node-9", address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
        python_path="/usr/bin/python3", root_dir="/root/app", ssh_host_key=HOST_KEY,
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1", miner_hotkey="5Miner", miner_coldkey="5Cold",
        miner_address="10.0.0.5", miner_port=8080, executors=[],
    )
    handled = JobResult(
        executor_info=executor, score=1, job_score=1, job_batch_id="batch-1",
        log_status="info", log_text="ok",
    )
    handler = MagicMock()
    handler.return_value.handle_result = AsyncMock(return_value=handled)
    with (
        patch("services.task.service.InteractiveShellService", _Shell),
        patch("services.task.service.ResultHandler", handler),
        patch("services.task.service.settings") as settings,
    ):
        settings.DRY_RUN = True
        return await _service(known_hosts).create_task(
            miner_info=miner, executor_info=executor, keypair=MagicMock(ss58_address="5Val"),
            private_key="key", public_key="pub", encrypted_files=MagicMock(),
            rented_data=MagicMock(), default_docker_image_digests={},
        )


@pytest.mark.asyncio
async def test_a_cycle_without_a_pinned_host_key_is_not_verified() -> None:
    result = await _run(None)

    assert result.ssh_host_key_verified is False


@pytest.mark.asyncio
async def test_a_connect_with_a_pinned_host_key_is_verified() -> None:
    pinned = asyncssh.import_known_hosts(f"[10.0.0.5]:2200 {HOST_KEY}\n")

    result = await _run(pinned)

    assert result.ssh_host_key_verified is True
