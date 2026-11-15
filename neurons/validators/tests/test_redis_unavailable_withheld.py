"""DAH-5805: a run the validator's own Redis ended gets no verdict, so the node is not scored 0 for it."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.exceptions
from services.executor_rollout import withhold_renter_gpu_reset_verdicts
from services.task.models import VALIDATOR_REDIS_UNAVAILABLE
from test_executor_rollout_grace import _failed, _scored


async def _run_with(error: Exception):
    from datura.requests.miner_requests import ExecutorSSHInfo
    from payload_models.payloads import MinerJobRequestPayload
    from services.task.service import TaskService

    class _Shell:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(
        return_value=MagicMock(
            known_hosts=None, attestation_digest=None, tee_type=None,
            gpu_attestation_passed=None, attestation_passed=False,
        )
    )
    service.pipeline_factory = MagicMock()
    service.pipeline_factory.build_context = AsyncMock(side_effect=error)
    executor = ExecutorSSHInfo(
        uuid="node-9", address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
        python_path="/usr/bin/python3", root_dir="/root/app",
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1", miner_hotkey="5Miner", miner_coldkey="5Cold",
        miner_address="10.0.0.5", miner_port=8080, executors=[],
    )
    with patch("services.task.service.InteractiveShellService", return_value=_Shell()):
        return await service.create_task(
            miner_info=miner, executor_info=executor, keypair=MagicMock(ss58_address="5Val"),
            private_key="key", public_key="pub", encrypted_files=MagicMock(),
            rented_data=MagicMock(), default_docker_image_digests={},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [redis.exceptions.ConnectionError("No connection available."), redis.exceptions.TimeoutError("Timeout")],
)
async def test_a_run_the_validator_redis_ended_is_named_and_leaves_the_node_listing_alone(error):
    result = await _run_with(error)

    assert result.failure_reason_code == VALIDATOR_REDIS_UNAVAILABLE
    # says nothing about the node: neither hides it nor lists it again
    assert result.availability_errors is None


@pytest.mark.asyncio
async def test_any_other_error_still_ends_the_run_as_before():
    result = await _run_with(ValueError("a check bug"))

    assert result.failure_reason_code is None and result.availability_errors == []


def test_the_cycle_withholds_a_run_the_validator_redis_ended():
    redis_fault = _failed("node-1", VALIDATOR_REDIS_UNAVAILABLE)
    node_fault = _failed("node-2", "GPU_LOST")
    scored = _scored("node-3")

    standing, withheld = withhold_renter_gpu_reset_verdicts({"5Miner": [redis_fault, node_fault, scored]})

    assert standing == {"5Miner": [node_fault, scored]}
    assert [w.result for w in withheld] == [redis_fault]
