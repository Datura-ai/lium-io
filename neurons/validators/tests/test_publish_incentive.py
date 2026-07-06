"""The validator must forward the per-executor incentive + default_job_owner to the
backend. They live on JobResult but were previously dropped when building the
MACHINE_SPEC_CHANNEL payload; this guards that they are included."""

from unittest.mock import AsyncMock, Mock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services.miner_service import MinerService
from services.task_service import JobResult


def _make_job(*, incentive: float | None, default_job_owner: str | None) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="executor-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="incentive-test-batch",
        log_status="success",
        log_text="ok",
        incentive=incentive,
        default_job_owner=default_job_owner,
    )


def _miner_service() -> MinerService:
    return MinerService(
        ssh_service=Mock(),
        task_service=Mock(),
        redis_service=AsyncMock(),
        attestation_service=Mock(),
    )


@pytest.mark.asyncio
async def test_publish_machine_specs_includes_incentive_fields():
    svc = _miner_service()
    job = _make_job(incentive=0.42, default_job_owner="lium")

    await svc.publish_machine_specs([job], miner_hotkey="miner_hk", miner_coldkey="miner_ck")

    _, published = svc.redis_service.publish.await_args.args
    assert published["incentive"] == 0.42
    assert published["default_job_owner"] == "lium"
