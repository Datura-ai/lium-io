"""Shared builders for the CVM lifecycle suites (DAH-2630 / DAH-2674 / DAH-2678).

One home for the fakes that encode `_record_and_grace_cvm_hosts`'s collaborator surface.
A second copy of the harness would have to be taught every new lifecycle method by hand —
and a suite that missed one would get a silently-passing MagicMock auto-attribute instead
of a failure. Task suites import from here, never from each other.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from datura.requests.miner_requests import ExecutorSSHInfo

from services.cvm_lifecycle import CvmdHost, CvmLifecycleService
from services.miner_service import MinerService
from services.task.models import JobResult

PAYLOAD = SimpleNamespace(miner_hotkey="5Miner", miner_coldkey="5Cold", job_batch_id="batch-1")


def cvmd_host(
    executor_uuid="e-1",
    address="203.0.113.7",
    *,
    gpu_model="NVIDIA H200",
    gpu_count=8,
):
    return CvmdHost(
        executor_uuid=executor_uuid,
        address=address,
        miner_hotkey="5Miner",
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        updated_at=0.0,
    )


def make_miner_service(hosts, assessments):
    """A miner service whose lifecycle answers `assessments` (one value, or one per host)."""
    service = MinerService.__new__(MinerService)
    service.redis_service = MagicMock()
    lifecycle = MagicMock(spec=CvmLifecycleService)
    lifecycle.record_host = AsyncMock()
    lifecycle.touch_host = AsyncMock()
    lifecycle.hosts_for_miner = AsyncMock(return_value=hosts)
    if isinstance(assessments, list):
        lifecycle.assess = AsyncMock(side_effect=assessments)
    else:
        lifecycle.assess = AsyncMock(return_value=assessments)
    lifecycle.schedule_ensure = MagicMock()
    lifecycle.schedule_attest_probe = MagicMock()
    service._cvm_lifecycle_service = lifecycle
    return service, lifecycle


def scored_result(executor_uuid="already-scored", tdx=False):
    """A node that answered its validation normally — the shape a synthesis must never beat."""
    return JobResult(
        spec=None,
        executor_info=ExecutorSSHInfo(
            uuid=executor_uuid,
            address="198.51.100.3",
            port=1,
            ssh_username="",
            ssh_port=0,
            python_path="",
            root_dir="",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch-1",
        log_status="success",
        log_text="ok",
        gpu_model="NVIDIA H200",
        gpu_count=8,
        tdx_attestation_passed=tdx,
    )


def rented_data_for(executor_uuid="e-rented", *, miner_hotkey="5Miner", spot=(), discord=None):
    """The backend's half of the two-party proof: this executor is under a paid rental."""
    return SimpleNamespace(
        executors={
            executor_uuid: SimpleNamespace(
                miner_hotkey=miner_hotkey,
                executor_ip_address="203.0.113.7",
                executor_ip_port="4001",
            )
        },
        spot_executor_ids=list(spot),
        provider_discord_connected_executor_ids=discord,
    )
