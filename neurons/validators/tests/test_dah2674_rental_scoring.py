"""DAH-2674: a node whose CVM a renter holds is scored, from host-side signals alone.

During RENTER_RUNNING the executor process is gone with the validation CVM, so the node
cannot answer a normal validation and — before this task — scored zero for the whole rental.
The claims pinned here:

  * a RENTER_RUNNING node gets the same forced-pass shape as a special manual rental:
    score 1.0, ``spec=None`` so the backend's executor upsert is untouched, ``is_rented``;
  * the ONLY evidence consumed is cvmd's signed state read — the synthesis takes nothing
    from inside the guest, so a root renter cannot move the provider's score (DAH-2676's
    rule, held by construction);
  * with ENABLE_CVM_RENTAL_SCORING off the sweep observes and contributes nothing;
  * a node that answered its validation normally is already scored and is never touched;
  * the registry entry is refreshed on every rented sweep, scored or observed — a 720-hour
    rental must not age out of the 7-day registry TTL and silently fall back to zero;
  * the spot-tier and Discord gates come off rented_data exactly as the manual synthesis
    reads them: a forced pass buys a place in the pool, not an exemption;
  * an unknown GPU model is skipped — same weight-setting guard as every other synthesis;
  * the zero-executor early return exempts a miner with registered cvmd hosts, so a miner
    whose ONLY node is rented reaches this sweep instead of failing wholesale.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from services.cvm_lifecycle import (
    CvmdHost,
    CvmLifecycleService,
    SwitchAssessment,
)
from services.miner_service import MinerService
from services.task.models import JobResult

HOST = CvmdHost(
    executor_uuid="e-rented",
    address="203.0.113.7",
    miner_hotkey="5Miner",
    gpu_model="NVIDIA H200",
    gpu_count=8,
    updated_at=0.0,
)

PAYLOAD = SimpleNamespace(miner_hotkey="5Miner", miner_coldkey="5Cold", job_batch_id="batch-1")


def make_miner_service(hosts, assessment):
    service = MinerService.__new__(MinerService)
    service.redis_service = MagicMock()
    lifecycle = MagicMock(spec=CvmLifecycleService)
    lifecycle.record_host = AsyncMock()
    lifecycle.hosts_for_miner = AsyncMock(return_value=hosts)
    lifecycle.assess = AsyncMock(return_value=assessment)
    lifecycle.schedule_ensure = MagicMock()
    lifecycle.schedule_attest_probe = MagicMock()
    service._cvm_lifecycle_service = lifecycle
    return service, lifecycle


def renter_running(ports=None):
    return SwitchAssessment(
        reachable=True,
        state="RENTER_RUNNING",
        has_cvm=True,
        cvm={"instance_id": "cvm-1", "rental_id": "rental-1", "ports": ports or []},
    )


def scored_result(executor_uuid="already-scored"):
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
    )


@pytest.fixture
def rental_scoring_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", True, raising=False)


class TestTheForcedPass:
    @pytest.mark.asyncio
    async def test_a_rented_node_scores_the_manual_rental_shape(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        assert len(results) == 1
        passed = results[0]
        assert passed.score == 1.0
        assert passed.job_score == 1.0
        assert passed.spec is None, "a scored rental cycle must not touch the executor row"
        assert passed.is_rented is True
        assert passed.gpu_model == "NVIDIA H200"
        assert passed.gpu_count == 8
        assert str(passed.executor_info.uuid) == "e-rented"
        assert "CVM_RENTAL_FORCED_PASS" in passed.log_text

    @pytest.mark.asyncio
    async def test_the_only_evidence_is_the_host_side_state_read(self, rental_scoring_on):
        """The synthesis must complete from the assessment alone — the whole point of the
        design is that nothing inside the guest participates. One state read, no other call
        that could reach the guest."""
        service, lifecycle = make_miner_service([HOST], renter_running())

        await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        lifecycle.assess.assert_awaited_once()
        # The probe is scheduled, but it is log-only by construction (DAH-2675) — the score
        # above was already built before it could ever run.
        lifecycle.schedule_ensure.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_node_that_answered_normally_is_untouched(self, rental_scoring_on):
        service, lifecycle = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[scored_result(executor_uuid="e-rented")]
        )

        assert results == []
        lifecycle.assess.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unknown_gpu_model_is_never_passed(self, rental_scoring_on):
        odd_host = CvmdHost(
            executor_uuid="e-odd",
            address="203.0.113.9",
            miner_hotkey="5Miner",
            gpu_model="Prototype GPU X",
            gpu_count=1,
            updated_at=0.0,
        )
        service, _ = make_miner_service([odd_host], renter_running())

        assert await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[]) == []


class TestObserveMode:
    @pytest.mark.asyncio
    async def test_flag_off_observes_and_contributes_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", False, raising=False)
        service, lifecycle = make_miner_service([HOST], renter_running())

        assert await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[]) == []
        # Observation still refreshes the registry: rollout-mode fleets have long rentals too.
        lifecycle.record_host.assert_awaited()


class TestTheRegistryOutlivesTheRental:
    @pytest.mark.asyncio
    async def test_a_rented_sweep_refreshes_the_hosts_entry(self, rental_scoring_on):
        service, lifecycle = make_miner_service([HOST], renter_running())

        await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        kwargs = lifecycle.record_host.await_args.kwargs
        assert kwargs["executor_uuid"] == "e-rented"
        assert kwargs["address"] == "203.0.113.7"
        assert kwargs["gpu_model"] == "NVIDIA H200"
        assert kwargs["gpu_count"] == 8


class TestIncentiveGates:
    @pytest.mark.asyncio
    async def test_spot_and_discord_exclusions_ride_rented_data(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())
        rented_data = SimpleNamespace(
            spot_executor_ids=["e-rented"],
            provider_discord_connected_executor_ids=[],
        )

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data
        )

        assert results[0].is_spot is True
        assert (
            results[0].provider_discord_connected is False
        ), "a forced pass buys a place in the pool, not an exemption from the gates"

    @pytest.mark.asyncio
    async def test_no_rented_data_reads_as_no_exclusions(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        assert results[0].is_spot is False
        assert results[0].provider_discord_connected is True


class TestTheAttestProbeHandoff:
    @pytest.mark.asyncio
    async def test_the_probe_is_scheduled_never_awaited(self, rental_scoring_on):
        """DAH-2675: the probe rides the rented branch fire-and-forget, so a slow or dead
        agent cannot extend the cycle — and being log-only it cannot move the score."""
        service, lifecycle = make_miner_service([HOST], renter_running())

        await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        lifecycle.schedule_attest_probe.assert_called_once()
        host, assessment = lifecycle.schedule_attest_probe.call_args.args
        assert host.executor_uuid == "e-rented"
        assert assessment.renter_running


class TestTheZeroExecutorExemption:
    @pytest.mark.asyncio
    async def test_a_miner_with_cvmd_hosts_is_exempt(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service, _ = make_miner_service([HOST], renter_running())

        assert await service._has_cvmd_hosts(PAYLOAD) is True

    @pytest.mark.asyncio
    async def test_lifecycle_off_keeps_todays_failure_path(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", False, raising=False)
        service, lifecycle = make_miner_service([HOST], renter_running())

        assert await service._has_cvmd_hosts(PAYLOAD) is False
        lifecycle.hosts_for_miner.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_registry_trouble_means_no_exemption(self, monkeypatch):
        """Fail toward the stricter outcome: a broken registry must not excuse a miner."""
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service, lifecycle = make_miner_service([], renter_running())
        lifecycle.hosts_for_miner.side_effect = RuntimeError("redis is down")

        assert await service._has_cvmd_hosts(PAYLOAD) is False
