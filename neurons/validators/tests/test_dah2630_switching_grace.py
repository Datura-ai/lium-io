"""DAH-2630: the RENTED↔UNRENTED switch window is availability-accounted (FR-I3).

A node mid-switch is neither rentable nor probeable BY DESIGN, so the accounting must
recognize the window instead of reading it as downtime. The claims pinned here:

  * a switching node inside its hardware-class budget gets a synthesized grace result —
    the same forced-pass shape as a special manual rental, ``spec=None`` so the backend's
    executor upsert is untouched;
  * a switch beyond its budget is a FLAG, never a longer window: no grace, a loud event;
  * with ENABLE_SWITCHING_GRACE off the sweep only observes — logs what it would have
    graced, contributes nothing;
  * a node that answered its validation normally is already scored and is never touched;
  * budgets resolve per GPU model with a safe fallback, because guest RAM return grows
    with CVM memory and one budget cannot fit both au11 and an H200 host.
"""

import json
from datetime import UTC, datetime, timedelta
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
from services.cvmd_relay import RelayResult
from services.miner_service import MinerService
from services.task.models import JobResult

HOST = CvmdHost(
    executor_uuid="e-switch",
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
    service._cvm_lifecycle_service = lifecycle
    return service, lifecycle


def switching(elapsed_seconds):
    return SwitchAssessment(
        reachable=True,
        state="SWITCHING",
        has_cvm=True,
        switching=True,
        elapsed_seconds=elapsed_seconds,
    )


def scored_result(executor_uuid="already-scored", tdx=False):
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


@pytest.fixture
def grace_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_SWITCHING_GRACE", True, raising=False)
    monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_DEFAULT", 300, raising=False)
    monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_BY_GPU_MODEL", "", raising=False)


class TestGraceWithinBudget:
    @pytest.mark.asyncio
    async def test_a_switching_node_inside_budget_is_graced(self, grace_on):
        service, _ = make_miner_service([HOST], switching(30.0))

        results = await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        assert len(results) == 1
        grace = results[0]
        assert grace.score == 1.0
        assert grace.spec is None, "a graced cycle must not touch the backend's executor row"
        assert grace.gpu_model == "NVIDIA H200"
        assert grace.gpu_count == 8
        assert str(grace.executor_info.uuid) == "e-switch"
        assert "CVM_SWITCH_GRACE" in grace.log_text

    @pytest.mark.asyncio
    async def test_a_node_that_already_answered_is_untouched(self, grace_on):
        service, lifecycle = make_miner_service([HOST], switching(30.0))

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[scored_result(executor_uuid="e-switch")]
        )

        assert results == []
        lifecycle.assess.assert_not_awaited()


class TestTheBudgetIsAHardEdge:
    @pytest.mark.asyncio
    async def test_beyond_budget_is_a_flag_not_a_longer_window(self, grace_on):
        service, _ = make_miner_service([HOST], switching(301.0))

        results = await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        assert results == []

    @pytest.mark.asyncio
    async def test_switching_with_no_readable_start_time_earns_no_grace(self, grace_on):
        service, _ = make_miner_service([HOST], switching(None))

        assert await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[]) == []

    def test_budgets_resolve_per_gpu_model_with_a_safe_fallback(self, monkeypatch):
        monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_DEFAULT", 300, raising=False)
        monkeypatch.setattr(
            settings,
            "CVM_SWITCH_BUDGET_SECONDS_BY_GPU_MODEL",
            json.dumps({"NVIDIA H200": 900}),
            raising=False,
        )
        assert settings.get_cvm_switch_budget_seconds("NVIDIA H200") == 900
        assert settings.get_cvm_switch_budget_seconds("NVIDIA A100") == 300
        assert settings.get_cvm_switch_budget_seconds(None) == 300

        monkeypatch.setattr(
            settings, "CVM_SWITCH_BUDGET_SECONDS_BY_GPU_MODEL", "not json", raising=False
        )
        assert settings.get_cvm_switch_budget_seconds("NVIDIA H200") == 300, (
            "bad config falls back rather than failing the cycle"
        )


class TestObserveMode:
    @pytest.mark.asyncio
    async def test_grace_off_observes_and_contributes_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_SWITCHING_GRACE", False, raising=False)
        monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_DEFAULT", 300, raising=False)
        monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_BY_GPU_MODEL", "", raising=False)
        service, _ = make_miner_service([HOST], switching(30.0))

        assert await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[]) == []

    @pytest.mark.asyncio
    async def test_lifecycle_off_probes_nothing_at_all(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", False, raising=False)
        service, lifecycle = make_miner_service([HOST], switching(30.0))

        assert await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[]) == []
        lifecycle.hosts_for_miner.assert_not_awaited()


class TestLifecycleHandoff:
    @pytest.mark.asyncio
    async def test_an_empty_idle_host_is_scheduled_for_a_launch_not_graced(self, grace_on):
        service, lifecycle = make_miner_service(
            [HOST], SwitchAssessment(reachable=True, state="RECONCILING", has_cvm=False)
        )

        results = await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[])

        assert results == []
        lifecycle.schedule_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_an_unknown_gpu_model_is_never_graced(self, grace_on):
        """Same guard as the manual-rental synthesis: an unknown model reaching the
        incentive layer aborts weight-setting for the whole subnet."""
        odd_host = CvmdHost(
            executor_uuid="e-odd",
            address="203.0.113.9",
            miner_hotkey="5Miner",
            gpu_model="Prototype GPU X",
            gpu_count=1,
            updated_at=0.0,
        )
        service, _ = make_miner_service([odd_host], switching(10.0))

        assert await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[]) == []


class TestTheRegistryFeed:
    @pytest.mark.asyncio
    async def test_attested_results_are_recorded_as_cvmd_hosts(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", False, raising=False)
        service, lifecycle = make_miner_service([], SwitchAssessment(reachable=False))

        attested = scored_result(executor_uuid="e-new", tdx=True)
        plain = scored_result(executor_uuid="e-plain", tdx=False)
        await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[attested, plain])

        lifecycle.record_host.assert_awaited_once()
        kwargs = lifecycle.record_host.await_args.kwargs
        assert kwargs["executor_uuid"] == "e-new"
        assert kwargs["address"] == "198.51.100.3"
        assert kwargs["gpu_model"] == "NVIDIA H200"


class TestAssessmentParsing:
    @pytest.mark.asyncio
    async def test_a_state_answer_becomes_a_switch_assessment(self):
        started = (datetime.now(UTC) - timedelta(seconds=42)).isoformat()
        redis = MagicMock()
        whitelist = MagicMock()
        from bittensor_wallet import Keypair

        service = CvmLifecycleService(redis, whitelist, Keypair.create_from_uri("//Alice"))
        service.client = MagicMock()
        service.client.state = AsyncMock(
            return_value=RelayResult(
                status=200,
                body={
                    "state": "SWITCHING",
                    "cvm": {"instance_id": "abc"},
                    "last_switch": {"started_at": started, "outcome": None},
                },
            )
        )

        assessment = await service.assess(HOST)

        assert assessment.reachable and assessment.switching
        assert assessment.elapsed_seconds == pytest.approx(42.0, abs=5.0)

    @pytest.mark.asyncio
    async def test_an_idle_answer_reads_as_empty(self):
        redis = MagicMock()
        whitelist = MagicMock()
        from bittensor_wallet import Keypair

        service = CvmLifecycleService(redis, whitelist, Keypair.create_from_uri("//Alice"))
        service.client = MagicMock()
        service.client.state = AsyncMock(
            return_value=RelayResult(status=200, body={"state": "RECONCILING", "cvm": None, "last_switch": None})
        )

        assessment = await service.assess(HOST)

        assert assessment.idle_without_cvm
        assert not assessment.switching
