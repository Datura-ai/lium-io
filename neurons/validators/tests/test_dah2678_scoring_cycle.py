"""DAH-2678, the local leg: one validation cycle against every cvmd lifecycle state.

Every validation cycle ends in the same sweep (`_record_and_grace_cvm_hosts`), and this suite
drives that sweep against a faked cvmd answer for each state a CVM v2 host can be in, pinning
what the provider scores in each. This is the regression net the flag flips lean on: each row
below is a claim about CURRENT behavior, so a change that moves one is a change someone must
have meant.

  state cvmd reports          what the sweep contributes
  ------------------------    -------------------------------------------------------------
  unreachable daemon          nothing — an unreachable cvmd says nothing about switching
  RECONCILING, no CVM         nothing this cycle; a validation-CVM launch is scheduled
  LAUNCHING                   nothing — the launch itself is the cycle's evidence
  VALIDATION_RUNNING          nothing — the real validation scored it (or failed it)
  SWITCHING inside budget     a grace: 1.0, spec=None
  SWITCHING beyond budget     nothing — the budget is a hard edge, flagged not extended
  RENTER_RUNNING              a rental forced pass: 1.0, spec=None, is_rented
  FAILED                      nothing — a failed node scores what a failed node scores

The hardware leg of DAH-2678 — the same sweep against a real cvmd on a TDX host — runs on
au11 via tools/cvm-local-e2e, not here.
"""

from types import SimpleNamespace

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from test_dah2674_rental_scoring import make_miner_service, rented_data_for

from core.config import settings
from services.cvm_lifecycle import CvmdHost, SwitchAssessment
from services.task.models import JobResult

PAYLOAD = SimpleNamespace(miner_hotkey="5Miner", miner_coldkey="5Cold", job_batch_id="batch-1")


def host(uuid="e-1", address="203.0.113.7"):
    return CvmdHost(
        executor_uuid=uuid,
        address=address,
        miner_hotkey="5Miner",
        gpu_model="NVIDIA H200",
        gpu_count=8,
        updated_at=0.0,
    )


UNREACHABLE = SwitchAssessment(reachable=False)
IDLE_EMPTY = SwitchAssessment(reachable=True, state="RECONCILING", has_cvm=False)
LAUNCHING = SwitchAssessment(reachable=True, state="LAUNCHING", has_cvm=False)
VALIDATION_RUNNING = SwitchAssessment(
    reachable=True, state="VALIDATION_RUNNING", has_cvm=True, cvm={"instance_id": "v-1"}
)
SWITCHING_IN_BUDGET = SwitchAssessment(
    reachable=True, state="SWITCHING", has_cvm=True, switching=True, elapsed_seconds=30.0
)
SWITCHING_OVER_BUDGET = SwitchAssessment(
    reachable=True, state="SWITCHING", has_cvm=True, switching=True, elapsed_seconds=301.0
)
RENTER_RUNNING = SwitchAssessment(
    reachable=True,
    state="RENTER_RUNNING",
    has_cvm=True,
    cvm={
        "instance_id": "r-1",
        "rental_id": "rental-1",
        "supervisor_alive": True,
        "ports": [],
    },
)
FAILED = SwitchAssessment(reachable=True, state="FAILED", has_cvm=False)


@pytest.fixture
def full_lifecycle_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_SWITCHING_GRACE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", True, raising=False)
    monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_DEFAULT", 300, raising=False)
    monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_BY_GPU_MODEL", "", raising=False)


class TestEveryStateInOneTable:
    @pytest.mark.parametrize(
        ("assessment", "synthesized", "launch_scheduled"),
        [
            pytest.param(UNREACHABLE, 0, False, id="unreachable"),
            pytest.param(IDLE_EMPTY, 0, True, id="idle-empty"),
            pytest.param(LAUNCHING, 0, False, id="launching"),
            pytest.param(VALIDATION_RUNNING, 0, False, id="validation-running"),
            pytest.param(SWITCHING_IN_BUDGET, 1, False, id="switching-in-budget"),
            pytest.param(SWITCHING_OVER_BUDGET, 0, False, id="switching-over-budget"),
            pytest.param(RENTER_RUNNING, 1, False, id="renter-running"),
            pytest.param(FAILED, 0, False, id="failed"),
        ],
    )
    @pytest.mark.asyncio
    async def test_what_one_state_earns(
        self, full_lifecycle_on, assessment, synthesized, launch_scheduled
    ):
        service, lifecycle = make_miner_service([host()], [assessment])

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for("e-1")
        )

        assert len(results) == synthesized
        for result in results:
            assert result.score == 1.0
            assert result.spec is None
        assert lifecycle.schedule_ensure.called is launch_scheduled


class TestOneCycleAcrossAMixedFleet:
    @pytest.mark.asyncio
    async def test_a_miner_with_every_state_at_once_scores_exactly_the_right_nodes(
        self, full_lifecycle_on
    ):
        """One miner, five cvmd hosts, one cycle: only the switching node inside budget and
        the rented node synthesize a pass; the empty host gets its launch scheduled; the rest
        contribute nothing. The per-state table above says each row; this says they compose."""
        fleet = [
            host("e-idle", "203.0.113.1"),
            host("e-validating", "203.0.113.2"),
            host("e-switching", "203.0.113.3"),
            host("e-rented", "203.0.113.4"),
            host("e-failed", "203.0.113.5"),
        ]
        service, lifecycle = make_miner_service(
            fleet,
            [IDLE_EMPTY, VALIDATION_RUNNING, SWITCHING_IN_BUDGET, RENTER_RUNNING, FAILED],
        )

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for("e-rented")
        )

        scored = {str(result.executor_info.uuid) for result in results}
        assert scored == {"e-switching", "e-rented"}
        assert all(result.score == 1.0 and result.spec is None for result in results)
        rented = next(r for r in results if str(r.executor_info.uuid) == "e-rented")
        assert rented.is_rented is True
        graced = next(r for r in results if str(r.executor_info.uuid) == "e-switching")
        assert graced.is_rented is False
        lifecycle.schedule_ensure.assert_called_once()
        lifecycle.schedule_attest_probe.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_normally_answered_node_is_never_double_scored(self, full_lifecycle_on):
        """A real result always beats a synthesized one, whatever state cvmd reports."""
        answered = JobResult(
            spec={"gpu": "real"},
            executor_info=ExecutorSSHInfo(
                uuid="e-rented",
                address="203.0.113.4",
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
        service, lifecycle = make_miner_service([host("e-rented")], [RENTER_RUNNING])

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[answered], rented_data=rented_data_for("e-rented")
        )

        assert results == []
        lifecycle.assess.assert_not_awaited()


class TestEverythingOffIsToday:
    @pytest.mark.asyncio
    async def test_all_flags_off_synthesizes_nothing_anywhere(self, monkeypatch):
        """The pre-CVM-v2 posture: whatever cvmd reports, the sweep contributes nothing.
        This is the row a production fleet sits on until each flag is deliberately flipped."""
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_SWITCHING_GRACE", False, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", False, raising=False)
        monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_DEFAULT", 300, raising=False)
        monkeypatch.setattr(settings, "CVM_SWITCH_BUDGET_SECONDS_BY_GPU_MODEL", "", raising=False)

        for assessment in (SWITCHING_IN_BUDGET, RENTER_RUNNING):
            service, _ = make_miner_service([host()], [assessment])
            results = await service._record_and_grace_cvm_hosts(
                PAYLOAD, existing=[], rented_data=rented_data_for("e-1")
            )
            assert results == []
