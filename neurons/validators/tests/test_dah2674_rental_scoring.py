"""DAH-2674: a node whose CVM a renter holds is scored — on two-party proof.

During RENTER_RUNNING the executor process is gone with the validation CVM, so the node
cannot answer a normal validation and — before this task — scored zero for the whole rental.
The claims pinned here:

  * the forced pass needs TWO independent parties to agree: cvmd's host-side state read says
    RENTER_RUNNING, and the backend's rented-executors list says this executor is under a
    paid rental for this miner. A host that fabricates its state earns nothing — and cannot
    even keep its registry entry alive, because the refresh is gated the same way;
  * nothing read from inside the guest participates, so a root renter cannot move the
    provider's score (DAH-2676's rule, held by construction);
  * a host whose own /proc read says the supervisor is dead (`supervisor_alive: false`) is a
    host-side failure, not a scored cycle;
  * the pass carries the rented executor's REAL address and port — the uptime ledger is
    keyed on "address:port", and a synthesized port of 0 would miss it every cycle;
  * with ENABLE_CVM_RENTAL_SCORING off the sweep observes and contributes nothing;
  * a node that answered its validation normally is already scored and is never touched;
  * the spot-tier and Discord gates ride rented_data exactly as the manual synthesis reads
    them: a forced pass buys a place in the pool, not an exemption;
  * an unknown GPU model is skipped — same weight-setting guard as every other synthesis.

The zero-executor decision is deliberately NOT an upfront registry exemption: the flow
defers it to after the sweep, so a genuinely broken miner still produces its failure record
when no synthesis fires (see `_record_and_grace_cvm_hosts` call sites in miner_service.py).
"""

import pytest
from cvm_helpers import PAYLOAD, cvmd_host, make_miner_service, rented_data_for, scored_result

from core.config import settings
from services.cvm_lifecycle import SwitchAssessment

HOST = cvmd_host("e-rented")


def renter_running(supervisor_alive=True):
    return SwitchAssessment(
        reachable=True,
        state="RENTER_RUNNING",
        cvm={
            "instance_id": "cvm-1",
            "rental_id": "rental-1",
            "supervisor_alive": supervisor_alive,
            "ports": [],
        },
        # What `assess` folds off the wire object above — see its parsing test in the
        # DAH-2630 suite; the sweep reads only this field.
        supervisor_alive=supervisor_alive,
    )


@pytest.fixture
def rental_scoring_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", True, raising=False)


class TestTheForcedPass:
    @pytest.mark.asyncio
    async def test_a_rented_node_scores_the_manual_rental_shape(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

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
    async def test_the_pass_carries_the_rented_executors_real_endpoint(self, rental_scoring_on):
        """The uptime ledger is keyed on "address:port"; a synthesized port of 0 would miss
        it every cycle and, on a fleet enforcing the collateral uptime penalty, silently
        multiply the mining score to zero."""
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        assert results[0].executor_info.address == "203.0.113.7"
        assert results[0].executor_info.port == 4001

    @pytest.mark.asyncio
    async def test_the_evidence_is_host_side_plus_backend_record_only(self, rental_scoring_on):
        """One state read and the backend's own list — nothing that touches the guest."""
        service, lifecycle = make_miner_service([HOST], renter_running())

        await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        lifecycle.assess.assert_awaited_once()
        lifecycle.schedule_ensure.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_node_that_answered_normally_is_untouched(self, rental_scoring_on):
        service, lifecycle = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[scored_result(executor_uuid="e-rented")],
            rented_data=rented_data_for(),
        )

        assert results == []
        lifecycle.assess.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unknown_gpu_model_is_never_passed(self, rental_scoring_on):
        odd_host = cvmd_host("e-odd", "203.0.113.9", gpu_model="Prototype GPU X", gpu_count=1)
        service, _ = make_miner_service([odd_host], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for("e-odd")
        )
        assert results == []


class TestTheBackendIsHalfTheProof:
    @pytest.mark.asyncio
    async def test_a_state_answer_with_no_backend_rental_earns_nothing(self, rental_scoring_on):
        """A host that fabricates RENTER_RUNNING has no rental the platform is billing —
        without this gate, any HTTPS server answering /v1/state could farm emissions."""
        service, lifecycle = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(PAYLOAD, existing=[], rented_data=None)

        assert results == []
        # A fabricated state answer must not keep its own registry entry alive either.
        lifecycle.record_host.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_rental_recorded_for_another_miner_earns_nothing(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[],
            rented_data=rented_data_for(miner_hotkey="5SomeoneElse"),
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_a_dead_supervisor_is_a_host_failure_not_a_scored_cycle(self, rental_scoring_on):
        """QEMU died mid-rental and nothing relaunched it: the state document still says
        RENTER_RUNNING, but the host's own /proc read says the guest is gone."""
        service, _ = make_miner_service([HOST], renter_running(supervisor_alive=False))

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        assert results == []


class TestObserveMode:
    @pytest.mark.asyncio
    async def test_flag_off_observes_and_contributes_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", False, raising=False)
        service, lifecycle = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        assert results == []
        # Observation still refreshes the registry for the CONFIRMED rental: rollout-mode
        # fleets have long rentals too, and the host must not age out mid-rental.
        lifecycle.record_host.assert_awaited()


class TestTheRegistryOutlivesTheRental:
    @pytest.mark.asyncio
    async def test_a_confirmed_rental_rerecords_the_host_from_backend_facts(
        self, rental_scoring_on
    ):
        """Re-recorded, not merely refreshed: the address and GPU shape written back are the
        BACKEND's copy from this cycle's response, so the registry entry that the DAH-2629
        launch path will need after the rental cannot drift from the system of record."""
        service, lifecycle = make_miner_service([HOST], renter_running())

        await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        lifecycle.record_host.assert_awaited_once()
        recorded = lifecycle.record_host.await_args.kwargs
        assert recorded["executor_uuid"] == "e-rented"
        assert recorded["address"] == "203.0.113.7"
        assert recorded["miner_hotkey"] == "5Miner"
        assert recorded["gpu_model"] == "NVIDIA H200"
        assert recorded["gpu_count"] == 8


class TestIncentiveGates:
    @pytest.mark.asyncio
    async def test_spot_and_discord_exclusions_ride_rented_data(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[],
            rented_data=rented_data_for(spot=["e-rented"], discord=[]),
        )

        assert results[0].is_spot is True
        assert (
            results[0].provider_discord_connected is False
        ), "a forced pass buys a place in the pool, not an exemption from the gates"

    @pytest.mark.asyncio
    async def test_no_exclusion_lists_read_as_no_exclusions(self, rental_scoring_on):
        service, _ = make_miner_service([HOST], renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        assert results[0].is_spot is False
        assert results[0].provider_discord_connected is True


class TestTheAttestProbeHandoff:
    @pytest.mark.asyncio
    async def test_the_probe_is_scheduled_never_awaited(self, rental_scoring_on):
        """DAH-2675: the probe rides the rented branch fire-and-forget, so a slow or dead
        agent cannot extend the cycle — and being log-only it cannot move the score."""
        service, lifecycle = make_miner_service([HOST], renter_running())

        await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        lifecycle.schedule_attest_probe.assert_called_once()
        host, assessment = lifecycle.schedule_attest_probe.call_args.args
        assert host.executor_uuid == "e-rented"
        assert assessment.renter_running
