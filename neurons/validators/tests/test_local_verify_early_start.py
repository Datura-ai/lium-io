"""liumd phase 3 (DAH-2834): the GPU `/verify` leaves right after the facts call, as a background
task, and `LocalVerifyCheck` judges it where it judges its own call today.

What changes: WHEN the one call is made. What does not: the intent (the same `_prepare`), the gate
(the same `_gate_reason`), the judge, the caps, the count (one GPU call per cycle). The fake executor
is phase 1's (`tests/test_local_verify.py`): it records every intent it receives, so a second call
would be visible. Also here: the derived facts deadline, the fast fail on a prestarted DinD
container a rental removed, and the batch/selected-ports accounting with a prestart.
"""
# ruff: noqa: F811 — pytest fixtures imported from the phase-1/2 test modules are re-bound as test parameters

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from neurons.validators.src.services.task.checks.local_facts import (
    FACTS_STEP_CAP_S,
    LocalFactsCheck,
    facts_deadline_s,
)
from neurons.validators.src.services.task.checks.local_verify import (
    LocalVerifyCheck,
    LocalVerifyOutcome,
    LocalVerifyStartCheck,
    PendingVerify,
)
from neurons.validators.src.services.task.checks.rental_verification import RentalProbe
from neurons.validators.src.services.task.pipeline import (
    CheckResult,
    LoggerSink,
    Pipeline,
    _settle_background_work,
)
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.const import GPU_HELD_VRAM_MB_LIMIT, GPU_WEDGE_UTILIZATION_MIN
from services.executor_connectivity.dind_probe import (
    DindContainerGone,
    DindVerifier,
)
from services.executor_connectivity.models import DindProbeResult, PortPair
from services.local_verify_client import CAPABILITY, EXECUTOR_DEADLINE_MIN_SECONDS
from services.local_verify_facts import LocalFacts

from core.config import Settings, settings
from tests.helpers import build_context_config, build_services, build_state, make_context
from tests.test_local_verify import (  # the phase-1 fakes
    EXECUTOR_UUID,
    SPECS,
    FakeExecutor,
    client_factory,
    keypair,  # noqa: F401 — fixture
    local_verify_on,  # noqa: F401 — fixture
    matmul_service,
    run_local_then_consumers,
    verifyx_service,  # noqa: F401 — fixture
)
from tests.test_local_verify_dind import executor_info, orchestrator, prepared

ADVERTISED = LocalFacts(capabilities=frozenset({CAPABILITY}), step_statuses={"docker": "ok", "ports": "ok", "inspector": "ok"})
NOT_ADVERTISED = LocalFacts(capabilities=frozenset())


@pytest.fixture
def early_on(monkeypatch, local_verify_on):  # noqa: F811
    monkeypatch.setattr(settings, "LOCAL_VERIFY_GPU_EARLY_START", True)


def context(keypair, executor_info, *, validation, verifyx, unscored=True, state=None, backend=None):
    services = dict(
        validation=validation,
        verifyx=verifyx,
        redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False)),
    )
    if backend is not None:
        services["backend"] = backend
        services["container_cleanup"] = SimpleNamespace(force_remove_health_checks=AsyncMock(return_value=0))
    return make_context(
        executor=executor_info,
        services=build_services(**services),
        config=build_context_config(
            validator_keypair=keypair, first_pass=unscored, unscored=unscored, verifyx_enabled=True
        ),
        state=state or build_state(specs=SPECS, local_facts=ADVERTISED),
    )


def ssh_must_not_run(validation, verifyx):
    validation.validate_gpu_model_and_process_job = AsyncMock(
        side_effect=AssertionError("SSH matmul must not run")
    )
    verifyx.validate_verifyx_and_process_job = AsyncMock(
        side_effect=AssertionError("SSH VerifyX must not run")
    )


def with_state(ctx, result: CheckResult):
    return ctx.model_copy(update={"state": result.updates.get("state", ctx.state)})


def outcome_lines(log) -> list[dict]:
    return [
        call.args[0].extra
        for call in log.info.call_args_list
        if str(call.args[0]) == "[local_verify] outcome"
    ]


# --- where it sits -------------------------------------------------------------------------------


def test_the_start_check_runs_right_after_the_facts_call_and_before_the_port_check():
    names = [type(c).__name__ for c in PipelineFactory.build_checks()]
    assert names.index("LocalFactsCheck") + 1 == names.index("LocalVerifyStartCheck")
    assert names.index("LocalVerifyStartCheck") < names.index("PortConnectivityCheck")
    assert names.index("LocalVerifyStartCheck") < names.index("TenantEnforcementCheck") < names.index("LocalVerifyCheck")
    assert "LocalVerifyStartCheck" not in [type(c).__name__ for c in PipelineFactory.build_dry_run_checks()]


def test_the_flag_ships_off_and_the_start_check_is_its_own_non_fatal_check():
    assert Settings.model_fields["LOCAL_VERIFY_GPU_EARLY_START"].default is False
    assert LocalVerifyStartCheck.fatal is False and LocalVerifyStartCheck.check_id != LocalVerifyCheck.check_id


@pytest.mark.asyncio
async def test_flag_off_starts_nothing_and_changes_no_state(keypair, monkeypatch, local_verify_on, verifyx_service):
    monkeypatch.setattr(settings, "LOCAL_VERIFY_GPU_EARLY_START", False)
    async with FakeExecutor(keypair) as executor:
        ctx = context(keypair, executor.executor_info, validation=matmul_service(monkeypatch), verifyx=verifyx_service)
        result = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert result.passed and result.event.reason_code == "LOCAL_VERIFY_DISABLED" and not result.updates
        await asyncio.sleep(0.05)
        assert executor.intents == []


# --- the same call, earlier --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_early_call_is_the_one_call_and_the_judge_consumes_it(
    keypair, monkeypatch, early_on, verifyx_service
):
    """Start → (the checks in between) → judge: exactly ONE `/verify` reaches the executor, the intent
    is the one `LocalVerifyCheck` would have sent, and both consumers take the local verdict."""
    validation = matmul_service(monkeypatch)
    ssh_must_not_run(validation, verifyx_service)
    async with FakeExecutor(keypair) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert start.passed and start.event.reason_code == "LOCAL_VERIFY_EARLY_STARTED", start.event
        assert start.event.what_we_saw["steps"] == ["matmul", "verifyx"]
        pending: PendingVerify = start.updates["state"].local_verify.pending
        assert isinstance(pending, PendingVerify) and not pending.consumed

        ctx2 = with_state(ctx, start)
        local, verifyx, capability = await run_local_then_consumers(
            ctx2, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
        assert local.passed and local.event.reason_code == "LOCAL_VERIFY_OK", local.event
        assert local.event.what_we_saw["consumed"] == ["matmul", "verifyx"]
        assert "early_lead_ms" in local.event.what_we_saw and "early_wait_ms" in local.event.what_we_saw
        outcome: LocalVerifyOutcome = local.updates["state"].local_verify
        assert outcome.matmul.success and outcome.verifyx.data["success"]
        assert outcome.pending is None  # consumed: nothing for the settle step
        assert pending.consumed and pending._closed
        assert capability.event.what_we_saw["transport"] == "local_verify"
        assert verifyx.event.what_we_saw["transport"] == "local_verify"

        # ONE `/verify` on the wire, and it is the phase-1 intent (the fake records the POSTs).
        assert len(executor.intents) == 1
        sent = executor.intents[0]
        assert sent["parallel_gpu"] is True and sent["steps"]["matmul"]["cipher_text"] == "deadbeef"
        assert sent["steps"]["verifyx"]["cipher_text"].startswith("vx")
        assert sent["steps"]["docker"] and sent["steps"]["ports"] and sent["steps"]["inspector"]


@pytest.mark.asyncio
async def test_the_gpu_work_overlaps_the_checks_in_between(keypair, monkeypatch, early_on, verifyx_service):
    """The executor takes 2.0 s; 1.5 s of 'port check' run in between; the judge waits ≈ 0.5 s, not
    2.0 — the time the overlap buys, reported as `early_lead_ms` / `early_wait_ms`. The bounds
    leave ≥ 1 s of slack each for a loaded runner; the order is what is asserted."""
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=2.0) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        t0 = time.perf_counter()
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert time.perf_counter() - t0 < 1.0  # the start does not wait for the executor
        await asyncio.sleep(1.5)  # stands in for PortConnectivity → TenantEnforcement
        judged_at = time.perf_counter()
        local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(with_state(ctx, start))
        waited = time.perf_counter() - judged_at
    assert local.event.reason_code == "LOCAL_VERIFY_OK"
    assert waited < 1.5, waited  # not the executor's 2.0 s
    assert time.perf_counter() - t0 < 4.0  # not 1.5 + 2.0
    lead, wait = local.event.what_we_saw["early_lead_ms"], local.event.what_we_saw["early_wait_ms"]
    assert lead >= 1400 and wait < 1500 and lead + wait >= 1900, (lead, wait)
    # The round trip judged against the SSH caps is still the call's own, not the judge's wait.
    assert local.event.what_we_saw["round_trip_ms"] >= 1900


@pytest.mark.asyncio
async def test_the_rental_probe_still_starts_where_the_judge_runs(keypair, monkeypatch, early_on, verifyx_service):
    """Phase 2b's probe needs `verified_ports` (the port check's result), so it cannot leave with
    the early call; it starts when LocalVerifyCheck picks the pending call up, and rides in the
    state as before."""
    monkeypatch.setattr(settings, "LOCAL_VERIFY_RENTAL_PROBE_PARALLEL", True)
    monkeypatch.setattr(settings, "SKIP_RENTAL_VERIFICATION", False)
    backend = AsyncMock()
    backend.check_executor_health = AsyncMock(return_value=SimpleNamespace(success=True, error=None, details={}))
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=0.3) as executor:
        ctx = context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend,
            state=build_state(specs={**SPECS, "verified_ports": [40001]}, local_facts=ADVERTISED),
        )
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        backend.check_executor_health.assert_not_awaited()
        local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(with_state(ctx, start))
    assert local.event.what_we_saw["rental_probe_started"] is True
    probe = local.updates["state"].local_verify.rental_probe
    assert probe is not None and probe.task.done()
    backend.check_executor_health.assert_awaited_once()


# --- when it does not start ----------------------------------------------------------------------


def rented_state(*, pods=True):
    rented = RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="p1", container_name="pod_p1", rented_ports=[40001])] if pods else [],
            )
        },
        banned_guids=[],
    )
    return build_state(specs=SPECS, local_facts=ADVERTISED, rented_data=rented)


def filler_state():
    rented = RentedExecutorsResponse(
        executors={}, banned_guids=[], filler_containers_by_executor={EXECUTOR_UUID: "filler_x"}
    )
    return build_state(specs=SPECS, local_facts=ADVERTISED, rented_data=rented)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case, reason",
    [
        ("no_facts", "no_facts"),
        ("not_advertised", "not_advertised"),
        ("facts_unanswered", "facts_unanswered"),
        ("scored", "not_first_pass"),
        ("rented", "rented"),
        ("wedge_candidate", "gpu_reread_pending"),
        ("ownerless_vram", "gpu_reread_pending"),
        ("filler", "filler_only"),
        ("no_specs", "no_specs"),
        ("no_keypair", "no_keypair"),
    ],
)
async def test_nothing_starts_early_when_the_check_itself_would_not_call_or_the_gpu_is_not_ours(
    keypair, monkeypatch, early_on, verifyx_service, case, reason
):
    validation = matmul_service(monkeypatch)
    state = build_state(specs=SPECS, local_facts=ADVERTISED)
    unscored, kp = True, keypair
    if case == "no_facts":
        state = build_state(specs=SPECS)
    elif case == "not_advertised":
        state = build_state(specs=SPECS, local_facts=NOT_ADVERTISED)
    elif case == "facts_unanswered":
        # `/version` advertised the capability but the facts POST itself failed (busy / timeout):
        # `LocalFactsCheck._unavailable` keeps the capabilities and no steps.
        state = build_state(specs=SPECS, local_facts=LocalFacts(capabilities=frozenset({CAPABILITY}), step_statuses={}))
    elif case == "scored":
        # The start check's own rule, not `_gate_reason`'s: with LOCAL_VERIFY_FIRST_PASS_ONLY off
        # the later check WOULD call on a scored cycle; the early start still does not.
        monkeypatch.setattr(settings, "LOCAL_VERIFY_FIRST_PASS_ONLY", False)
        unscored = False
    elif case == "wedge_candidate":
        # GpuUsageCheck would cure and re-sample the card: no process, full utilisation, no memory.
        state = build_state(specs=SPECS, local_facts=ADVERTISED, gpu_processes=[],
                            gpu_details=[{"uuid": "GPU-1", "gpu_utilization": GPU_WEDGE_UTILIZATION_MIN, "memory_utilization": 0}])
    elif case == "ownerless_vram":
        # GpuUsageCheck would confirm the held VRAM on the live card: memory above the floor, no process.
        state = build_state(specs=SPECS, local_facts=ADVERTISED, gpu_processes=[],
                            gpu_details=[{"uuid": "GPU-1", "gpu_utilization": 0, "memory_utilization": 5,
                                          "memory_used_mb": GPU_HELD_VRAM_MB_LIMIT + 1}])
    elif case == "rented":
        state = rented_state()
    elif case == "filler":
        state = filler_state()
    elif case == "no_specs":
        state = build_state(specs={}, local_facts=ADVERTISED)
    elif case == "no_keypair":
        kp = None
    async with FakeExecutor(keypair) as executor:
        ctx = make_context(
            executor=executor.executor_info,
            services=build_services(validation=validation, verifyx=verifyx_service),
            config=build_context_config(validator_keypair=kp, first_pass=unscored, unscored=unscored, verifyx_enabled=True),
            state=state,
        )
        with MagicMock() as log:
            monkeypatch.setattr("neurons.validators.src.services.task.checks.local_verify.logger", log)
            result = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        await asyncio.sleep(0.05)
        assert result.passed and result.event.reason_code == "LOCAL_VERIFY_EARLY_SKIPPED", result.event
        assert result.event.what_we_saw["reason"] == reason and not result.updates
        assert executor.intents == []
        lines = outcome_lines(log)
        assert lines and lines[-1]["step"] == "gpu_early" and lines[-1]["reason"] == reason
        # Never the later check's `step=call` label: that check reports its own outcome once.
        assert all(line["step"] != "call" for line in lines)


@pytest.mark.asyncio
async def test_an_idle_card_the_usage_check_reads_from_the_snapshot_alone_gets_the_early_call(
    keypair, monkeypatch, early_on, verifyx_service
):
    """The negative control for `gpu_reread_pending`: an idle card below the wedge and held-VRAM
    thresholds gives GpuUsageCheck nothing to re-read, so the call leaves early."""
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=5.0) as executor:
        state = build_state(specs=SPECS, local_facts=ADVERTISED, gpu_processes=[],
                            gpu_details=[{"uuid": "GPU-1", "gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 300}])
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, state=state)
        result = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert result.event.reason_code == "LOCAL_VERIFY_EARLY_STARTED", result.event
        await result.updates["state"].local_verify.pending.cancel_and_await()


@pytest.mark.asyncio
async def test_a_rented_node_never_gets_the_early_call_even_when_the_later_gate_would_pass(
    keypair, monkeypatch, early_on, verifyx_service
):
    """`_gate_reason` knows nothing about customer pods (TenantEnforcementCheck halts them before
    the later check runs); the start check must, or a renter's GPU would run our matmul."""
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, state=rented_state())
        result = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert result.event.what_we_saw["reason"] == "rented"
        # A node whose rented entry has no pods (a partially rented split node's row without a pod)
        # is not "rented" for this purpose, exactly as `_get_filler_only_container` reads it.
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, state=rented_state(pods=False))
        result = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert result.event.reason_code == "LOCAL_VERIFY_EARLY_STARTED"
        await result.updates["state"].local_verify.pending.cancel_and_await()


@pytest.mark.asyncio
async def test_without_an_early_start_the_later_check_calls_as_today(keypair, monkeypatch, early_on, verifyx_service):
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service,
                      state=build_state(specs=SPECS, local_facts=NOT_ADVERTISED))
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        assert start.event.reason_code == "LOCAL_VERIFY_EARLY_SKIPPED"
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
        assert local.event.reason_code == "LOCAL_VERIFY_OK" and "early_lead_ms" not in local.event.what_we_saw
        assert len(executor.intents) == 1


@pytest.mark.asyncio
async def test_a_second_start_on_a_pending_call_is_refused(keypair, monkeypatch, early_on, verifyx_service):
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=5.0) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        first = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        second = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(with_state(ctx, first))
        assert second.event.what_we_saw["reason"] == "already_pending" and not second.updates
        for _ in range(100):  # the first call reaches the fake (which is inside its 5 s step)
            if executor.intents:
                break
            await asyncio.sleep(0.02)
        assert len(executor.intents) == 1
        assert await first.updates["state"].local_verify.pending.cancel_and_await() == "cancelled"


@pytest.mark.asyncio
async def test_a_native_error_while_preparing_is_the_start_checks_own_line_and_the_later_check_calls(
    keypair, monkeypatch, early_on, verifyx_service
):
    """`_prepare` failing under the start check is reported as `step=gpu_early reason=prepare_failed`
    (an EARLY_SKIPPED event, nothing open); the later check builds its own challenges and makes its
    own call — so the per-cycle `step=call` count stays one."""
    validation = matmul_service(monkeypatch)
    good = validation.prepare_matmul_challenge
    validation.prepare_matmul_challenge = MagicMock(side_effect=RuntimeError("no libmatmul"))
    async with FakeExecutor(keypair) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        with MagicMock() as log:
            monkeypatch.setattr("neurons.validators.src.services.task.checks.local_verify.logger", log)
            start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
            assert start.event.reason_code == "LOCAL_VERIFY_EARLY_SKIPPED" and not start.updates
            assert start.event.what_we_saw["reason"] == "prepare_failed"
            assert [(line["step"], line["reason"]) for line in outcome_lines(log)] == [("gpu_early", "prepare_failed")]
            validation.prepare_matmul_challenge = good
            local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
        assert local.event.reason_code == "LOCAL_VERIFY_OK" and len(executor.intents) == 1
        # The start check's failure was never reported under the later check's `step=call` label.
        assert [line["step"] for line in outcome_lines(log) if line["reason"] == "prepare_failed"] == ["gpu_early"]


@pytest.mark.asyncio
async def test_an_error_between_taking_the_call_and_awaiting_it_still_cancels_the_task(
    keypair, monkeypatch, early_on, verifyx_service
):
    """The rental probe's request is built after the check picks the pending call up and before it
    awaits it; if that raises, `run` reports internal_error AND the task is cancelled — not left
    running with nobody to retrieve it."""
    monkeypatch.setattr(settings, "LOCAL_VERIFY_RENTAL_PROBE_PARALLEL", True)
    monkeypatch.setattr(
        "neurons.validators.src.services.task.checks.local_verify.rental_probe_request",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=5.0) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        pending = start.updates["state"].local_verify.pending
        t0 = time.perf_counter()
        local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(with_state(ctx, start))
        assert time.perf_counter() - t0 < 2.0
    assert local.event.reason_code == "LOCAL_VERIFY_FALLBACK" and local.event.what_we_saw["reason"] == "internal_error"
    assert pending.consumed and pending.task.done() and pending._closed


# --- a call nobody judges is settled -------------------------------------------------------------


class _Fatal:
    check_id, fatal = "test.fatal", True

    async def run(self, ctx):
        from neurons.validators.src.services.task.messages import (
            LocalVerifyMessages,
            render_message,
        )

        return CheckResult(passed=False, event=render_message(LocalVerifyMessages.SKIPPED, ctx=ctx, check_id=self.check_id))


@pytest.mark.asyncio
async def test_a_halt_before_the_judge_cancels_the_early_call_and_frees_the_challenge(
    keypair, monkeypatch, early_on, verifyx_service
):
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=5.0) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        pipeline = Pipeline([LocalVerifyStartCheck(client_factory=client_factory(keypair)), _Fatal()], sink=LoggerSink(MagicMock()))
        with MagicMock() as log:
            monkeypatch.setattr("neurons.validators.src.services.task.pipeline.logger", log)
            t0 = time.perf_counter()
            ok, events, final = await pipeline.run(ctx)
        assert not ok and time.perf_counter() - t0 < 2.0  # not the executor's 5 s
        pending = final.state.local_verify.pending
        assert pending.consumed and pending.task.cancelled() and pending._closed
        line = outcome_lines(log)[-1]
        assert line["step"] == "gpu_early" and line["reason"] == "unconsumed_cancelled"


@pytest.mark.asyncio
async def test_the_settle_step_handles_both_kinds_of_in_flight_work(monkeypatch):
    """One state carrying the early GPU call AND the rental probe: both are cancelled, both marked
    consumed, one outcome line each (`gpu_early`, `rental_probe`); a second settle does nothing."""

    async def never():
        await asyncio.sleep(60)

    pending = PendingVerify(task=asyncio.create_task(never()), matmul_challenge=MagicMock(), verifyx_challenge=None)
    probe = RentalProbe(task=asyncio.create_task(never()), request={})
    ctx = make_context(state=build_state(local_verify=LocalVerifyOutcome(pending=pending, rental_probe=probe)))
    with MagicMock() as log:
        monkeypatch.setattr("neurons.validators.src.services.task.pipeline.logger", log)
        await _settle_background_work(ctx)
        assert pending.consumed and pending.task.cancelled() and probe.consumed and probe.task.cancelled()
        pending.matmul_challenge.close.assert_called_once()
        assert sorted((line["step"], line["reason"]) for line in outcome_lines(log)) == [
            ("gpu_early", "unconsumed_cancelled"),
            ("rental_probe", "unconsumed_cancelled"),
        ]
        await _settle_background_work(ctx)  # idempotent
        pending.matmul_challenge.close.assert_called_once()
        assert len(outcome_lines(log)) == 2

    done = PendingVerify(task=asyncio.create_task(asyncio.sleep(0)), matmul_challenge=None, verifyx_challenge=None)
    await asyncio.sleep(0.01)
    ctx = make_context(state=build_state(local_verify=LocalVerifyOutcome(pending=done)))
    await _settle_background_work(ctx)
    assert done.consumed and not done.task.cancelled()


@pytest.mark.asyncio
async def test_a_gate_that_closes_between_start_and_judge_settles_the_call(
    keypair, monkeypatch, early_on, verifyx_service
):
    """The port check can refresh `rented_data`; a filler that appears makes the later check skip
    before it gets to the pending call — which it then settles itself, so nothing leaks to the
    pipeline's finally and the metric says so."""
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=3.0) as executor:
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        pending = start.updates["state"].local_verify.pending
        ctx2 = ctx.model_copy(
            update={"state": replace(filler_state(), local_verify=start.updates["state"].local_verify)}
        )
        with MagicMock() as log:
            monkeypatch.setattr("neurons.validators.src.services.task.checks.local_verify.logger", log)
            t0 = time.perf_counter()
            local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx2)
        assert local.event.reason_code == "LOCAL_VERIFY_SKIPPED" and time.perf_counter() - t0 < 1.5
        assert pending.consumed and pending.task.cancelled() and pending._closed
        line = outcome_lines(log)[-1]
        assert line["step"] == "gpu_early" and line["reason"] == "unconsumed_cancelled"


@pytest.mark.asyncio
async def test_an_early_call_that_fails_falls_back_to_ssh_like_a_late_one(
    keypair, monkeypatch, early_on, verifyx_service
):
    from neurons.validators.src.services import matrix_validation_service as mvs
    from neurons.validators.src.services import verifyx_validation_service as vvs

    validation = matmul_service(monkeypatch)
    validation.validate_gpu_model_and_process_job = AsyncMock(return_value=mvs.ValidationResult(success=True, metrics={"from": "ssh"}))
    verifyx_service.validate_verifyx_and_process_job = AsyncMock(return_value=vvs.VerifyXResponse(data={"success": True, "network": {}}))
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = lambda intent: (503, {"detail": "busy"})
        ctx = context(keypair, executor.executor_info, validation=validation, verifyx=verifyx_service)
        start = await LocalVerifyStartCheck(client_factory=client_factory(keypair)).run(ctx)
        pending = start.updates["state"].local_verify.pending
        local, verifyx, capability = await run_local_then_consumers(
            with_state(ctx, start), LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.event.reason_code == "LOCAL_VERIFY_FALLBACK"
    assert local.event.what_we_saw["reason"] in {"busy_or_replay", "refused", "http_error"}
    assert pending.consumed and pending._closed
    assert verifyx.event.what_we_saw["transport"] == "ssh" and capability.event.what_we_saw["transport"] == "ssh"
    assert len(executor.intents) == 1  # no retry


# --- the facts deadline follows the facts timeout -----------------------------------------------


@pytest.mark.parametrize(
    "timeout_s, expected", [(25, 20), (12, 7), (3, EXECUTOR_DEADLINE_MIN_SECONDS), (100, FACTS_STEP_CAP_S), (30, 20)]
)
def test_the_facts_deadline_is_the_timeout_less_a_margin_inside_the_executors_cap(timeout_s, expected):
    assert facts_deadline_s(timeout_s) == expected


def test_the_shipped_facts_timeout_still_asks_for_the_executors_full_cap():
    assert facts_deadline_s(Settings.model_fields["LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS"].default) == FACTS_STEP_CAP_S


@pytest.mark.asyncio
async def test_the_facts_intent_carries_the_derived_deadline(keypair, monkeypatch, local_verify_on):
    from tests.test_local_verify_facts import facts_answer

    monkeypatch.setattr(settings, "LOCAL_VERIFY_FACTS_ENABLED", True)
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FACTS_TIMEOUT_SECONDS", 14)
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = facts_answer
        ctx = make_context(
            executor=executor.executor_info,
            services=build_services(),
            config=build_context_config(validator_keypair=keypair),
            state=build_state(specs=SPECS),
        )
        result = await LocalFactsCheck(client_factory=client_factory(keypair)).run(ctx)
        assert result.passed and executor.intents[0]["deadline_s"] == 9


# --- a prestarted DinD container a rental removed does not cost the ready timeout ---------------


def _verifier(monkeypatch, *, deadline=6.0, poll=0.05):
    import asyncssh
    from services.executor_connectivity import dind_probe

    monkeypatch.setattr(dind_probe, "DIND_SSH_READY_TIMEOUT_SECONDS", deadline)
    monkeypatch.setattr(dind_probe, "DIND_SSH_POLL_INTERVAL_SECONDS", poll)
    monkeypatch.setattr(dind_probe, "DIND_SSH_CONNECT_TIMEOUT_SECONDS", 1.0)
    attempts = []

    async def refused(**kwargs):
        attempts.append(kwargs)
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(asyncssh, "connect", refused)
    return DindVerifier(ssh_service=SimpleNamespace()), attempts


@pytest.mark.asyncio
async def test_a_removed_prestarted_container_fails_at_the_first_refusal_not_at_the_deadline(monkeypatch):
    verifier, attempts = _verifier(monkeypatch, deadline=6.0)

    async def still_running():
        return False

    t0 = time.perf_counter()
    with pytest.raises(DindContainerGone):
        await verifier._connect_retrying_until_sshd_answers("127.0.0.1", PortPair(40000, 40000), "pkey", {}, still_running=still_running)
    assert time.perf_counter() - t0 < 1.0 and len(attempts) == 1


@pytest.mark.asyncio
async def test_a_container_still_running_is_polled_to_the_deadline_as_today(monkeypatch):
    verifier, attempts = _verifier(monkeypatch, deadline=1.0, poll=0.05)
    asked = []

    async def still_running():
        asked.append(1)
        return True

    with pytest.raises(ConnectionRefusedError):
        await verifier._connect_retrying_until_sshd_answers("127.0.0.1", PortPair(40000, 40000), "pkey", {}, still_running=still_running)
    assert len(attempts) >= 2 and len(asked) == len(attempts)


@pytest.mark.asyncio
async def test_an_unreadable_liveness_answer_is_not_gone(monkeypatch):
    verifier, attempts = _verifier(monkeypatch, deadline=1.0, poll=0.05)

    async def still_running():
        raise OSError("ssh dropped")

    with pytest.raises(ConnectionRefusedError):
        await verifier._connect_retrying_until_sshd_answers("127.0.0.1", PortPair(40000, 40000), "pkey", {}, still_running=still_running)
    assert len(attempts) >= 2


@pytest.mark.asyncio
async def test_a_container_that_answers_is_never_asked_whether_it_is_running(monkeypatch):
    """The liveness read is not on the path of a container whose sshd answers: no inspect, no
    extra round trip on the happy path."""
    import asyncssh

    verifier, attempts = _verifier(monkeypatch, deadline=6.0)
    connection = MagicMock(name="ssh")

    async def answers(**kwargs):
        attempts.append(kwargs)
        return connection

    monkeypatch.setattr(asyncssh, "connect", answers)
    still_running = AsyncMock(return_value=True)
    got = await verifier._connect_retrying_until_sshd_answers(
        "127.0.0.1", PortPair(40000, 40000), "pkey", {}, still_running=still_running
    )
    assert got is connection and len(attempts) == 1
    still_running.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_hung_liveness_read_is_cut_and_the_poll_goes_on(monkeypatch):
    """The `docker inspect` after a failed connect is bounded by its own timeout; a host that does
    not answer it is treated like an unreadable one (not "gone") and cannot stretch the probe."""
    import asyncssh
    from services.executor_connectivity import dind_probe

    verifier, attempts = _verifier(monkeypatch, deadline=1.0, poll=0.05)
    monkeypatch.setattr(dind_probe, "DIND_LIVENESS_READ_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(asyncssh, "import_private_key", lambda key: f"pkey({key})")

    async def hangs_on_inspect(command, *_a, **_k):
        if "docker inspect" in command:
            await asyncio.sleep(30)
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    host = SimpleNamespace(run=AsyncMock(side_effect=hangs_on_inspect))
    t0 = time.perf_counter()
    result = await verifier.verify(
        PortPair(40000, 40000), ssh_client=host, host="127.0.0.1", container_name_prefix="container_miner-hotkey",
        sysbox=True, log_ctx={}, prestarted=prepared(40000),
    )
    assert not result.success and "check failed" in result.log_text and time.perf_counter() - t0 < 5.0
    assert len(attempts) >= 2


@pytest.mark.asyncio
async def test_without_a_prestart_no_liveness_question_is_asked(monkeypatch):
    verifier, attempts = _verifier(monkeypatch, deadline=1.0, poll=0.05)
    with pytest.raises(ConnectionRefusedError):
        await verifier._connect_retrying_until_sshd_answers("127.0.0.1", PortPair(40000, 40000), "pkey", {})
    assert len(attempts) >= 2


@pytest.mark.asyncio
async def test_the_verifier_asks_the_host_with_one_inspect_and_reports_the_prestart_gone(monkeypatch):
    """End to end through `verify`: the first refusal → `docker inspect` says the container is not
    running → the probe fails as 'prestart gone' (no error-level traceback), removes by name and
    returns inside a second; the orchestrator then runs today's probe."""
    import asyncssh

    verifier, attempts = _verifier(monkeypatch, deadline=6.0)
    monkeypatch.setattr(asyncssh, "import_private_key", lambda key: f"pkey({key})")
    host = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=1, stdout="", stderr="")))
    p = prepared(40000)
    t0 = time.perf_counter()
    result = await verifier.verify(
        PortPair(40000, 40000), ssh_client=host, host="127.0.0.1", container_name_prefix="container_miner-hotkey",
        sysbox=True, log_ctx={}, prestarted=p,
    )
    assert not result.success and "prestart gone" in result.log_text and time.perf_counter() - t0 < 1.5
    assert p.consumed and len(attempts) == 1
    commands = [c.args[0] for c in host.run.await_args_list]
    assert commands[0].startswith("/usr/bin/docker inspect container_miner-hotkey_40000 --format")
    assert commands[-1] == "/usr/bin/docker rm -fv container_miner-hotkey_40000"


@pytest.mark.asyncio
async def test_a_running_prestart_whose_sshd_is_still_booting_keeps_the_poll(monkeypatch):
    import asyncssh

    verifier, attempts = _verifier(monkeypatch, deadline=1.0, poll=0.05)
    monkeypatch.setattr(asyncssh, "import_private_key", lambda key: f"pkey({key})")
    host = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout="true\n", stderr="")))
    result = await verifier.verify(
        PortPair(40000, 40000), ssh_client=host, host="127.0.0.1", container_name_prefix="container_miner-hotkey",
        sysbox=True, log_ctx={}, prestarted=prepared(40000),
    )
    assert not result.success and "check failed" in result.log_text and len(attempts) >= 2


# --- the prestarted port is one of the batch's ports -------------------------------------------


@pytest.mark.asyncio
async def test_the_prestarted_port_counts_as_selected_and_the_batch_asks_for_one_less(monkeypatch):
    from services.executor_connectivity import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "BATCH_PORT_VERIFICATION_SIZE", 3)
    orch, probe, dind = orchestrator([DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40000, 40000))])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None, prestarted_dind=prepared(40000))
    batch = probe.probe.await_args.args[0]
    assert len(batch) == 2 and PortPair(40000, 40000) not in batch  # 2 + the prestart = the batch of 3
    assert PortPair(40000, 40000) in result.selected_ports and len(result.selected_ports) == 3
    assert result.dind_port == PortPair(40000, 40000) and PortPair(40000, 40000) in result.successful_ports

    orch, probe, dind = orchestrator([DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40001, 40001))])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None)
    assert len(probe.probe.await_args.args[0]) == 3 and len(result.selected_ports) == 3  # today's count


@pytest.mark.asyncio
async def test_a_failed_prestart_is_reprobed_on_its_freed_port_which_stays_in_the_count():
    """#1346 (taiberium): the container that did not answer was the validator's own; today's probe
    reruns on the port it just freed and decides for it — the port is never dropped from the count,
    so a host with exactly MIN_PORT_COUNT ports cannot fail because our container was the one that
    did not answer. The prestarted port stays one of the selected (phase 3)."""
    orch, probe, dind = orchestrator([
        DindProbeResult(success=False, sysbox_runtime=True, port=PortPair(40000, 40000)),
        DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40000, 40000)),
    ])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None, prestarted_dind=prepared(40000))
    assert [c.args[0] for c in dind.verify.await_args_list] == [PortPair(40000, 40000), PortPair(40000, 40000)]
    assert PortPair(40000, 40000) in result.successful_ports and PortPair(40000, 40000) in result.selected_ports
    assert PortPair(40000, 40000) not in result.failed_ports and result.status == "ok"

    # When the rerun fails too, the port is a failed port like any DinD port that does not answer.
    orch, probe, dind = orchestrator([
        DindProbeResult(success=False, sysbox_runtime=True, port=PortPair(40000, 40000)),
        DindProbeResult(success=False, sysbox_runtime=True, port=PortPair(40000, 40000)),
    ])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[], ssh_client=None, prestarted_dind=prepared(40000))
    assert PortPair(40000, 40000) in result.failed_ports and PortPair(40000, 40000) in result.selected_ports
    assert PortPair(40000, 40000) not in result.successful_ports

    # The only-port branch: the same rerun on the freed port decides for it alone.
    orch, probe, dind = orchestrator([
        DindProbeResult(success=False, sysbox_runtime=True, port=PortPair(40003, 40003)),
        DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40003, 40003)),
    ])
    result = await orch.verify(executor_info=executor_info(), miner_hotkey="miner-hotkey", sysbox_runtime=True,
                              unavailable_ports=[40000, 40001, 40002], ssh_client=None, prestarted_dind=prepared(40003))
    assert result.successful_ports == (PortPair(40003, 40003),) and result.failed_ports == ()
    assert result.selected_ports == (PortPair(40003, 40003),)
