"""liumd phase 2 (DAH-2834): the backend rental probe started beside the executor's GPU steps.

`LocalVerifyCheck` starts `check_executor_health` as a task right before its one `/verify` call
(flag `LOCAL_VERIFY_RENTAL_PROBE_PARALLEL`, first pass only); `RentalVerificationCheck` awaits that
task instead of making the call itself — the same request, the same verdict code; `Pipeline.run`
settles a task nobody consumed. Fake executor and fake GPU services are the phase-1 ones
(`tests/test_local_verify.py`); the backend is an AsyncMock whose call has a configurable sleep.
"""
# ruff: noqa: F811 — pytest fixtures imported from test_local_verify are re-bound as test parameters

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.services.task.checks.local_verify import LocalVerifyCheck
from neurons.validators.src.services.task.checks.rental_verification import (
    RentalProbe,
    RentalVerificationCheck,
    health_check_request,
    rental_probe_request,
)
from neurons.validators.src.services.task.messages import RentalVerificationMessages as RentalMsg
from neurons.validators.src.services.task.pipeline import CheckResult, LoggerSink, Pipeline
from protocol.vc_protocol.compute_requests import (
    ExecutorHealthCheckResponse,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.local_verify_client import LocalVerifyOutcome

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
    verifyx_service,  # noqa: F401 — fixture
)

VERIFIED_SPECS = {**SPECS, "verified_ports": [40001, 40002]}
OK = ExecutorHealthCheckResponse(success=True, error=None, details={"container_healthy": True})


@pytest.fixture
def probe_on(monkeypatch, local_verify_on):  # noqa: F811
    monkeypatch.setattr(settings, "LOCAL_VERIFY_RENTAL_PROBE_PARALLEL", True)
    monkeypatch.setattr(settings, "SKIP_RENTAL_VERIFICATION", False)
    monkeypatch.setattr(settings, "FILLER_LIVENESS_CHECK_ENABLED", True)
    monkeypatch.setattr(settings, "FILLER_LIVENESS_ENFORCEMENT_ENABLED", True)
    monkeypatch.setattr(settings, "RENTAL_CPU_LIMIT_CHECK_ENABLED", False)


def backend_with(response=OK, *, sleep: float = 0.0, raises: Exception | None = None):
    """A backend whose `check_executor_health` records when it was entered and takes `sleep`."""
    backend = AsyncMock()
    backend.entered: list[float] = []

    async def health(**kwargs):
        backend.entered.append(time.perf_counter())
        await asyncio.sleep(sleep)
        if raises is not None:
            raise raises
        return response

    backend.check_executor_health = AsyncMock(side_effect=health)
    return backend


def probe_context(keypair, executor_info, *, validation, verifyx, backend, first_pass=True, state=None):
    cleanup = SimpleNamespace(force_remove_health_checks=AsyncMock(return_value=0))
    return make_context(
        executor=executor_info,
        services=build_services(
            validation=validation,
            verifyx=verifyx,
            backend=backend,
            container_cleanup=cleanup,
            redis=SimpleNamespace(renting_in_progress=AsyncMock(return_value=False)),
        ),
        # `unscored` mirrors the factory (the caller's word); `first_pass` is the fast-path-sized one.
        config=build_context_config(
            validator_keypair=keypair, first_pass=first_pass, unscored=first_pass, verifyx_enabled=True
        ),
        state=state or build_state(specs=VERIFIED_SPECS),
    )


async def run_local_then_rental(ctx, check: LocalVerifyCheck):
    local = await check.run(ctx)
    ctx2 = ctx.model_copy(update={"state": local.updates.get("state", ctx.state)})
    rental = await RentalVerificationCheck().run(ctx2)
    return local, rental, ctx2


def outcome_lines(log) -> list[dict]:
    return [
        call.args[0].extra
        for call in log.info.call_args_list
        if str(call.args[0]) == "[local_verify] outcome"
    ]


# --- the request builder: one for both callers ---------------------------------------------------


def test_the_probe_request_is_the_request_the_check_sends(monkeypatch, probe_on):
    """`rental_probe_request` must be exactly what `RentalVerificationCheck.run` passes to
    `check_executor_health` from the same context — the property the consume step checks for."""
    state = build_state(
        specs={**VERIFIED_SPECS, "gpu": {"details": [{"uuid": "GPU-1"}, {"uuid": "GPU-2"}, {"nouuid": 1}]}}
    )
    ctx = make_context(state=state)
    request = rental_probe_request(ctx)
    assert asdict(request) == {
        "miner_address": "127.0.0.1",
        "miner_port": 8000,
        "miner_hotkey": "miner-hotkey",
        "container_port": 40001,
        "executor_id": "executor-123",
        "rental_in_progress": False,
        "gpu_uuids": ["GPU-1", "GPU-2"],
        "cpu_count": None,
    }
    assert request == health_check_request(ctx, container_port=40001, rental_in_progress=False)


@pytest.mark.asyncio
async def test_the_check_itself_sends_the_builders_request(probe_on):
    backend = backend_with()
    ctx = probe_context(
        None, make_context().executor, validation=None, verifyx=None, backend=backend
    )
    result = await RentalVerificationCheck().run(ctx)
    assert result.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_args.kwargs == asdict(rental_probe_request(ctx))


@pytest.mark.parametrize(
    "case",
    ["skip_setting", "customer_rental", "filler", "create_killed_enforced", "no_ports"],
)
def test_the_probe_request_is_none_where_the_check_would_return_early(monkeypatch, probe_on, case):
    """Each early return of `RentalVerificationCheck.run` before the backend call (or the cheap
    rented call) is mirrored: no probe pod is rented for a cycle that would not ask for one."""
    specs = dict(VERIFIED_SPECS)
    rented = None
    if case == "skip_setting":
        monkeypatch.setattr(settings, "SKIP_RENTAL_VERIFICATION", True)
    elif case == "customer_rental":
        rented = RentedExecutorsResponse(
            executors={
                EXECUTOR_UUID: RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address="127.0.0.1",
                    executor_ip_port="22",
                    pods=[RentedPod(pod_id="p1", container_name="pod_p1", rented_ports=[40001])],
                )
            },
            banned_guids=[],
        )
    elif case == "filler":
        rented = RentedExecutorsResponse(
            executors={}, banned_guids=[], filler_containers_by_executor={EXECUTOR_UUID: "filler_x"}
        )
    elif case == "create_killed_enforced":
        rented = RentedExecutorsResponse(
            executors={}, banned_guids=[], filler_create_kill_executor_ids=[EXECUTOR_UUID]
        )
    elif case == "no_ports":
        specs.pop("verified_ports")
    ctx = make_context(state=build_state(specs=specs, rented_data=rented))
    assert rental_probe_request(ctx) is None


def test_a_create_kill_in_shadow_still_probes(monkeypatch, probe_on):
    """Shadow mode falls through to the backend call in `run` (a pass there would be an upgrade),
    so the early probe is started for it too."""
    monkeypatch.setattr(settings, "FILLER_LIVENESS_ENFORCEMENT_ENABLED", False)
    rented = RentedExecutorsResponse(
        executors={}, banned_guids=[], filler_create_kill_executor_ids=[EXECUTOR_UUID]
    )
    ctx = make_context(state=build_state(specs=VERIFIED_SPECS, rented_data=rented))
    assert rental_probe_request(ctx) is not None


# --- the probe beside the GPU steps -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_starts_no_probe_and_the_check_calls_as_today(
    keypair, monkeypatch, local_verify_on, verifyx_service
):
    assert Settings.model_fields["LOCAL_VERIFY_RENTAL_PROBE_PARALLEL"].default is False  # shipped off
    monkeypatch.setattr(settings, "LOCAL_VERIFY_RENTAL_PROBE_PARALLEL", False)
    monkeypatch.setattr(settings, "SKIP_RENTAL_VERIFICATION", False)
    backend = backend_with()
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        check = LocalVerifyCheck(client_factory=client_factory(keypair))
        local = await check.run(ctx)
        local_done = time.perf_counter()
        ctx2 = ctx.model_copy(update={"state": local.updates.get("state", ctx.state)})
        rental = await RentalVerificationCheck().run(ctx2)
    assert local.event.what_we_saw["consumed"] == ["matmul", "verifyx"]
    assert local.event.what_we_saw["rental_probe_started"] is False
    assert local.updates["state"].local_verify.rental_probe is None
    assert rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_count == 1
    # the one call happened inside RentalVerificationCheck, after the local check returned
    assert backend.entered and backend.entered[0] > local_done


@pytest.mark.asyncio
async def test_the_probe_runs_beside_the_gpu_steps_and_is_consumed_once(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """Backend probe 0.6 s, executor GPU steps 0.6 s: serial would be ≥ 1.2 s, side by side is
    < 1.0 s. One backend call in total; RentalVerificationCheck's verdict is the probe's answer
    through the unchanged VERIFIED path; the metric line says consumed/ok."""
    backend = backend_with(sleep=0.6)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=0.6) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        started = time.perf_counter()
        with patch("neurons.validators.src.services.task.checks.rental_verification.logger") as log:
            local, rental, ctx2 = await run_local_then_rental(
                ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
            )
        elapsed = time.perf_counter() - started
        # the probe was entered before the executor answered the intent
        assert backend.entered[0] < started + 0.3
    assert elapsed < 1.0, elapsed
    assert local.event.what_we_saw["consumed"] == ["matmul", "verifyx"]
    assert local.event.what_we_saw["rental_probe_started"] is True
    probe = ctx2.state.local_verify.rental_probe
    assert isinstance(probe, RentalProbe) and probe.consumed and probe.task.done()
    assert rental.passed and rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert rental.event.what_we_saw["details"] == {"container_healthy": True}
    assert backend.check_executor_health.await_count == 1
    assert backend.check_executor_health.await_args.kwargs == asdict(probe.request)
    # DAH-1991: the health_check_* force-remove still runs after the consumed probe
    ctx2.services.container_cleanup.force_remove_health_checks.assert_awaited_once()
    lines = outcome_lines(log)
    assert [(l["outcome"], l["step"], l["reason"]) for l in lines] == [("consumed", "rental_probe", "ok")]
    assert lines[0]["first_pass"] is True and "probe_age_ms" in lines[0]


@pytest.mark.asyncio
async def test_a_probe_exception_takes_the_api_error_path_without_a_second_rent(
    keypair, monkeypatch, probe_on, verifyx_service
):
    backend = backend_with(raises=RuntimeError("backend down"))
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        local, rental, ctx2 = await run_local_then_rental(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.event.what_we_saw["rental_probe_started"] is True
    assert rental.passed is False and rental.event.reason_code == RentalMsg.API_ERROR.reason
    assert rental.event.what_we_saw["error"] == "backend down"
    assert backend.check_executor_health.await_count == 1
    ctx2.services.container_cleanup.force_remove_health_checks.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_probe_answer_that_fails_is_judged_exactly_as_a_direct_one(
    keypair, monkeypatch, probe_on, verifyx_service
):
    failed = ExecutorHealthCheckResponse(success=False, error="sshd never answered", details={"x": 1})
    backend = backend_with(response=failed)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        _, rental, _ = await run_local_then_rental(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert rental.passed is False and rental.event.reason_code == RentalMsg.FAILED.reason
    assert rental.event.what_we_saw["error"] == "sshd never answered"
    assert backend.check_executor_health.await_count == 1


@pytest.mark.asyncio
async def test_not_the_first_pass_starts_no_probe(keypair, monkeypatch, probe_on, verifyx_service):
    """Full-size cycles keep the serial order (the `parallel_gpu` rule: the probe pod would take
    the GPUs while a full-size VerifyX runs). The gate of phase 2d is turned off here so the call
    itself is made; the probe still is not."""
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FIRST_PASS_ONLY", False)
    backend = backend_with()
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair,
            executor.executor_info,
            validation=validation,
            verifyx=verifyx_service,
            backend=backend,
            first_pass=False,
        )
        local, rental, _ = await run_local_then_rental(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
        assert len(executor.intents) == 1
    assert local.event.what_we_saw["rental_probe_started"] is False
    assert rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_count == 1


@pytest.mark.asyncio
async def test_the_express_lane_with_the_fast_path_off_makes_the_call_but_starts_no_probe(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """The 2d split: `unscored=True` (the caller's first pass) makes the one call through the
    LOCAL_VERIFY_FIRST_PASS_ONLY gate, but with FIRST_PASS_FAST_PATH_ENABLED off `first_pass` is
    False, the GPU steps are full size and serial, and the probe pod may not share the GPUs with
    them — no probe; RentalVerificationCheck makes its own direct call."""
    monkeypatch.setattr(settings, "LOCAL_VERIFY_FIRST_PASS_ONLY", True)
    backend = backend_with()
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair,
            executor.executor_info,
            validation=validation,
            verifyx=verifyx_service,
            backend=backend,
            first_pass=False,
        )
        ctx = ctx.model_copy(update={"config": replace(ctx.config, unscored=True)})
        local, rental, _ = await run_local_then_rental(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
        assert len(executor.intents) == 1
        assert executor.intents[0]["parallel_gpu"] is False
    assert local.event.reason_code == "LOCAL_VERIFY_OK"
    assert local.event.what_we_saw["rental_probe_started"] is False
    assert rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_count == 1


@pytest.mark.asyncio
async def test_a_call_that_falls_back_still_hands_the_probe_to_the_rental_check(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """`/verify` answers 404 after the probe was started: the FALLBACK result carries the probe in
    its state (no consumed GPU step), RentalVerificationCheck awaits it — one rent, not two."""
    backend = backend_with()
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        executor.answer_override = lambda intent: (404, {"detail": "not found"})
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        local, rental, ctx2 = await run_local_then_rental(
            ctx, LocalVerifyCheck(client_factory=client_factory(keypair))
        )
    assert local.event.reason_code == "LOCAL_VERIFY_FALLBACK"
    assert local.event.what_we_saw["reason"] == "not_supported"
    outcome = local.updates["state"].local_verify
    assert isinstance(outcome, LocalVerifyOutcome)
    assert outcome.matmul is None and outcome.verifyx is None
    assert isinstance(outcome.rental_probe, RentalProbe)
    assert rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_count == 1


@pytest.mark.asyncio
async def test_a_bug_after_the_probe_started_still_hands_it_over(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """The internal-error guard of `run` attaches a probe `_run` had started before raising."""
    backend = backend_with()
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        check = LocalVerifyCheck(client_factory=client_factory(keypair))
        with patch.object(
            LocalVerifyCheck, "_judge", side_effect=RuntimeError("judge bug")
        ):
            local, rental, _ = await run_local_then_rental(ctx, check)
    assert local.event.what_we_saw["reason"] == "internal_error"
    assert isinstance(local.updates["state"].local_verify.rental_probe, RentalProbe)
    assert rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_count == 1


@pytest.mark.asyncio
async def test_a_probe_for_a_different_request_is_settled_and_the_check_calls_itself(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """The verdict never rests on a request other than the one the check would send now: if the
    state changed in between (here the verified ports), the early task is cancelled and awaited,
    and the check makes its own call."""
    backend = backend_with(sleep=5.0)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        local = await LocalVerifyCheck(client_factory=client_factory(keypair)).run(ctx)
        state = local.updates["state"]
        probe = state.local_verify.rental_probe
        assert isinstance(probe, RentalProbe)
        changed = replace(state, specs={**state.specs, "verified_ports": [40002]})
        ctx2 = ctx.model_copy(update={"state": changed})
        # the second call answers at once
        backend.check_executor_health.side_effect = None
        backend.check_executor_health.return_value = OK
        with patch("neurons.validators.src.services.task.checks.rental_verification.logger") as log:
            started = time.perf_counter()
            rental = await RentalVerificationCheck().run(ctx2)
            assert time.perf_counter() - started < 1.0  # the 5 s task was cancelled, not awaited
    assert probe.consumed and probe.task.cancelled()
    assert rental.event.reason_code == RentalMsg.VERIFIED.reason
    assert backend.check_executor_health.await_args.kwargs["container_port"] == 40002
    lines = outcome_lines(log)
    assert [(l["outcome"], l["step"], l["reason"], l["settled"]) for l in lines] == [
        ("fallback", "rental_probe", "request_mismatch", "cancelled")
    ]


# --- halt safety: Pipeline.run settles what nobody consumed ------------------------------------------


class _Fatal:
    check_id = "test.fatal"
    fatal = True

    async def run(self, ctx):
        from neurons.validators.src.services.task.messages import (
            LocalVerifyMessages,
            render_message,
        )

        return CheckResult(
            passed=False,
            event=render_message(LocalVerifyMessages.SKIPPED, ctx=ctx, check_id=self.check_id, what={}),
        )


@pytest.mark.asyncio
async def test_a_fatal_halt_between_the_two_checks_cancels_the_pending_probe(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """TdxHost / Capability can end the pipeline between LocalVerifyCheck and
    RentalVerificationCheck; the pending task is cancelled and awaited in `Pipeline.run`'s
    finally, with one `[local_verify] outcome step=rental_probe` line, and the rental check
    (never reached) makes no call."""
    backend = backend_with(sleep=5.0)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        pipeline = Pipeline(
            [LocalVerifyCheck(client_factory=client_factory(keypair)), _Fatal(), RentalVerificationCheck()],
            sink=LoggerSink(MagicMock()),
        )
        with patch("neurons.validators.src.services.task.pipeline.logger") as log:
            started = time.perf_counter()
            ok, events, final_ctx = await pipeline.run(ctx)
            assert time.perf_counter() - started < 1.0
    assert ok is False and [e.check_id for e in events] == ["executor.local_verify", "test.fatal"]
    probe = final_ctx.state.local_verify.rental_probe
    assert probe.consumed and probe.task.cancelled()
    assert backend.check_executor_health.await_count == 1  # the probe's own entry, cancelled
    # DAH-1991 on the settle path: the backend spawned a health_check_* pod for the probe and no
    # RentalVerificationCheck ran to remove it, so the pipeline removes it — a regression here
    # leaves a probe pod on the executor that races the next rental.
    ctx.services.container_cleanup.force_remove_health_checks.assert_awaited_once_with(
        ctx.ssh, ctx.executor.uuid
    )
    lines = [c.args[0].extra for c in log.info.call_args_list if str(c.args[0]) == "[local_verify] outcome"]
    assert [(l["outcome"], l["step"], l["reason"]) for l in lines] == [
        ("fallback", "rental_probe", "unconsumed_cancelled")
    ]


@pytest.mark.asyncio
async def test_cancelling_the_check_mid_call_cancels_the_probe_it_started(
    keypair, monkeypatch, probe_on, verifyx_service
):
    """miner_service wraps each executor's pipeline in `asyncio.wait_for(...)`: a timeout cancels
    the check while `client.verify` is awaited, nothing is returned, and the pipeline's `finally`
    has no state to settle. The check must cancel the probe itself on the way out — a probe pod
    nobody reads is not rented on."""
    backend = backend_with(sleep=30)
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=30) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        check = LocalVerifyCheck(client_factory=client_factory(keypair))
        running = asyncio.create_task(check.run(ctx))
        while not backend.entered:  # the probe was started, the call is in flight
            await asyncio.sleep(0.01)
        probes = [t for t in asyncio.all_tasks() if t.get_name() == "local_verify.rental_probe"]
        assert len(probes) == 1 and not probes[0].done()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await asyncio.sleep(0)  # let the probe's cancellation land
        assert probes[0].cancelled()
    assert backend.check_executor_health.await_count == 1


@pytest.mark.asyncio
async def test_a_finished_but_unconsumed_probe_is_retrieved_not_cancelled(
    keypair, monkeypatch, probe_on, verifyx_service
):
    backend = backend_with(raises=RuntimeError("late failure"))
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair, step_sleep=0.2) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        pipeline = Pipeline(
            [LocalVerifyCheck(client_factory=client_factory(keypair)), _Fatal()],
            sink=LoggerSink(MagicMock()),
        )
        with patch("neurons.validators.src.services.task.pipeline.logger") as log:
            _, _, final_ctx = await pipeline.run(ctx)
    probe = final_ctx.state.local_verify.rental_probe
    assert probe.task.done() and not probe.task.cancelled()
    # Retrieved by the settle step already: asyncio clears `_log_traceback` when the exception is
    # read (`Task.exception()` / awaiting it), and it is that flag that drives the "exception was
    # never retrieved" report. Checked BEFORE this test reads the exception itself.
    assert probe.task._log_traceback is False
    assert isinstance(probe.task.exception(), RuntimeError)
    lines = [c.args[0].extra for c in log.info.call_args_list if str(c.args[0]) == "[local_verify] outcome"]
    assert lines[-1]["reason"] == "unconsumed_done"
    assert lines[-1]["health_checks_removed"] == 0  # the fake cleanup's answer, awaited once
    ctx.services.container_cleanup.force_remove_health_checks.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_consumed_probe_is_left_alone_by_the_settle_step(
    keypair, monkeypatch, probe_on, verifyx_service
):
    backend = backend_with()
    validation = matmul_service(monkeypatch)
    async with FakeExecutor(keypair) as executor:
        ctx = probe_context(
            keypair, executor.executor_info, validation=validation, verifyx=verifyx_service, backend=backend
        )
        pipeline = Pipeline(
            [LocalVerifyCheck(client_factory=client_factory(keypair)), RentalVerificationCheck()],
            sink=LoggerSink(MagicMock()),
        )
        with patch("neurons.validators.src.services.task.pipeline.logger") as log:
            ok, events, _ = await pipeline.run(ctx)
    assert ok is True and events[-1].reason_code == RentalMsg.VERIFIED.reason
    assert not [c for c in log.info.call_args_list if str(c.args[0]) == "[local_verify] outcome"]
    # consumed by the rental check, whose own finally removed the probe pod: once, not twice
    ctx.services.container_cleanup.force_remove_health_checks.assert_awaited_once()


def test_pipeline_without_local_verify_state_settles_nothing():
    """The dry-run pipeline and every pre-phase-2 context: no state, no work, no log line."""
    ctx = make_context()
    with patch("neurons.validators.src.services.task.pipeline.logger") as log:
        asyncio.run(Pipeline([], sink=LoggerSink(MagicMock())).run(ctx))
    log.info.assert_not_called()
