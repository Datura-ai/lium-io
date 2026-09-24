"""Validation fast path (VALIDATION_FAST_PATH_ENABLED): a new node's first verification waits less
and decides exactly as the serial pipeline does.

- a ParallelStage gives the same verdict, the same failing check and the same final context as the
  serial list on the same checks;
- the collateral read started early is the one CollateralCheck awaits, with the gate unchanged;
- the fast-path check list is the serial list re-ordered — no check added, none dropped;
- the express lane's shorter waits apply only with the flag on.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from neurons.validators.src.core import express_lane as express_lane_module
from neurons.validators.src.core.express_lane import (
    MINER_DID_NOT_RETURN_EXECUTOR,
    RETRY_SECONDS,
    max_attempts_for,
    retry_seconds_for,
)
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.services.task.checks import (
    CachedTemplateVerificationCheck,
    CapabilityCheck,
    CollateralCheck,
    CollateralPrefetchCheck,
    GpuFaultProbeCheck,
    PortConnectivityCheck,
    PortCountCheck,
    RentalVerificationCheck,
    SysboxRequiredCheck,
    TenantEnforcementCheck,
    VerifyXCheck,
)
from neurons.validators.src.services.task.checks.collateral_prefetch import collateral_read_args
from neurons.validators.src.services.task.messages import CollateralMessages
from neurons.validators.src.services.task.models import ValidationEvent
from neurons.validators.src.services.task.pipeline import (
    CheckResult,
    ParallelStage,
    Pipeline,
    merge_state,
)
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

from core.config import settings
from tests.helpers import build_services, build_state

# --- fixtures -------------------------------------------------------------------------------------


class _Sink:
    def __init__(self):
        self.emitted: list[ValidationEvent] = []

    async def emit(self, event: ValidationEvent) -> None:
        self.emitted.append(event)


def _event(check_id: str, passed: bool) -> ValidationEvent:
    return ValidationEvent(
        event=f"{check_id} ran",
        reason_code=f"{check_id.upper()}_{'OK' if passed else 'FAILED'}",
        severity="info" if passed else "error",
        impact="",
        check_id=check_id,
        when=datetime.now(UTC),
    )


class _StateCheck:
    """A check that sleeps `seconds`, records when it ran, and writes its own key into state.specs
    plus one top-level state field, the way the real checks do (VerifyX: specs; ports: verified_port_count)."""

    def __init__(self, check_id, *, seconds=0.0, passed=True, fatal=True, halt=False, spec=None, state_field=None, log=None):
        self.check_id = check_id
        self.fatal = fatal
        self._seconds = seconds
        self._passed = passed
        self._halt = halt
        self._spec = spec
        self._state_field = state_field
        self._log = log if log is not None else []

    async def run(self, ctx) -> CheckResult:
        self._log.append((self.check_id, "start", asyncio.get_running_loop().time()))
        try:
            await asyncio.sleep(self._seconds)
        except asyncio.CancelledError:
            self._log.append((self.check_id, "cancelled", asyncio.get_running_loop().time()))
            raise
        self._log.append((self.check_id, "end", asyncio.get_running_loop().time()))
        updates = {}
        if self._passed and (self._spec or self._state_field):
            specs = dict(ctx.state.specs)
            if self._spec:
                specs.update(self._spec)
            changes = {"specs": specs}
            if self._state_field:
                changes.update(self._state_field)
            updates["state"] = replace(ctx.state, **changes)
        return CheckResult(passed=self._passed, event=_event(self.check_id, self._passed), updates=updates, halt=self._halt)


def _lanes(log, *, ports_pass=True, matmul_pass=True, halt_on=None):
    host = [
        _StateCheck("ports", seconds=0.03, passed=ports_pass, log=log, spec={"verified_ports": [1, 2]}, state_field={"verified_port_count": 2}),
        _StateCheck("sysbox", seconds=0.01, log=log, spec={"sysbox": True}, halt=halt_on == "sysbox"),
    ]
    gpu = [
        _StateCheck("matmul", seconds=0.02, passed=matmul_pass, log=log, spec={"matmul": "ok"}, state_field={"gpu_count": 8}),
        _StateCheck("fault_probe", seconds=0.01, log=log, spec={"fault": "none"}),
    ]
    return host, gpu


async def _run(checks, state=None, context_factory=None):
    sink = _Sink()
    ctx = context_factory(state=state or build_state(specs={"gpu": {"count": 8}}))
    ok, events, last = await Pipeline(checks, sink).run(ctx)
    return ok, events, last


def _verdict(ok, events, last):
    return (
        ok,
        [e.check_id for e in events],
        events[-1].check_id,
        events[-1].reason_code,
        dict(last.state.specs),
        last.state.verified_port_count,
        last.state.gpu_count,
    )


# --- (a) parallel fan-out ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"ports_pass": False},
        {"matmul_pass": False},
        {"halt_on": "sysbox"},
    ],
    ids=["all-pass", "host-lane-fails", "gpu-lane-fails", "host-lane-halts"],
)
async def test_parallel_stage_gives_the_serial_verdict_and_context(context_factory, kwargs):
    log_serial, log_parallel = [], []
    host_s, gpu_s = _lanes(log_serial, **kwargs)
    host_p, gpu_p = _lanes(log_parallel, **kwargs)
    tail = lambda log: [_StateCheck("score", log=log), _StateCheck("finalize", log=log)]  # noqa: E731

    serial = await _run([*host_s, *gpu_s, *tail(log_serial)], context_factory=context_factory)
    parallel = await _run([ParallelStage([host_p, gpu_p]), *tail(log_parallel)], context_factory=context_factory)

    s_ok, s_ids, s_last, s_reason, s_specs, s_ports, s_gpus = _verdict(*serial)
    p_ok, p_ids, p_last, p_reason, p_specs, p_ports, p_gpus = _verdict(*parallel)
    assert p_ok == s_ok
    assert (p_last, p_reason) == (s_last, s_reason), "the run ends on the same check with the same event"
    if s_ok:
        assert (p_specs, p_ports, p_gpus) == (s_specs, s_ports, s_gpus), "both lanes' state changes land"
        assert set(p_ids) == set(s_ids)
    else:
        # A failing run stops the other lane where it is: the parallel run never emits a check the
        # serial run did not reach, and its state carries no key the serial run's does not.
        assert set(p_ids) <= set(s_ids)
        assert set(p_specs) <= set(s_specs)
        base = build_state()
        assert p_ports in (s_ports, base.verified_port_count) and p_gpus in (s_gpus, base.gpu_count)
    # The lanes did overlap in time: the GPU lane started before the host lane's first check was over.
    starts = {c: t for c, kind, t in log_parallel if kind == "start"}
    over = {c: t for c, kind, t in log_parallel if kind in {"end", "cancelled"}}
    assert starts["matmul"] < over["ports"]


@pytest.mark.asyncio
async def test_parallel_stage_wall_time_is_the_slowest_lane(context_factory):
    log = []
    host, gpu = _lanes(log)
    ok, events, _ = await _run([ParallelStage([host, gpu]), _StateCheck("finalize", log=log)], context_factory=context_factory)
    assert ok
    stage_events = [e for e in events if e.check_id in {"ports", "sysbox", "matmul", "fault_probe"}]
    # The stage's wall time is its slowest lane's, less than the two lanes' work added up; the
    # per-step figures still say what each check cost.
    wall_ms = max(e.context["elapsed_time_ms"] for e in stage_events)
    total_ms = sum(e.context["execution_time_ms"] for e in stage_events)
    assert wall_ms < total_ms
    assert events[-1].what_we_saw["steps_total_s"] * 1000 <= total_ms + 50


class _RaisingCheck(_StateCheck):
    def __init__(self, check_id, *, seconds=0.0, exc=None, log=None):
        super().__init__(check_id, seconds=seconds, log=log)
        self._exc = exc or RuntimeError(f"{check_id} blew up")

    async def run(self, ctx):
        await super().run(ctx)
        raise self._exc


class _FakePrefetchTask:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return self.cancelled

    def cancel(self):
        self.cancelled = True


def _with_prefetch(state):
    task = _FakePrefetchTask()
    return replace(state, collateral_prefetch=SimpleNamespace(task=task, args=None)), task


async def _settled(log, check_id):
    """Wait long enough for `check_id` to have started if nothing had stopped it."""
    await asyncio.sleep(0.1)
    return any(c == check_id and kind == "start" for c, kind, _ in log)


@pytest.mark.asyncio
async def test_parallel_stage_cancels_the_sibling_lane_when_a_check_raises(context_factory):
    log = []
    host = [_StateCheck("ports", seconds=0.05, log=log), _StateCheck("rental_verification", log=log)]
    gpu = [_RaisingCheck("matmul", seconds=0.01, log=log)]
    state, prefetch = _with_prefetch(build_state(specs={"gpu": {"count": 8}}))
    ctx = context_factory(state=state)
    tasks_before = len(asyncio.all_tasks())

    with pytest.raises(RuntimeError, match="matmul blew up"):
        await Pipeline([ParallelStage([host, gpu]), _StateCheck("finalize", log=log)], _Sink()).run(ctx)

    kinds = {(c, kind) for c, kind, _ in log}
    assert ("ports", "cancelled") in kinds, "the in-flight sibling check was cancelled"
    assert not await _settled(log, "rental_verification"), "the sibling lane's next check never started"
    assert ("finalize", "start") not in kinds
    assert len(asyncio.all_tasks()) == tasks_before, "no lane task is left running"
    assert prefetch.cancelled, "the early collateral read nobody will consume is cancelled"


@pytest.mark.asyncio
async def test_parallel_stage_raises_the_first_exception_when_both_lanes_raise(context_factory):
    log = []
    host = [_RaisingCheck("ports", seconds=0.01, exc=ValueError("ports"), log=log)]
    gpu = [_RaisingCheck("matmul", seconds=0.01, exc=RuntimeError("matmul"), log=log)]
    tasks_before = len(asyncio.all_tasks())
    with pytest.raises((ValueError, RuntimeError)):
        await _run([ParallelStage([host, gpu])], context_factory=context_factory)
    assert len(asyncio.all_tasks()) == tasks_before
    assert {(c, kind) for c, kind, _ in log} >= {("ports", "start"), ("matmul", "start")}


@pytest.mark.asyncio
async def test_parallel_stage_stops_the_sibling_lane_at_a_fatal_failure(context_factory):
    log = []
    host = [_StateCheck("ports", seconds=0.05, log=log), _StateCheck("rental_verification", log=log)]
    gpu = [_StateCheck("matmul", seconds=0.01, passed=False, log=log), _StateCheck("fault_probe", log=log)]
    state, prefetch = _with_prefetch(build_state(specs={"gpu": {"count": 8}}))
    sink = _Sink()
    ctx = context_factory(state=state)
    ok, events, last = await Pipeline([ParallelStage([host, gpu]), _StateCheck("finalize", log=log)], sink).run(ctx)

    assert ok is False
    assert [e.check_id for e in events] == ["matmul"], "the run ends on the failing check, as serial would"
    assert events[-1].what_we_saw["steps"] == {"matmul": pytest.approx(0.01, abs=0.02)}, "only emitted checks in the summary"
    assert events[-1].what_we_saw["steps_failed"] == "matmul"
    kinds = {(c, kind) for c, kind, _ in log}
    assert ("ports", "cancelled") in kinds
    assert not await _settled(log, "rental_verification"), "a failed node never rents the probe container"
    assert ("fault_probe", "start") not in kinds and ("finalize", "start") not in kinds
    assert "verified_ports" not in last.state.specs
    assert prefetch.cancelled


@pytest.mark.asyncio
async def test_pipeline_cancels_a_pending_prefetch_when_a_serial_check_fails(context_factory):
    state, prefetch = _with_prefetch(build_state(specs={"gpu": {"count": 8}}))
    ok, events, _ = await _run([_StateCheck("gpu_count", passed=False), _StateCheck("collateral")], state=state, context_factory=context_factory)
    assert ok is False and [e.check_id for e in events] == ["gpu_count"]
    assert prefetch.cancelled


def test_merge_state_applies_each_lane_key_by_key():
    base = build_state(specs={"gpu": {"count": 8}, "ram": {"total": 1}})
    lane_a = replace(base, specs={**base.specs, "verified_ports": [1]}, verified_port_count=1)
    lane_b = replace(base, specs={**base.specs, "matmul": "ok"}, gpu_count=8)

    merged = merge_state(base, base, lane_a)
    merged = merge_state(merged, base, lane_b)

    assert merged.specs == {"gpu": {"count": 8}, "ram": {"total": 1}, "verified_ports": [1], "matmul": "ok"}
    assert merged.verified_port_count == 1
    assert merged.gpu_count == 8
    # A lane that changed nothing leaves the current state as it is.
    assert merge_state(merged, base, base) is merged


# --- (c) collateral read started early ------------------------------------------------------------


class _CountingCollateral:
    def __init__(self, deposited=True):
        self.calls = []
        self.deposited = deposited

    async def is_eligible_executor(self, *, miner_hotkey, executor_uuid, gpu_model, gpu_count):
        self.calls.append((miner_hotkey, executor_uuid, gpu_model, gpu_count))
        await asyncio.sleep(0.01)
        return self.deposited, None if self.deposited else "no bond", "1.2"


def _gpu_state(count=2, model="NVIDIA H200"):
    return build_state(specs={"gpu": {"count": count, "details": [{"name": model}] * count}})


@pytest.mark.asyncio
@pytest.mark.parametrize("deposited", [True, False])
async def test_collateral_check_awaits_the_prefetched_read_and_keeps_its_gate(context_factory, deposited):
    service = _CountingCollateral(deposited)
    ctx = context_factory(services=build_services(collateral=service), state=_gpu_state())

    started = await CollateralPrefetchCheck().run(ctx)
    assert started.passed and started.event.reason_code == "COLLATERAL_READ_STARTED"
    ctx = ctx.model_copy(update=started.updates)
    assert ctx.state.collateral_prefetch is not None

    result = await CollateralCheck().run(ctx)

    assert len(service.calls) == 1, "one contract read, started early, awaited here"
    assert result.passed is deposited, "the fatal collateral gate is unchanged"
    assert result.event.reason_code == (CollateralMessages.VERIFIED.reason if deposited else CollateralMessages.MISSING.reason)
    assert result.event.what_we_saw["prefetched"] is True
    assert result.updates["collateral_deposited"] is deposited


@pytest.mark.asyncio
async def test_collateral_check_reads_again_when_the_prefetch_asked_a_different_question(context_factory):
    service = _CountingCollateral()
    ctx = context_factory(services=build_services(collateral=service), state=_gpu_state(count=2))
    started = await CollateralPrefetchCheck().run(ctx)
    ctx = ctx.model_copy(update=started.updates)
    # A later check learned the real GPU count; the early answer is for another question.
    ctx = ctx.model_copy(update={"state": replace(ctx.state, gpu_count=4)})

    result = await CollateralCheck().run(ctx)

    assert result.passed
    assert "prefetched" not in result.event.what_we_saw
    assert [c[3] for c in service.calls][-1] == 4, "the check asked its own question"
    assert ctx.state.collateral_prefetch.task.cancelled() or ctx.state.collateral_prefetch.task.done()
    assert collateral_read_args(ctx).gpu_count == 4


@pytest.mark.asyncio
async def test_collateral_prefetch_skips_a_node_with_no_gpu_yet(context_factory):
    service = _CountingCollateral()
    ctx = context_factory(services=build_services(collateral=service), state=build_state(specs={}))
    result = await CollateralPrefetchCheck().run(ctx)
    assert result.passed and result.event.reason_code == "COLLATERAL_READ_SKIPPED"
    assert not result.updates and service.calls == []


# --- the fast-path list is the serial list, re-ordered -----------------------------------------------


def _flatten(checks):
    out = []
    for step in checks:
        out.extend(step.checks if isinstance(step, ParallelStage) else [step])
    return out


def test_fast_path_checks_are_the_serial_checks_re_ordered():
    serial = PipelineFactory.build_checks()
    fast = PipelineFactory.build_fast_path_checks()
    flat = _flatten(fast)

    serial_ids = sorted(type(c).__name__ for c in serial)
    fast_ids = sorted(type(c).__name__ for c in flat)
    assert fast_ids == sorted(serial_ids + ["CollateralPrefetchCheck"]), "one check added (the early read), none dropped"
    assert [type(c) for c in PipelineFactory.build_checks(fast_path=True)] == [type(c) for c in fast]

    names = [type(c).__name__ for c in fast]
    stage = next(s for s in fast if isinstance(s, ParallelStage))
    stage_at = fast.index(stage)
    # The gates that must decide before any GPU or port work stay ahead of the stage.
    for gate in (CollateralCheck, TenantEnforcementCheck, VerifyXCheck):
        assert names.index(gate.__name__) < stage_at
    assert next(c for c in fast if isinstance(c, CollateralCheck)).fatal is True
    assert names.index("CollateralPrefetchCheck") < names.index("CollateralCheck")
    # VerifyX measures the network alone; the executor refuses VerifyX beside the matmul.
    lane_types = [[type(c) for c in lane] for lane in stage.lanes]
    assert VerifyXCheck not in [t for lane in lane_types for t in lane]
    assert lane_types == [
        [PortConnectivityCheck, PortCountCheck, SysboxRequiredCheck, RentalVerificationCheck],
        [CapabilityCheck, GpuFaultProbeCheck, CachedTemplateVerificationCheck],
    ]


def _rented(executor_uuid="exe-1", pods=(), fillers=()):
    executors = {}
    if pods:
        executors[executor_uuid] = RentedExecutor(
            miner_hotkey="m", executor_ip_address="1.1.1.1", executor_ip_port="1",
            pods=[RentedPod(pod_id=p, container_name=f"c-{p}") for p in pods],
        )
    return RentedExecutorsResponse(executors=executors, all_filler_containers_by_executor={executor_uuid: list(fillers)} if fillers else {})


@pytest.mark.parametrize(
    "flag,first_pass,rented,expected",
    [
        (False, True, _rented(), False),
        (True, False, _rented(), False),
        (True, True, None, False),
        (True, True, _rented(pods=("p1",)), False),
        (True, True, _rented(fillers=("filler_1",)), False),
        (True, True, _rented(), True),
    ],
    ids=["flag-off", "scored-cycle", "no-backend-answer", "rented", "filler", "idle-first-pass"],
)
def test_takes_fast_path_only_for_an_idle_first_pass_with_the_flag_on(monkeypatch, flag, first_pass, rented, expected):
    monkeypatch.setattr(settings, "VALIDATION_FAST_PATH_ENABLED", flag)
    assert PipelineFactory.takes_fast_path(first_pass, "exe-1", rented) is expected


# --- (b) shorter waits --------------------------------------------------------------------------------


def test_express_lane_waits_shorten_only_with_the_flag_on(monkeypatch):
    monkeypatch.setattr(settings, "EXPRESS_LANE_TICK_SECONDS", 30)
    monkeypatch.setattr(settings, "EXPRESS_LANE_FAST_TICK_SECONDS", 15)
    monkeypatch.setattr(settings, "EXPRESS_LANE_MINER_SNAPSHOT_RETRY_SECONDS", 35)

    monkeypatch.setattr(settings, "VALIDATION_FAST_PATH_ENABLED", False)
    assert settings.express_lane_tick_seconds() == 30
    assert retry_seconds_for(MINER_DID_NOT_RETURN_EXECUTOR) == RETRY_SECONDS == 120
    assert retry_seconds_for("verification failed") == RETRY_SECONDS

    monkeypatch.setattr(settings, "VALIDATION_FAST_PATH_ENABLED", True)
    assert settings.express_lane_tick_seconds() == 15
    assert retry_seconds_for(MINER_DID_NOT_RETURN_EXECUTOR) == 35, "> the central miner's 30-s snapshot TTL"
    assert retry_seconds_for("verification failed") == RETRY_SECONDS, "a failed run keeps the 120-s pause"
    # A tick set below the fast tick is kept: the fast path only ever shortens.
    monkeypatch.setattr(settings, "EXPRESS_LANE_TICK_SECONDS", 10)
    assert settings.express_lane_tick_seconds() == 10
    assert express_lane_module.MAX_ATTEMPTS == 3
    # The fast re-ask gets more asks so the window it covers (7 × 35 s) is not shorter than the
    # serial one (2 × 120 s); every other reason keeps MAX_ATTEMPTS.
    monkeypatch.setattr(settings, "EXPRESS_LANE_MINER_SNAPSHOT_MAX_ATTEMPTS", 8)
    assert max_attempts_for(MINER_DID_NOT_RETURN_EXECUTOR) == 8
    assert (8 - 1) * 35 >= (3 - 1) * 120
    assert max_attempts_for("verification failed") == 3
    monkeypatch.setattr(settings, "VALIDATION_FAST_PATH_ENABLED", False)
    assert max_attempts_for(MINER_DID_NOT_RETURN_EXECUTOR) == 3
