"""Validation fast path (VALIDATION_FAST_PATH_ENABLED): a new node's first verification waits less
and decides exactly as the serial pipeline does.

- a ParallelStage gives the same verdict, the same failing check and the same final context as the
  serial list on the same checks;
- the collateral read started early is the one CollateralCheck awaits, with the gate unchanged;
- the fast-path check list is the serial list re-ordered — no check added, none dropped;
- the express lane's shorter waits apply only with the flag on;
- the validation-progress record walks its phases and is readable over HTTP.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from neurons.validators.src.core import express_lane as express_lane_module
from neurons.validators.src.core import validation_progress as vp
from neurons.validators.src.core.express_lane import (
    MINER_DID_NOT_RETURN_EXECUTOR,
    RETRY_SECONDS,
    retry_seconds_for,
)
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.routes.validation_progress import router as progress_router
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
        await asyncio.sleep(self._seconds)
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
    assert (p_specs, p_ports, p_gpus) == (s_specs, s_ports, s_gpus), "both lanes' state changes land"
    # Every check the serial run reached, the parallel run reached too (the other lane may have run
    # further — that work is what the stage saves, and it never changes the verdict).
    assert set(s_ids) <= set(p_ids)
    # The lanes did overlap in time: the GPU lane started before the host lane finished.
    starts = {c: t for c, kind, t in log_parallel if kind == "start"}
    ends = {c: t for c, kind, t in log_parallel if kind == "end"}
    assert starts["matmul"] < ends["ports"]


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


# --- (e) validation progress ---------------------------------------------------------------------------


def test_progress_record_walks_the_express_lane_phases():
    registry = vp.ValidationProgress()
    registry.discovered("exe-1", "miner-a", "express")
    assert registry.get("exe-1")["phase"] == vp.DISCOVERED

    registry.asking_miner("exe-1", "miner-a", "express", attempt=1)
    registry.retry_scheduled("exe-1", MINER_DID_NOT_RETURN_EXECUTOR, 35, attempt=1)
    record = registry.get("exe-1")
    assert record["phase"] == vp.WAITING_TO_RETRY and record["attempts"] == 1
    assert record["detail"] == f"retry in 35 s: {MINER_DID_NOT_RETURN_EXECUTOR}"
    assert record["last_error"] == MINER_DID_NOT_RETURN_EXECUTOR

    registry.asking_miner("exe-1", "miner-a", "express", attempt=2)
    registry.run_started("exe-1", "miner-a", "express")
    assert registry.get("exe-1")["phase"] == vp.CONNECTING

    registry.step_started("exe-1", "gpu.scrape.machine_spec")
    record = registry.get("exe-1")
    assert record["phase"] == vp.RUNNING and record["current_check_id"] == "gpu.scrape.machine_spec"
    assert record["current_check_since"] is not None
    registry.step_finished("exe-1", "gpu.scrape.machine_spec", "SCRAPE_OK", passed=True)
    registry.step_started("exe-1", "gpu.validate.verifyx")
    registry.step_finished("exe-1", "gpu.validate.verifyx", "VERIFYX_FAILED", passed=False)
    record = registry.get("exe-1")
    assert record["current_check_id"] is None
    assert record["last_reason_code"] == "VERIFYX_FAILED" and record["last_error"] == "VERIFYX_FAILED"
    assert [(s["check_id"], s["passed"]) for s in record["steps"]] == [
        ("gpu.scrape.machine_spec", True),
        ("gpu.validate.verifyx", False),
    ]

    registry.run_finished("exe-1", passed=False, reason_code="VERIFYX_FAILED")
    assert registry.get("exe-1")["phase"] == vp.FAILED
    registry.published("exe-1", passed=False)
    assert registry.get("exe-1")["phase"] == vp.FAILED

    registry.run_started("exe-1", "miner-a", "express")
    assert registry.get("exe-1")["steps"] == [], "a new run starts a new timeline"
    registry.step_started("exe-1", "pipeline.finalize")
    registry.step_finished("exe-1", "pipeline.finalize", "OK", passed=True)
    registry.run_finished("exe-1", passed=True, reason_code="OK")
    assert registry.get("exe-1")["phase"] == vp.VERIFIED
    registry.published("exe-1", passed=True)
    assert registry.get("exe-1")["phase"] == vp.PUBLISHED

    registry.left_to_cycle("exe-2", MINER_DID_NOT_RETURN_EXECUTOR, attempt=3)
    assert registry.get("exe-2")["phase"] == vp.LEFT_TO_CYCLE


def test_progress_prunes_finished_records_after_retention(monkeypatch):
    registry = vp.ValidationProgress()
    registry.run_started("old", "m", "cycle")
    registry.run_finished("old", passed=True)
    registry.run_started("live", "m", "cycle")
    now = [1000.0]
    monkeypatch.setattr(vp.time, "monotonic", lambda: now[0])
    for r in registry._records.values():
        r._touched_monotonic = 0.0
    now[0] = vp.RETENTION_SECONDS + 1.0
    uuids = {r["executor_uuid"] for r in registry.snapshot()}
    assert uuids == {"live"}, "a finished record ages out; a running one stays"


@pytest.mark.asyncio
async def test_pipeline_reports_each_check_to_the_registry(context_factory, monkeypatch):
    registry = vp.ValidationProgress()
    sink = _Sink()
    stage = ParallelStage([[_StateCheck("b")], [_StateCheck("c", passed=False)]])
    checks = [_StateCheck("a"), stage, _StateCheck("d")]
    ctx = context_factory()
    registry.run_started(ctx.executor.uuid, ctx.miner_hotkey, "express")
    ok, events, _ = await Pipeline(checks, sink, progress=vp.PipelineProgress(registry)).run(ctx)
    assert ok is False
    record = registry.get(ctx.executor.uuid)
    assert [s["check_id"] for s in record["steps"]] == ["a", "b", "c"]
    assert record["last_error"] == "C_FAILED"


def test_progress_route_serves_one_and_all(monkeypatch):
    registry = vp.ValidationProgress()
    monkeypatch.setattr("neurons.validators.src.routes.validation_progress.progress", registry)
    registry.run_started("exe-1", "miner-a", "express")
    registry.step_started("exe-1", "gpu.validate.verifyx")
    registry.left_to_cycle("exe-2", MINER_DID_NOT_RETURN_EXECUTOR, attempt=3)
    app = FastAPI()
    app.include_router(progress_router)
    client = TestClient(app)

    one = client.get("/validation-progress/exe-1")
    assert one.status_code == 200
    assert one.json()["phase"] == vp.RUNNING
    assert one.json()["current_check_id"] == "gpu.validate.verifyx"
    assert client.get("/validation-progress/nope").status_code == 404
    everything = client.get("/validation-progress").json()
    assert everything["count"] == 2
    left = client.get("/validation-progress", params={"phase": vp.LEFT_TO_CYCLE}).json()
    assert left["count"] == 1
    express = client.get("/validation-progress", params={"lane": "express"}).json()
    assert express["count"] == 1
