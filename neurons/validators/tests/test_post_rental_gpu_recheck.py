"""POST_RENTAL_GPU_RECHECK (DAH-4482): a node flagged during a rental proves every card before it re-lists."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.task.checks import (
    GpuFaultProbeCheck,
    GpuModelValidCheck,
    PinnedMemoryProbeCheck,
    PostRentalGpuRecheckCheck,
    ScoreCheck,
    TenantEnforcementCheck,
)
from neurons.validators.src.services.task.checks import gpu_fault_probe as probe_module
from neurons.validators.src.services.task.checks import post_rental_gpu_recheck as module
from neurons.validators.src.services.task.checks.gpu_memory_health import GpuMemoryHealthCheck
from neurons.validators.src.services.task.messages import PostRentalGpuRecheckMessages as Msg
from neurons.validators.src.services.task.pipeline import ParallelStage
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

from tests.helpers import build_context_config, build_services, build_state, default_executor
from tests.test_gpu_fault_probe_check import FakeRunner, probe_stdout
from tests.test_gpu_memory_health_check import FLAGS, _live_rental, _report

KEY = module.marker_key("miner-hotkey", default_executor().uuid)
MARKER = {"at": "2026-10-15T00:00:00+00:00", "source": "gpu.validate.memory_health", "flags": []}


class FakeValidation:
    def __init__(self, all_cards=None):
        self.all_cards = all_cards
        self.calls = 0

    async def probe_all_claimed_cards(self, ssh_client, executor_info, default_extra, machine_spec):
        self.calls += 1
        return self.all_cards


def _all_cards(*reasons, over_wall_clock=False):
    per_card = [{"card_index": i, "ok": not r, "reason": r} for i, r in enumerate(reasons)]
    passed = not any(reasons) and not over_wall_clock
    return SimpleNamespace(
        passed=passed,
        failure_reason="" if passed else "1/2 card(s) failed device work-proof",
        over_wall_clock=over_wall_clock,
        per_card=per_card,
    )


def _idle_ctx(context_factory, runner, validation, *, marker=MARKER, gpu_count=2):
    services = build_services(validation=validation)
    if marker is not None:
        services.redis.store[KEY] = json.dumps(marker)
    return context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": gpu_count}}),
        runner=runner,
    )


@pytest.fixture
def probe_flags_off():
    # no GPU fault probe flags are left to patch; kept so the tests below read unchanged
    yield


# --- the marker, set on a rented cycle ----------------------------------------------------------------------


def _memory_health_ctx(context_factory, *, rented: bool):
    return context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(
            specs={"gpu": {"count": 8}}, rented_data=_live_rental() if rented else None
        ),
        runner=FakeRunner(_report("flagged", flags=FLAGS)),
    )


@pytest.mark.asyncio
async def test_a_card_flagged_under_a_live_rental_records_the_marker(context_factory):
    ctx = _memory_health_ctx(context_factory, rented=True)

    result = await GpuMemoryHealthCheck().run(ctx)

    assert result.passed is False and result.fatal is False
    redis = ctx.services.redis
    assert redis.ttl[KEY] == module.MARKER_TTL_SECONDS
    marker = json.loads(redis.store[KEY])
    assert marker["source"] == "gpu.validate.memory_health" and marker["at"]
    assert (marker["verdict"], marker["flags"]) == ("flagged", FLAGS)


@pytest.mark.asyncio
async def test_an_idle_flagged_or_a_clean_rented_read_records_nothing(context_factory):
    # idle: the read itself already fails the node every cycle until the card is fixed
    idle = _memory_health_ctx(context_factory, rented=False)
    clean = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(specs={"gpu": {"count": 8}}, rented_data=_live_rental()),
        runner=FakeRunner(_report("ok")),
    )

    for ctx in (idle, clean):
        await GpuMemoryHealthCheck().run(ctx)
        assert KEY not in ctx.services.redis.store


# --- the recheck, on the next unrented cycle ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_marker_runs_nothing(context_factory, probe_flags_off):
    runner, validation = FakeRunner(), FakeValidation()

    result = await PostRentalGpuRecheckCheck().run(
        _idle_ctx(context_factory, runner, validation, marker=None)
    )

    assert result.passed and result.event.reason_code == Msg.NO_MARKER.reason
    assert runner.calls == [] and validation.calls == 0


@pytest.mark.asyncio
async def test_on_a_fault_fails_the_node_and_keeps_the_marker(context_factory, probe_flags_off):
    runner = FakeRunner(probe_stdout("fault", faults=["gpu 1: wrong result in gather"]))
    validation = FakeValidation(_all_cards("", ""))
    ctx = _idle_ctx(context_factory, runner, validation)

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed is False and PostRentalGpuRecheckCheck.fatal is True
    assert result.event.reason_code == Msg.FAILED.reason
    assert "gpu 1: wrong result" in result.event.what_we_saw["error"]
    assert validation.calls == 1 and len(runner.calls) == 1
    assert KEY in ctx.services.redis.store
    assert ctx.services.redis.ttl[KEY] == module.MARKER_TTL_SECONDS


@pytest.mark.asyncio
async def test_on_a_failed_all_cards_matmul_fails_the_node(context_factory, probe_flags_off):
    runner = FakeRunner(probe_stdout("ok"))
    validation = FakeValidation(_all_cards("", "exit_status=1"))
    ctx = _idle_ctx(context_factory, runner, validation)

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed is False and result.event.reason_code == Msg.FAILED.reason
    assert "1/2 card(s)" in result.event.what_we_saw["error"]
    assert KEY in ctx.services.redis.store


@pytest.mark.asyncio
async def test_on_a_pass_clears_the_marker(context_factory, probe_flags_off, caplog):
    runner = FakeRunner(probe_stdout("ok"))
    validation = FakeValidation(_all_cards("", ""))
    ctx = _idle_ctx(context_factory, runner, validation)

    with caplog.at_level("INFO", logger=module.__name__):
        result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.PASSED.reason
    assert KEY not in ctx.services.redis.store
    assert any(r.getMessage() == "POST_RENTAL_GPU_RECHECK_PASSED" for r in caplog.records)


@pytest.mark.asyncio
async def test_on_a_probe_without_a_verdict_keeps_the_marker_and_skips_the_streak(
    context_factory, probe_flags_off
):
    runner = FakeRunner("python: not found", exit_code=127)
    ctx = _idle_ctx(context_factory, runner, FakeValidation())
    probe_module._unknown_streak.clear()

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert KEY in ctx.services.redis.store
    assert ctx.services.redis.ttl[KEY] == module.MARKER_TTL_SECONDS
    # the pipeline's own probe already counted this cycle
    assert probe_module._unknown_streak == {}


@pytest.mark.parametrize(("gpu_count", "cleared"), [(2, False), (1, True)])
@pytest.mark.asyncio
async def test_no_all_cards_matmul_is_inconclusive_on_a_multi_card_node(
    context_factory, probe_flags_off, gpu_count, cleared
):
    runner = FakeRunner(probe_stdout("ok"))
    ctx = _idle_ctx(context_factory, runner, FakeValidation(None), gpu_count=gpu_count)

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed
    assert result.event.reason_code == (Msg.PASSED if cleared else Msg.INCONCLUSIVE).reason
    assert (KEY in ctx.services.redis.store) is not cleared


@pytest.mark.parametrize(
    "all_cards",
    [
        _all_cards("", "timeout"),
        _all_cards("ssh_error: channel closed", ""),
        _all_cards("", "", over_wall_clock=True),
    ],
)
@pytest.mark.asyncio
async def test_an_all_cards_matmul_lost_to_ssh_or_time_is_inconclusive(
    context_factory, probe_flags_off, all_cards
):
    ctx = _idle_ctx(context_factory, FakeRunner(probe_stdout("ok")), FakeValidation(all_cards))

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert ctx.services.redis.ttl[KEY] == module.MARKER_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_filler_cycle_renews_the_marker(context_factory, probe_flags_off):
    runner, validation = FakeRunner(), FakeValidation()
    ctx = _idle_ctx(context_factory, runner, validation)

    with patch.object(module, "get_filler_only_container", return_value={"id": "filler"}):
        result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert runner.calls == [] and validation.calls == 0
    assert ctx.services.redis.ttl[KEY] == module.MARKER_TTL_SECONDS
    assert json.loads(ctx.services.redis.store[KEY])["filler_skips"] == 1


@pytest.mark.asyncio
async def test_a_node_that_always_runs_a_filler_is_probed_after_the_skip_limit(
    context_factory, probe_flags_off
):
    filler = "filler_run-1"
    runner, validation = FakeRunner(probe_stdout("ok")), FakeValidation(_all_cards("", ""))
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="", stderr="")
    services = build_services(validation=validation)
    services.redis.store[KEY] = json.dumps({**MARKER, "filler_skips": module.FILLER_SKIP_LIMIT})
    uuid = default_executor().uuid
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(
            specs={"gpu": {"count": 2}},
            rented_data=RentedExecutorsResponse(
                all_filler_containers_by_executor={uuid: [filler]},
                default_job_owner_by_executor={uuid: "lium"},
                executors={},
            ),
        ),
        runner=runner,
        ssh=ssh,
    )

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert ctx.ssh.run.await_args.args[0] == f"/usr/bin/docker rm -fv {filler}"
    assert result.event.what_we_saw["removed_fillers"] == [filler]
    assert validation.calls == 1
    assert result.passed and result.event.reason_code == Msg.PASSED.reason
    assert KEY not in ctx.services.redis.store



@pytest.mark.asyncio
async def test_a_filler_that_cannot_be_removed_is_inconclusive_not_a_failure(context_factory, probe_flags_off):
    runner, validation = FakeRunner(probe_stdout("ok")), FakeValidation(_all_cards("", ""))
    services = build_services(validation=validation)
    services.redis.store[KEY] = json.dumps({**MARKER, "filler_skips": module.FILLER_SKIP_LIMIT})
    uuid = default_executor().uuid
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(
            specs={"gpu": {"count": 2}},
            rented_data=RentedExecutorsResponse(
                all_filler_containers_by_executor={uuid: ["filler_run-1"]},
                default_job_owner_by_executor={uuid: "other"},
                executors={},
            ),
        ),
        runner=runner,
    )

    with patch.object(module, "get_filler_only_container", return_value={"id": "filler"}):
        result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert runner.calls == [] and validation.calls == 0
    assert KEY in ctx.services.redis.store

# --- placement ----------------------------------------------------------------------------------------------


def _flat(checks) -> list[type]:
    kinds: list[type] = []
    for check in checks:
        if isinstance(check, ParallelStage):
            for lane in check.lanes:
                kinds += _flat(lane)
        else:
            kinds.append(type(check))
    return kinds


@pytest.mark.parametrize(
    "build",
    [
        PipelineFactory.build_checks,
        PipelineFactory.build_dry_run_checks,
        PipelineFactory.build_fast_path_checks,
    ],
)
def test_the_recheck_runs_unrented_after_the_gpu_probes_before_the_score(build):
    kinds = _flat(build())

    at = kinds.index(PostRentalGpuRecheckCheck)
    assert kinds.index(TenantEnforcementCheck) < kinds.index(GpuFaultProbeCheck) < at
    assert kinds[at - 1] is PinnedMemoryProbeCheck
    assert at < kinds.index(ScoreCheck)


# --- a card that falls off the bus -------------------------------------------------------------------------

UUIDS = [f"GPU-{i}" for i in range(3)]


@pytest.mark.asyncio
async def test_a_dropped_card_owes_a_recheck(context_factory):
    """A 3-card node whose scrape lists 2 rows: GPU-2 fell off the bus (GPU_DETAILS_MISMATCH)."""
    services = build_services()
    details = [{"name": "NVIDIA GeForce RTX 3090", "uuid": u} for u in UUIDS[:2]]
    ctx = context_factory(
        services=services,
        config=build_context_config(gpu_model_rates={"NVIDIA GeForce RTX 3090": 0.05}),
        state=build_state(
            specs={"gpu": {"count": 3, "details": details}}, gpu_count=3, gpu_details=details
        ),
        verified={"uuids": ",".join(UUIDS)},
    )

    await GpuModelValidCheck().run(ctx)

    marker = json.loads(services.redis.store[KEY])
    assert marker["source"] == "gpu_details_mismatch"
    assert marker["cards"] == [{"uuid": "GPU-2", "verdict": "missing"}]


# --- renter reports: a GeForce card has no ECC or row-remap counters to flag ---------------------------------


def _reported_ctx(context_factory, runner, validation, *, owner="miner-hotkey"):
    uuid = default_executor().uuid
    details = [{"name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-0"}]
    return context_factory(
        services=build_services(validation=validation),
        config=build_context_config(),
        state=build_state(
            specs={"gpu": {"count": 1, "details": details}},
            rented_data=RentedExecutorsResponse(
                executors={},
                banned_guids=[],
                renter_hardware_fault_executor_ids=[uuid],
                executor_owner_hotkeys={uuid: owner},
            ),
        ),
        runner=runner,
    )


@pytest.mark.asyncio
async def test_a_geforce_node_flagged_only_by_renters_gets_the_marker(context_factory, probe_flags_off):
    runner = FakeRunner(probe_stdout("fault", faults=["gpu 0: wrong result in gather"]))
    ctx = _reported_ctx(context_factory, runner, FakeValidation(None))

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed is False and result.event.reason_code == Msg.FAILED.reason
    marker = json.loads(ctx.services.redis.store[KEY])
    assert (marker["source"], marker["verdict"]) == ("renter_report", "renter_reported")
    assert ctx.services.redis.ttl[KEY] == module.MARKER_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_report_on_a_node_listed_under_another_hotkey_marks_nothing(context_factory):
    ctx = _reported_ctx(context_factory, FakeRunner(), FakeValidation(), owner="other-hotkey")

    result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.event.reason_code == Msg.NO_MARKER.reason
    assert KEY not in ctx.services.redis.store


@pytest.mark.asyncio
async def test_a_reported_node_that_just_passed_is_not_re_marked(context_factory, probe_flags_off):
    ctx = _reported_ctx(context_factory, FakeRunner(probe_stdout("ok")), FakeValidation(None))

    first = await PostRentalGpuRecheckCheck().run(ctx)
    second = await PostRentalGpuRecheckCheck().run(ctx)

    assert first.event.reason_code == Msg.PASSED.reason
    assert second.event.reason_code == Msg.NO_MARKER.reason
    assert KEY not in ctx.services.redis.store
    cleared = module.cleared_key("miner-hotkey", default_executor().uuid)
    assert ctx.services.redis.ttl[cleared] == module.MARKER_TTL_SECONDS
