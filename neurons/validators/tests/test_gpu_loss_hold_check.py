"""GPU_LOST: a node whose GPU dropped out is hidden and takes no new rentals until its cards stay back (DAH-4457)."""

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.availability import (
    AvailabilityErrorCode,
    availability_errors,
    silence_availability_errors_on_our_own_outage,
)
from neurons.validators.src.services.task.checks import (
    GpuLossHoldCheck,
    GpuModelValidCheck,
    MachineSpecScrapeCheck,
    RentedGpuDropCheck,
    TenantEnforcementCheck,
    gpu_loss_hold,
    machine_spec_scrape,
)
from neurons.validators.src.services.task.messages import GpuLossHoldMessages as Msg
from neurons.validators.src.services.task.models import JobResult
from neurons.validators.src.services.task.pipeline import CheckResult, Context, Pipeline, ValidationEvent
from neurons.validators.src.services.task.runner import SSHCommandResult
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import (
    FakeRedis,
    build_context_config,
    build_services,
    build_state,
    default_executor,
)

MODEL = "NVIDIA GeForce RTX 3090"
UUIDS = [f"GPU-a0a{i}-0000-0000-0000-00000000000{i}" for i in range(4)]
LOST_CARD = {"listed": UUIDS[:3], "scrape_error": "NVMLError(999)"}
# the scrape listed one card twice: a possible glitch, judged again next cycle (`confirm_first`)
DUPLICATED = {"listed": UUIDS[:3] + [UUIDS[0]]}


def _details(uuids: list[str]) -> list[dict]:
    return [{"name": MODEL, "uuid": uuid, "capacity": 24576} for uuid in uuids]


def _pod(n: int, gpu_count: int) -> RentedPod:
    return RentedPod(
        pod_id=f"00000000-0000-4000-8000-00000000000{n}",
        container_name=f"container_pod{n}",
        gpu_count=gpu_count,
        status="RUNNING",
    )


def _ctx(
    context_factory,
    redis,
    *,
    listed=UUIDS,
    count=4,
    scrape_error=None,
    pods=(4,),
    owner="miner-hotkey",
    anchor=UUIDS,
):
    executor = default_executor()
    specs: dict = {"gpu": {"count": count, "details": _details(listed)}}
    if scrape_error is not None:
        specs["gpu_scrape_error"] = scrape_error
    rented = {}
    if pods:
        rented[executor.uuid] = RentedExecutor(
            miner_hotkey=owner,
            executor_ip_address="127.0.0.1",
            executor_ip_port="8001",
            pods=[_pod(n, count) for n, count in enumerate(pods, start=1)],
        )
    state = build_state(
        specs=specs,
        gpu_count=count,
        gpu_details=_details(listed),
        gpu_uuids=",".join(listed),
        rented_data=RentedExecutorsResponse(executors=rented),
    )
    return context_factory(
        services=build_services(redis=redis),
        state=state,
        executor=executor,
        verified={"spec": f"{MODEL}:4", "uuids": ",".join(anchor)},
        config=build_context_config(gpu_model_rates={MODEL: 1.0}),
    )


async def _cycle(context_factory, redis, **kwargs):
    return await GpuLossHoldCheck().run(_ctx(context_factory, redis, **kwargs))


def _fatal(result) -> bool:
    return result.fatal if result.fatal is not None else GpuLossHoldCheck.fatal


def _hidden(result) -> bool:
    return [e.reason_code for e in availability_errors([result.event])] == [
        AvailabilityErrorCode.GPU_LOST
    ]


@pytest.mark.asyncio
async def test_a_lost_card_on_a_rented_node_hides_it_and_the_cycle_fails_its_gpu_checks(
    context_factory,
):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, **LOST_CARD)

    result = await GpuLossHoldCheck().run(ctx)

    assert not result.passed and not _fatal(result) and _hidden(result)
    assert result.event.what_we_saw["reach_target"] == "gpu"
    assert result.event.what_we_saw["nvml_error_code"] == 999
    assert result.event.what_we_saw["missing_uuids"] == [UUIDS[3]]
    # the same scrape still fails the fatal GPU check after it, verification reset included, as before
    model = await GpuModelValidCheck().run(ctx)
    assert not model.passed and model.event.reason_code == "GPU_DETAILS_MISMATCH"
    assert redis.ttl[gpu_loss_hold._key(ctx)] == gpu_loss_hold.settings.GPU_LOSS_HOLD_SECONDS


@pytest.mark.asyncio
async def test_a_rented_node_whose_card_reads_clean_again_stays_hidden_but_keeps_its_rental(
    context_factory,
):
    redis = FakeRedis()
    await _cycle(context_factory, redis, **LOST_CARD)

    result = await _cycle(context_factory, redis)

    # hidden, and not fatal under a live rental: the rented check still runs and the renter is not touched
    assert not result.passed and not _fatal(result) and _hidden(result)
    assert result.event.what_we_saw["hold_until"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pods, fatal", [pytest.param((), True, id="idle"), pytest.param((4,), False, id="rented")]
)
async def test_an_nvml_loss_code_over_a_full_card_list_is_fatal_on_an_idle_node(
    context_factory, pods, fatal
):
    redis = FakeRedis()

    # every card is listed, so no later GPU check fails this cycle
    result = await _cycle(context_factory, redis, pods=pods, scrape_error="NVMLError(999)")

    assert not result.passed and _hidden(result) and _fatal(result) is fatal


@pytest.mark.asyncio
async def test_old_marks_without_since_still_hold(context_factory):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, pods=())
    old = {"held": True, "last_seen": "2026-10-17T00:00:00+00:00"}
    redis.store[gpu_loss_hold._key(ctx)] = json.dumps(old)

    result = await GpuLossHoldCheck().run(ctx)

    assert not result.passed and _fatal(result) and result.event.what_we_saw["since"]


@pytest.mark.asyncio
async def test_after_the_rental_ends_the_held_node_fails_at_once(context_factory):
    redis = FakeRedis()
    await _cycle(context_factory, redis, **LOST_CARD)

    result = await _cycle(context_factory, redis, pods=())

    assert not result.passed and _fatal(result) and _hidden(result)


@pytest.mark.asyncio
async def test_a_lost_card_on_an_idle_node_hides_it(context_factory):
    redis = FakeRedis()

    lost = await _cycle(context_factory, redis, pods=(), **LOST_CARD)
    clean = await _cycle(context_factory, redis, pods=())

    assert _hidden(lost) and _hidden(clean) and _fatal(clean)


@pytest.mark.asyncio
async def test_the_node_is_listed_again_once_the_hold_ran_out(context_factory):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, **LOST_CARD)
    await GpuLossHoldCheck().run(ctx)
    del redis.store[gpu_loss_hold._key(ctx)]  # Redis expired the hold

    result = await _cycle(context_factory, redis)

    assert result.passed and result.event.reason_code == Msg.OK.reason and not _hidden(result)


@pytest.mark.asyncio
async def test_a_healthy_rented_node_passes_and_writes_nothing(context_factory):
    redis = FakeRedis()

    results = [await _cycle(context_factory, redis) for _ in range(3)]

    assert all(
        r.passed and r.event.reason_code == Msg.OK.reason and not _hidden(r) for r in results
    )
    assert redis.store == {}


@pytest.mark.asyncio
async def test_weak_evidence_hides_the_node_only_on_its_second_cycle_in_a_row(context_factory):
    redis = FakeRedis()

    first = await _cycle(context_factory, redis, **DUPLICATED)
    assert first.passed and first.event.reason_code == Msg.PENDING.reason and not _hidden(first)
    assert redis.ttl[next(iter(redis.store))] == gpu_loss_hold.GPU_LOSS_CONFIRM_SECONDS

    second = await _cycle(context_factory, redis, **DUPLICATED)
    assert not second.passed and _hidden(second)


@pytest.mark.asyncio
async def test_weak_evidence_the_next_cycle_does_not_confirm_is_forgotten(context_factory):
    redis = FakeRedis()

    await _cycle(context_factory, redis, **DUPLICATED)
    clean = await _cycle(context_factory, redis)
    again = await _cycle(context_factory, redis, **DUPLICATED)

    assert clean.passed and not _hidden(clean)
    assert again.passed and again.event.reason_code == Msg.PENDING.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pods",
    [
        pytest.param((2, 2), id="both_shares_rented"),
        pytest.param((2,), id="lost_card_outside_the_rental"),
    ],
)
async def test_a_split_node_that_lost_a_card_is_hidden_for_new_rentals(context_factory, pods):
    redis = FakeRedis()

    # the driver no longer counts the fourth card, so the anchor (and the rental, when fully rented) sees it
    lost = await _cycle(context_factory, redis, listed=UUIDS[:3], count=3, pods=pods)
    clean = await _cycle(context_factory, redis, pods=pods)

    assert _hidden(lost) and lost.event.what_we_saw["missing_uuids"] == [UUIDS[3]]
    assert _hidden(clean) and not _fatal(clean)


@pytest.mark.asyncio
async def test_another_miners_rental_on_the_same_uuid_does_not_shield_or_hold_this_node(
    context_factory,
):
    redis = FakeRedis()
    await _cycle(context_factory, redis, **LOST_CARD)

    # the rental belongs to another hotkey, so this miner's node counts as idle and fails
    result = await _cycle(context_factory, redis, owner="another-hotkey")

    assert _hidden(result) and _fatal(result)
    assert all(key.startswith("gpu_loss_hold:miner-hotkey:") for key in redis.store)


@pytest.mark.asyncio
async def test_a_dry_run_reads_the_hold_and_writes_none(context_factory):
    redis = FakeRedis()
    with patch.object(gpu_loss_hold.settings, "DRY_RUN", True):
        result = await _cycle(context_factory, redis, **LOST_CARD)

    assert _hidden(result) and redis.store == {}


@pytest.mark.asyncio
async def test_redis_down_is_no_hold(context_factory):
    result = await _cycle(context_factory, FakeRedis(failing=True))

    assert result.passed and not _hidden(result)


@pytest.mark.asyncio
async def test_a_scrape_that_lists_no_gpu_starts_the_hold(context_factory):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, listed=[])

    events = await machine_spec_scrape._report_rented_gpu_loss(ctx, {"gpu": {"count": 4, "details": []}})

    assert json.loads(redis.store[gpu_loss_hold._key(ctx)])["held"] is True
    assert [e.reason_code for e in events] == [AvailabilityErrorCode.GPU_LOST]


class _ListSink:
    def __init__(self):
        self.events = []

    async def emit(self, event: ValidationEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_a_driver_crash_scrape_hides_the_node_through_the_run_it_ends(context_factory):
    # node 2288a966, 26 Oct 2026: NVML raised on every scrape, the run ended at the scrape, and the GPU_LOST
    # entry the hold built never reached the backend, so the node stayed listed
    ctx = _ctx(context_factory, FakeRedis(), listed=[], count=0, pods=())
    driver_report = json.dumps(
        {"error": "no_gpu_details", "data": {"data_gpu": {"gpu_count": 0, "gpu_details": []}, "gpu_scrape_error": "NVMLError(3)"}}
    )
    now = datetime.now(UTC)
    scrape_run = SSHCommandResult(
        command="scrape", command_id="c1", exit_code=1, stdout=driver_report, stderr="", duration_ms=1,
        started_at=now, finished_at=now, success=False,
    )

    class _ScrapeCheckStub:
        check_id = MachineSpecScrapeCheck.check_id
        fatal = True

        async def run(self, run_ctx: Context) -> CheckResult:
            return await MachineSpecScrapeCheck()._check_result_from_scrape_run(run_ctx, scrape_run, delivery="stdin")

    ok, events, _ = await Pipeline([_ScrapeCheckStub()], _ListSink()).run(ctx)

    assert not ok
    assert [e.reason_code for e in events] == [AvailabilityErrorCode.GPU_LOST, "SCRAPE_FAILED_DRIVER"]
    assert [e.reason_code for e in availability_errors(events)] == [AvailabilityErrorCode.GPU_LOST]


def test_gpu_lost_entries_do_not_count_toward_our_own_outage():
    lost = [{"reason_code": "GPU_LOST", "reach_source": "validator", "reach_target": "gpu"}]
    results = [JobResult.model_construct(availability_errors=list(lost)) for _ in range(6)]

    assert silence_availability_errors_on_our_own_outage(results) == 0
    assert all(result.availability_errors == lost for result in results)


def test_our_own_outage_keeps_gpu_lost_entries():
    ssh = {
        "reason_code": "EXECUTOR_SSH_UNREACHABLE",
        "reach_source": "validator",
        "reach_target": "ssh",
    }
    lost = {"reason_code": "GPU_LOST", "reach_source": "validator", "reach_target": "gpu"}
    results = [JobResult.model_construct(availability_errors=[ssh]) for _ in range(9)]
    results.append(JobResult.model_construct(availability_errors=[ssh, lost]))

    assert silence_availability_errors_on_our_own_outage(results) == 10
    assert [r.availability_errors for r in results] == [None] * 9 + [[lost]]


@pytest.mark.parametrize(
    "build",
    [
        PipelineFactory.build_checks,
        PipelineFactory.build_dry_run_checks,
        PipelineFactory.build_fast_path_checks,
    ],
)
def test_the_check_runs_after_the_scrape_and_before_the_gpu_checks_and_the_rented_short_circuit(
    build,
):
    kinds = [type(check) for check in build()]

    at = kinds.index(GpuLossHoldCheck)
    assert kinds.index(MachineSpecScrapeCheck) < kinds.index(RentedGpuDropCheck) < at
    assert at < kinds.index(GpuModelValidCheck) < kinds.index(TenantEnforcementCheck)


@pytest.mark.asyncio
async def test_a_card_missing_while_the_renter_resets_it_holds_the_cycle_without_a_hold(context_factory):
    """DAH-4658: the cards leave nvidia-smi for ~4-40 s during a renter's reset; that scrape is not a GPU loss."""
    from services.redis_service import RedisService

    class PendingRedis(FakeRedis):
        get_pending_pods = RedisService.get_pending_pods
        add_pending_pod = RedisService.add_pending_pod
        gpu_reset_marks = RedisService.gpu_reset_marks

    redis = PendingRedis()
    ctx = _ctx(context_factory, redis, listed=UUIDS[:2], count=2, pods=(2, 2))
    await redis.add_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, "pod2", kind="gpu_reset", gpu_uuids=UUIDS[2:])

    result = await GpuLossHoldCheck().run(ctx)

    # held as the zero-GPU scrape is, so the cycle carries the last score instead of GPU_MISSING zeroing it
    assert not result.passed and result.event.reason_code == Msg.MISSING_GPUS_UNDER_RESET.reason == "SCRAPE_GPU_RESET_IN_PROGRESS"
    assert await redis.get(gpu_loss_hold._key(ctx)) is None

    # a card of the other pod missing during the reset is still a loss
    other = _ctx(context_factory, redis, listed=UUIDS[1:], count=3, pods=(2, 2))
    assert (await GpuLossHoldCheck().run(other)).event.reason_code != Msg.MISSING_GPUS_UNDER_RESET.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("pods", [(), (4,)])
async def test_a_node_whose_whole_gpu_set_was_replaced_is_held(context_factory, pods):
    """Replaced cards are held like lost ones (a VM host reboot, 27 Oct fcd3835d); on an idle node the cycle is not
    fatal here, so GpuFingerprintCheck still re-anchors them, and the node stays hidden for the hold window."""
    redis = FakeRedis()
    replaced = [f"GPU-b0b{i}-0000-0000-0000-00000000000{i}" for i in range(4)]
    ctx = _ctx(context_factory, redis, listed=replaced, pods=pods)

    result = await GpuLossHoldCheck().run(ctx)

    assert _hidden(result) and not result.passed
    assert redis.ttl[gpu_loss_hold._key(ctx)] == gpu_loss_hold.settings.GPU_LOSS_HOLD_SECONDS
    if not pods:
        assert not _fatal(result)
        assert result.event.what_we_saw["faults"] == [gpu_loss_hold.FAULT_SET_REPLACED]
        assert result.event.event == Msg.SET_REPLACED.event
        # next cycle: the new cards are the anchor and read clean, the node is still hidden (fatal while idle)
        reanchored = _ctx(context_factory, redis, listed=replaced, pods=(), anchor=replaced)
        clean = await GpuLossHoldCheck().run(reanchored)
        assert _hidden(clean) and _fatal(clean)
        # a second whole swap inside the hold is judged against the new anchor and starts the hold again
        third = [f"GPU-c0c{i}-0000-0000-0000-00000000000{i}" for i in range(4)]
        again = await GpuLossHoldCheck().run(_ctx(context_factory, redis, listed=third, pods=(), anchor=replaced))
        assert _hidden(again) and again.event.what_we_saw["faults"] == [gpu_loss_hold.FAULT_SET_REPLACED]
