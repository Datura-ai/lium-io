"""GPU_LOST: a node whose GPU dropped out is hidden and takes no new rentals until its cards stay back (DAH-4457)."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.availability import (
    AvailabilityErrorCode,
    availability_errors,
    silence_availability_errors_on_our_own_outage,
)
from neurons.validators.src.services.task.checks import (
    GpuFingerprintCheck,
    GpuLossHoldCheck,
    GpuModelValidCheck,
    MachineSpecScrapeCheck,
    PostRentalGpuRecheckCheck,
    RentedGpuDropCheck,
    TenantEnforcementCheck,
    gpu_loss_hold,
    machine_spec_scrape,
    post_rental_gpu_recheck,
    rented_gpu_drop,
)
from neurons.validators.src.services.task.messages import GpuLossHoldMessages as Msg
from neurons.validators.src.services.task.messages import PostRentalGpuRecheckMessages as RecheckMsg
from neurons.validators.src.services.task.models import JobResult
from neurons.validators.src.services.task.pipeline import CheckResult, Context, Pipeline, ValidationEvent
from neurons.validators.src.services.task.runner import SSHCommandResult
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedGpuDropResponse,
    RentedPod,
)

from tests.test_gpu_fault_probe_check import FakeRunner, probe_stdout
from tests.test_post_rental_gpu_recheck import FakeValidation, _all_cards
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
RECHECK_KEY = post_rental_gpu_recheck.marker_key("miner-hotkey", default_executor().uuid)
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
    pod_gpus=None,
):
    """`pod_gpus`: the GPUs the host's docker record gives each pod's container (default: the listed cards)."""
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
        ssh=_DockerHost(listed if pod_gpus is None else pod_gpus, [f"container_pod{n}" for n in range(1, len(pods) + 1)]),
    )


class _DockerHost:
    """The host's `docker inspect` of the pods' containers: each given `gpus` (None = cannot be read)."""

    def __init__(self, gpus, names):
        self.gpus, self.names, self.calls = gpus, names, []

    async def run(self, cmd, timeout=None):
        self.calls.append(cmd)
        if self.gpus is None:
            raise OSError("ssh lost")
        request = [{"Driver": "nvidia", "DeviceIDs": list(self.gpus), "Capabilities": [["gpu"]]}]
        return SimpleNamespace(stdout="".join(f'"/{n}" {json.dumps(request)}\n' for n in self.names))


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
    assert redis.ttl[gpu_loss_hold._key(ctx)] == gpu_loss_hold.GPU_LOSS_MARK_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_rented_node_whose_card_reads_clean_again_stays_hidden_but_keeps_its_rental(
    context_factory,
):
    redis = FakeRedis()
    await _cycle(context_factory, redis, **LOST_CARD)

    result = await _cycle(context_factory, redis)

    # hidden, and not fatal under a live rental: the rented check still runs and the renter is not touched
    assert not result.passed and not _fatal(result) and _hidden(result)
    assert "hold_until" not in result.event.what_we_saw
    assert result.event.event == Msg.LOST.event


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
    ctx = _ctx(context_factory, redis)
    old = {"held": True, "last_seen": "2026-10-17T00:00:00+00:00"}
    redis.store[gpu_loss_hold._key(ctx)] = json.dumps(old)

    result = await GpuLossHoldCheck().run(ctx)

    assert not result.passed and _hidden(result) and result.event.what_we_saw["since"]


@pytest.mark.asyncio
async def test_after_the_rental_ends_the_first_clean_read_ends_the_hold_and_owes_the_load_test(context_factory):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, **LOST_CARD)
    await GpuLossHoldCheck().run(ctx)

    result = await _cycle(context_factory, redis, pods=())

    assert result.passed and result.event.reason_code == Msg.CLEAN_AGAIN.reason and not _hidden(result)
    assert gpu_loss_hold._key(ctx) not in redis.store
    assert json.loads(redis.store[RECHECK_KEY])["source"] == "gpu_loss_hold"


@pytest.mark.asyncio
async def test_a_dropout_keeps_the_node_hidden_while_cards_are_missing_with_no_timer(context_factory):
    """Regression: a node hidden for a fixed window, or listed again because a window ran out while a card was
    still missing."""
    redis = FakeRedis()

    cycles = [await _cycle(context_factory, redis, pods=(), **LOST_CARD) for _ in range(5)]

    assert all(_hidden(c) and not c.passed for c in cycles)
    assert all("hold_until" not in c.event.what_we_saw for c in cycles)
    assert not hasattr(gpu_loss_hold.settings, "GPU_LOSS_HOLD_SECONDS")
    assert RECHECK_KEY not in redis.store  # the load test is owed once the cards are back


def _hold_then_load_test_pipeline(context_factory, redis, *, all_cards, probe):
    ctx = _ctx(context_factory, redis, pods=())
    validation = FakeValidation(all_cards)
    ctx = ctx.model_copy(
        update={"services": build_services(redis=redis, validation=validation), "runner": FakeRunner(probe)}
    )
    return ctx, validation, Pipeline([GpuLossHoldCheck(), PostRentalGpuRecheckCheck()], _ListSink())


@pytest.mark.asyncio
async def test_the_first_clean_read_load_tests_every_card_the_same_cycle_and_a_pass_lists_the_node(context_factory):
    redis = FakeRedis()
    await _cycle(context_factory, redis, pods=(), **LOST_CARD)
    ctx, validation, pipeline = _hold_then_load_test_pipeline(
        context_factory, redis, all_cards=_all_cards("", "", "", ""), probe=probe_stdout("ok")
    )

    ok, events, _ = await pipeline.run(ctx)

    assert ok and validation.calls == 1
    assert [e.reason_code for e in events] == [Msg.CLEAN_AGAIN.reason, "POST_RENTAL_GPU_RECHECK_PASSED"]
    assert availability_errors(events) == []
    assert gpu_loss_hold._key(ctx) not in redis.store and RECHECK_KEY not in redis.store


@pytest.mark.asyncio
async def test_a_card_failing_the_load_test_keeps_the_node_out_until_a_clean_read_passes_it(context_factory):
    redis = FakeRedis()
    await _cycle(context_factory, redis, pods=(), **LOST_CARD)
    ctx, _, pipeline = _hold_then_load_test_pipeline(
        context_factory, redis, all_cards=_all_cards("", "", "exit_status=1", ""), probe=probe_stdout("ok")
    )

    ok, events, _ = await pipeline.run(ctx)

    assert not ok and events[-1].reason_code == "POST_RENTAL_GPU_RECHECK_FAILED"
    assert [e.reason_code for e in availability_errors(events)] == [AvailabilityErrorCode.GPU_LOST]
    assert RECHECK_KEY in redis.store  # still owed: the next clean read tests every card again
    # the card drops again: hidden again
    again = await _cycle(context_factory, redis, pods=(), **LOST_CARD)
    assert _hidden(again) and not again.passed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("filler", id="skipped_under_a_filler"),
        pytest.param("no_verdict", id="no_result"),
        pytest.param("probe_error", id="probe_errors"),
    ],
)
async def test_a_node_back_from_a_dropout_is_not_listed_until_the_load_test_has_passed(context_factory, case):
    """Regression: the node listed on the cycle that skipped the every-card test (a filler) or got no result from it."""
    redis = FakeRedis()
    await _cycle(context_factory, redis, pods=(), **LOST_CARD)
    all_cards = None if case == "no_verdict" else _all_cards("", "", "", "")
    probe = "not json" if case == "probe_error" else probe_stdout("ok")
    ctx, _, pipeline = _hold_then_load_test_pipeline(context_factory, redis, all_cards=all_cards, probe=probe)

    filler = {"id": "filler"} if case == "filler" else None
    with patch.object(post_rental_gpu_recheck, "get_filler_only_container", return_value=filler):
        ok, events, _ = await pipeline.run(ctx)

    assert not ok
    assert events[-1].reason_code == RecheckMsg.WAITING_AFTER_GPU_LOSS.reason
    # an availability entry: the backend hides the node even while its row is still active
    assert [e.reason_code for e in availability_errors(events)] == [AvailabilityErrorCode.GPU_LOST]
    assert events[-1].event == "GPUs back; waiting for the load test on every card"
    assert json.loads(redis.store[RECHECK_KEY])["source"] == "gpu_loss_hold"  # still owed

    # the next check runs it and passes: listed, no longer owed
    ctx, validation, pipeline = _hold_then_load_test_pipeline(
        context_factory, redis, all_cards=_all_cards("", "", "", ""), probe=probe_stdout("ok")
    )
    ok, events, _ = await pipeline.run(ctx)
    assert ok and validation.calls == 1 and RECHECK_KEY not in redis.store
    assert events[-1].reason_code == RecheckMsg.PASSED.reason


@pytest.mark.asyncio
async def test_a_post_rental_recheck_still_proceeds_on_a_skip(context_factory):
    """Only the dropout marker blocks on a skip; an ordinary post-rental recheck keeps its behaviour."""
    redis = FakeRedis()
    redis.store[RECHECK_KEY] = json.dumps({"source": "gpu.validate.memory_health"})
    ctx, _, _ = _hold_then_load_test_pipeline(context_factory, redis, all_cards=None, probe=probe_stdout("ok"))

    with patch.object(post_rental_gpu_recheck, "get_filler_only_container", return_value={"id": "filler"}):
        result = await PostRentalGpuRecheckCheck().run(ctx)

    assert result.passed and result.event.reason_code == RecheckMsg.FILLER_SKIPPED.reason
    assert availability_errors([result.event]) == []


def test_no_fixed_hold_duration_in_the_provider_facing_texts():
    texts = [
        getattr(t, field) or ""
        for t in vars(Msg).values()
        if hasattr(t, "remediation")
        for field in ("event", "impact", "remediation")
    ]
    assert texts and not any(word in text for text in texts for word in ("hold window", "hold_until", "2 hours"))


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

    # the rental belongs to another hotkey, so this miner's node counts as idle: its clean read owes the load test
    result = await _cycle(context_factory, redis, owner="another-hotkey")

    assert result.event.reason_code == Msg.CLEAN_AGAIN.reason
    assert not any(key.startswith("gpu_loss_hold:") for key in redis.store) and RECHECK_KEY in redis.store


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


OTHER_CARDS = [f"GPU-b0b{i}-0000-0000-0000-00000000000{i}" for i in range(4)]


@pytest.mark.asyncio
@pytest.mark.parametrize("pods", [pytest.param((), id="idle"), pytest.param((4,), id="rented")])
@pytest.mark.parametrize("listed", [pytest.param(OTHER_CARDS, id="all"), pytest.param(UUIDS[:3] + OTHER_CARDS[:1], id="one")])
async def test_a_full_set_of_other_cards_is_no_hold_and_verifies_the_same_cycle(context_factory, pods, listed):
    """Regression: a node back with the same number of other GPUs (a VM host reboot) was hidden for the hold window,
    or held under a rental, instead of being re-anchored and verified at once."""
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, listed=listed, pods=pods)

    result = await GpuLossHoldCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.OK.reason and not _hidden(result)
    assert gpu_loss_hold._key(ctx) not in redis.store
    fingerprint = await GpuFingerprintCheck().run(ctx)
    assert fingerprint.passed and fingerprint.updates == {"gpu_reanchor": True}


@pytest.mark.asyncio
async def test_a_hold_left_by_the_retired_set_replaced_rule_is_dropped(context_factory):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, pods=())
    old = {"held": True, "since": "2026-10-29T00:00:00+00:00", "last_seen": "2026-10-29T00:00:00+00:00", "replaced": True}
    redis.store[gpu_loss_hold._key(ctx)] = json.dumps(old)

    result = await GpuLossHoldCheck().run(ctx)

    assert result.passed and not _hidden(result) and gpu_loss_hold._key(ctx) not in redis.store


@pytest.mark.asyncio
async def test_fewer_cards_after_a_set_change_is_still_a_dropout_and_held(context_factory):
    redis = FakeRedis()

    result = await _cycle(context_factory, redis, listed=OTHER_CARDS[:3], count=3, pods=())

    assert not result.passed and _hidden(result)
    assert redis.ttl[gpu_loss_hold._key(_ctx(context_factory, redis, pods=()))] == gpu_loss_hold.GPU_LOSS_MARK_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_set_change_under_a_rental_posts_the_renters_incident_and_a_pod_on_the_old_cards_stays_unavailable(
    context_factory,
):
    """The renter hears about the swap through the rented GPU-drop report; the pod still made for the old cards has
    no GPU, so the cycle fails (score 0, node hidden) and nothing re-anchors until the pod is recreated."""
    rented_gpu_drop._LOCAL_MARKS.clear()
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, listed=OTHER_CARDS, pods=(4,), pod_gpus=UUIDS)
    ctx.services.backend.report_rented_gpu_drop.return_value = RentedGpuDropResponse(
        recorded=True, delivery="notified"
    )

    with patch.object(rented_gpu_drop.settings, "DRY_RUN", False):
        incident = await RentedGpuDropCheck().run(ctx)
    hold = await GpuLossHoldCheck().run(ctx)

    call = ctx.services.backend.report_rented_gpu_drop.await_args
    assert call.args[0] == _pod(1, 4).pod_id and call.kwargs["state"] == "fault"
    assert call.kwargs["missing_uuids"] == sorted(UUIDS)
    assert incident.passed and not _hidden(incident)
    assert not hold.passed and _fatal(hold) and _hidden(hold)
    assert hold.event.event == Msg.POD_GPUS_GONE.event
    assert hold.event.what_we_saw["missing_uuids"] == sorted(UUIDS)
    assert "docker inspect" in ctx.ssh.calls[0] and "exec" not in ctx.ssh.calls[0]
    rented_gpu_drop._LOCAL_MARKS.clear()


HELD_BY_A_DROPOUT = {
    "held": True, "since": "2026-10-29T10:28:46+00:00", "last_seen": "2026-10-29T10:28:46+00:00", "replaced": False,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("pods", [pytest.param((), id="idle"), pytest.param((4,), id="rented-pod-recreated")])
async def test_a_dropout_hold_ends_when_the_node_is_back_with_a_full_set_of_other_cards(context_factory, pods):
    """A hold an earlier dropout left must not keep a node back with a full new set hidden for the whole rental."""
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, listed=OTHER_CARDS, pods=pods)
    redis.store[gpu_loss_hold._key(ctx)] = json.dumps(HELD_BY_A_DROPOUT)

    result = await GpuLossHoldCheck().run(ctx)

    assert result.passed and not _hidden(result) and gpu_loss_hold._key(ctx) not in redis.store
    fingerprint = await GpuFingerprintCheck().run(ctx)
    assert fingerprint.passed and fingerprint.updates == {"gpu_reanchor": True}


PARTLY_SWAPPED = UUIDS[:3] + OTHER_CARDS[:1]


@pytest.mark.asyncio
async def test_a_dropout_hold_on_a_node_back_with_a_partial_swap_still_owes_the_load_test(context_factory):
    """Regression (review of #1956): a partial swap at equal count cleared the hold without the load-test marker."""
    redis = FakeRedis()
    rented = _ctx(context_factory, redis, listed=PARTLY_SWAPPED, pods=(4,))
    redis.store[gpu_loss_hold._key(rented)] = json.dumps(HELD_BY_A_DROPOUT)

    # under a rental the test cannot run: the node stays hidden (not fatal) and the hold keeps the obligation
    result = await GpuLossHoldCheck().run(rented)
    assert not result.passed and not _fatal(result) and _hidden(result)
    assert gpu_loss_hold._key(rented) in redis.store and RECHECK_KEY not in redis.store

    # the first idle cycle (re-anchored or not) ends the hold and owes the load test
    for anchor in (UUIDS, PARTLY_SWAPPED):
        redis.store[gpu_loss_hold._key(rented)] = json.dumps(HELD_BY_A_DROPOUT)
        redis.store.pop(RECHECK_KEY, None)
        result = await _cycle(context_factory, redis, listed=PARTLY_SWAPPED, anchor=anchor, pods=())
        assert result.passed and result.event.reason_code == Msg.CLEAN_AGAIN.reason
        assert gpu_loss_hold._key(rented) not in redis.store
        assert json.loads(redis.store[RECHECK_KEY])["source"] == "gpu_loss_hold"


@pytest.mark.asyncio
async def test_a_pod_on_the_old_cards_keeps_the_node_back_and_drops_the_dropout_hold(context_factory):
    redis = FakeRedis()
    ctx = _ctx(context_factory, redis, listed=OTHER_CARDS, pods=(4,), pod_gpus=UUIDS)
    redis.store[gpu_loss_hold._key(ctx)] = json.dumps(HELD_BY_A_DROPOUT)

    result = await GpuLossHoldCheck().run(ctx)

    assert not result.passed and _fatal(result) and _hidden(result)
    assert gpu_loss_hold._key(ctx) not in redis.store
    # the pod recreated on the new cards: the next cycle passes and re-anchors
    cured = _ctx(context_factory, redis, listed=OTHER_CARDS, pods=(4,))
    assert (await GpuLossHoldCheck().run(cured)).passed
    assert (await GpuFingerprintCheck().run(cured)).updates == {"gpu_reanchor": True}


@pytest.mark.asyncio
async def test_a_pod_that_cannot_be_read_after_a_set_change_is_not_taken_as_served(context_factory):
    ctx = _ctx(context_factory, FakeRedis(), listed=OTHER_CARDS, pods=(4,))
    ctx.ssh.gpus = None  # the SSH read fails

    unread = await GpuLossHoldCheck().run(ctx)

    assert not unread.passed and _hidden(unread) and unread.event.what_we_saw["pods_unread"] == ["container_pod1"]


@pytest.mark.asyncio
async def test_a_pod_given_every_gpu_is_served_by_the_new_cards(context_factory):
    ctx = _ctx(context_factory, FakeRedis(), listed=OTHER_CARDS, pods=(4,))
    all_gpus = [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]

    async def run(cmd, timeout=None):
        return SimpleNamespace(stdout=f'"/container_pod1" {json.dumps(all_gpus)}\n')

    ctx.ssh.run = run

    assert (await GpuLossHoldCheck().run(ctx)).passed
