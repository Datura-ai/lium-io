"""RENTED_GPU_DROP: a rented node that lost a GPU is reported the cycle it is seen, once per incident."""

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.checks import (
    GpuModelValidCheck,
    MachineSpecScrapeCheck,
    RentedGpuDropCheck,
    TenantEnforcementCheck,
    rented_gpu_drop,
)
from neurons.validators.src.services.task.checks.rented_gpu_drop import (
    FAULT_ANCHORED_MISSING,
    FAULT_BELOW_RENTED,
    FAULT_DETAILS_SHORT,
    FAULT_NVML_ERROR,
    judge_rented_gpus,
    nvml_error_code,
)
from neurons.validators.src.services.task.messages import MachineSpecMessages
from neurons.validators.src.services.task.messages import RentedGpuDropMessages as Msg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedGpuDropResponse,
    RentedPod,
)

from tests.helpers import (
    FakeRedis,
    build_context_config,
    build_services,
    build_state,
    default_executor,
)
from tests.test_machine_spec_scrape_check import DummySSHCommandRunner, make_command_result

MODEL = "NVIDIA GeForce RTX 5090"
UUIDS = [f"GPU-a0a{i}-0000-0000-0000-00000000000{i}" for i in range(8)]
POD_ID = "00000000-0000-4000-8000-000000000001"
NOTIFIED = RentedGpuDropResponse(recorded=True, delivery="notified")
DUPLICATED = UUIDS[:7] + [UUIDS[0]]


@pytest.fixture(autouse=True)
def _check_on():
    rented_gpu_drop._LOCAL_MARKS.clear()
    with patch.object(rented_gpu_drop.settings, "RENTED_GPU_DROP_CHECK_ENABLED", True):
        yield
    rented_gpu_drop._LOCAL_MARKS.clear()


def _details(uuids: list[str]) -> list[dict]:
    return [{"name": MODEL, "uuid": uuid, "capacity": 32607} for uuid in uuids]


def _rented(
    executor_uuid: str,
    *,
    status: str | None = "RUNNING",
    gpu_count: int | None = 8,
    pods: list[RentedPod] | None = None,
) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8001",
                pods=pods
                or [
                    RentedPod(
                        pod_id=POD_ID,
                        container_name="container_pod1",
                        gpu_count=gpu_count,
                        status=status,
                    )
                ],
            )
        }
    )


def _ctx(
    context_factory,
    services,
    *,
    listed: list[str],
    count: int = 8,
    scrape_error: str | None = None,
    rented=True,
    anchor: list[str] = UUIDS,
    **rented_kwargs,
):
    executor = default_executor()
    specs: dict = {"gpu": {"count": count, "details": _details(listed)}}
    if scrape_error is not None:
        specs["gpu_scrape_error"] = scrape_error
    state = build_state(
        specs=specs,
        gpu_count=count,
        gpu_details=_details(listed),
        gpu_uuids=",".join(listed),
        rented_data=_rented(executor.uuid, **rented_kwargs)
        if rented
        else RentedExecutorsResponse(executors={}),
    )
    return context_factory(
        services=services,
        state=state,
        executor=executor,
        verified={"spec": f"{MODEL}:8", "uuids": ",".join(anchor)},
    )


def _services(answer=NOTIFIED, redis=None):
    services = build_services(redis=redis or FakeRedis())
    services.backend.report_rented_gpu_drop.return_value = answer
    return services


async def _run(context_factory, services, **kwargs):
    return await RentedGpuDropCheck().run(_ctx(context_factory, services, **kwargs))


def _states(services) -> list[str]:
    return [
        call.kwargs["state"] for call in services.backend.report_rented_gpu_drop.await_args_list
    ]


# --- the pure rule ------------------------------------------------------------------------------------------------


def test_nvml_error_code_reads_the_scrape_repr():
    assert nvml_error_code("NVMLError(999)") == 999
    assert nvml_error_code("NVMLError_GpuIsLost(15)") == 15
    assert nvml_error_code("OSError('libnvidia-ml.so.1: cannot open')") is None
    assert nvml_error_code(None) is None


BR, DS, AM, NV = FAULT_BELOW_RENTED, FAULT_DETAILS_SHORT, FAULT_ANCHORED_MISSING, FAULT_NVML_ERROR


@pytest.mark.parametrize(
    ("rented", "anchor", "nvml", "listed", "error", "faults", "missing", "confirm_first"),
    [
        (8, UUIDS, 8, UUIDS[:5], "NVMLError(999)", [BR, DS, AM, NV], UUIDS[5:], False),
        (8, UUIDS, 8, UUIDS, "NVMLError(15)", [NV], [], False),
        (8, UUIDS, 8, UUIDS, "NVMLError(3)", None, None, None),
        (8, UUIDS, 8, UUIDS[:5], "AttributeError('x')", [BR, DS, AM], UUIDS[5:], True),
        (4, UUIDS, 7, UUIDS[:7], None, [AM], [UUIDS[7]], False),
        (8, UUIDS[:7], 8, DUPLICATED, None, None, None, None),
        (8, UUIDS, 8, DUPLICATED, None, [AM], [UUIDS[7]], True),
        (8, UUIDS, 8, UUIDS[:6] + [UUIDS[0]], None, [BR, DS, AM], UUIDS[6:], False),
        (8, UUIDS, 8, DUPLICATED, "NVMLError(15)", [AM, NV], [UUIDS[7]], False),
        (8, UUIDS, 8, UUIDS[:3], "NVMLError_Timeout(10)", [BR, DS, AM, NV], UUIDS[3:], True),
        (8, UUIDS, 7, UUIDS[:7], "NVMLError_NotSupported(3)", [BR, AM, NV], [UUIDS[7]], False),
        (8, UUIDS, 7, UUIDS[:5], "NVMLError_Timeout(10)", [BR, DS, AM, NV], UUIDS[5:], False),
        (8, UUIDS, 0, [], "NVMLError_DriverNotLoaded(9)", [BR, AM, NV], UUIDS, False),
        (None, [], 0, [], "NVMLError_LibraryNotFound(12)", [BR, NV], [], False),
        (None, [], 0, [], None, [BR], [], False),
        (None, [], 2, UUIDS[:2], None, None, None, None),
    ],
    ids=[
        "count_drop_names_every_fault",
        "loss_code_with_every_card_listed",
        "non_loss_code_with_every_card_listed_is_healthy",
        "non_nvml_scrape_error_is_not_labelled_nvml",
        "split_node_card_outside_the_rental_seen_by_the_anchor",
        "card_listed_twice_counted_by_rows_is_healthy",
        "card_listed_twice_against_a_full_anchor_is_held",
        "card_listed_twice_with_rows_short_is_not_held",
        "card_listed_twice_with_a_loss_code_is_not_held",
        "scrape_cut_short_by_a_timeout_is_held",
        "driver_count_below_the_rental_is_not_held",
        "cut_short_and_driver_short_is_not_held",
        "zero_cards_with_a_rental_count_and_anchor",
        "zero_cards_with_no_rental_count_or_anchor",
        "zero_cards_with_no_scrape_error",
        "no_rental_count_or_anchor_healthy",
    ],
)
def test_judge_rented_gpus(rented, anchor, nvml, listed, error, faults, missing, confirm_first):
    drop = judge_rented_gpus(
        rented_gpu_count=rented,
        anchor_uuids=anchor,
        nvml_count=nvml,
        listed_uuids=listed,
        listed_count=len(listed),
        scrape_error=error,
    )

    if faults is None:
        assert drop is None
        return
    assert drop is not None and drop.visible == len(listed)
    assert (drop.faults, drop.missing_uuids, drop.confirm_first) == (faults, missing, confirm_first)


# --- the check ----------------------------------------------------------------------------------------------------


GLITCHES = [
    pytest.param({"listed": DUPLICATED}, id="card_listed_twice"),
    pytest.param(
        {"listed": UUIDS[:3], "scrape_error": "NVMLError_Timeout(10)"}, id="scrape_timeout"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("glitch", GLITCHES)
async def test_a_possible_glitch_is_held_then_posted_if_it_persists(context_factory, glitch):
    services = _services()

    first = await _run(context_factory, services, **glitch)
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert first.event.reason_code == Msg.DROP.reason
    assert first.event.what_we_saw["pods"][0]["held"] is True

    await _run(context_factory, services, **glitch)
    services.backend.report_rented_gpu_drop.assert_awaited_once()
    assert services.backend.report_rented_gpu_drop.await_args.kwargs["consecutive_cycles"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("glitch", GLITCHES)
async def test_a_held_glitch_that_clears_is_never_posted_nor_called_recovered(
    context_factory, glitch
):
    services = _services()

    await _run(context_factory, services, **glitch)
    cleared = await _run(context_factory, services, listed=UUIDS)

    assert cleared.event.reason_code == Msg.OK.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.store == {}


@pytest.mark.asyncio
async def test_a_count_drop_on_a_rented_node_is_reported_on_the_first_cycle(context_factory):
    services = _services()

    result = await _run(context_factory, services, listed=UUIDS[:5], scrape_error="NVMLError(999)")

    assert result.passed is True
    assert result.event.reason_code == Msg.DROP.reason
    assert result.event.what_we_saw["expected_gpu_count"] == 8
    assert result.event.what_we_saw["visible_gpu_count"] == 5
    assert result.event.what_we_saw["missing_uuids"] == UUIDS[5:]
    assert result.event.what_we_saw["nvml_error_code"] == 999
    assert "updates" not in result.model_dump(exclude_defaults=True)
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.args == (POD_ID,)
    assert call.kwargs["state"] == "fault"
    assert call.kwargs["expected_gpu_count"] == 8 and call.kwargs["visible_gpu_count"] == 5
    assert call.kwargs["nvml_error_code"] == 999
    assert call.kwargs["executor_id"] == default_executor().uuid
    assert call.kwargs["pod_gpu_count"] == 8
    assert call.kwargs["rented_gpu_count"] == 8 and call.kwargs["nvml_gpu_count"] == 8


@pytest.mark.asyncio
async def test_a_zero_card_scrape_is_reported_when_the_pod_has_no_gpu_count_and_no_anchor(
    context_factory,
):
    services = _services()

    result = await _run(
        context_factory,
        services,
        listed=[],
        count=0,
        scrape_error="NVMLError_LibraryNotFound(12)",
        gpu_count=None,
        anchor=[],
    )

    assert result.event.reason_code == Msg.DROP.reason
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["visible_gpu_count"] == 0 and call.kwargs["expected_gpu_count"] == 1


async def _scrape_listing_no_gpu(
    context_factory, services, *, count, scrape_error, anchor=UUIDS, **rented_kwargs
):
    data: dict = {"data_gpu": {"gpu_count": count, "gpu_details": []}}
    if scrape_error:
        data["gpu_scrape_error"] = scrape_error
    report = json.dumps({"error": "no_gpu_details", "data": data})
    executor = default_executor()
    ctx = context_factory(
        services=services,
        config=build_context_config(
            machine_scrape_filename="scrape.sh", machine_scrape_timeout=300, obfuscation_keys={}
        ),
        state=build_state(
            remote_dir="/remote/path", rented_data=_rented(executor.uuid, **rented_kwargs)
        ),
        runner=DummySSHCommandRunner(
            result=make_command_result(success=False, exit_code=1, stdout=report)
        ),
        executor=executor,
        verified={"uuids": ",".join(anchor)},
        encrypt_key="test-encrypt-key",
    )
    return await MachineSpecScrapeCheck().run(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "scrape_error", "anchor", "pod_gpus", "faults"),
    [
        (8, "NVMLError_GpuIsLost(15)", UUIDS, 8, [BR, DS, AM, NV]),
        (0, "NVMLError_DriverNotLoaded(9)", [], None, [BR, NV]),
    ],
    ids=["card_0_lost", "driver_down_no_rental_count_no_anchor"],
)
async def test_a_scrape_that_lists_no_gpu_fails_as_before_and_reports_the_rented_node(
    context_factory, count, scrape_error, anchor, pod_gpus, faults
):
    services = _services()

    result = await _scrape_listing_no_gpu(
        context_factory,
        services,
        count=count,
        scrape_error=scrape_error,
        anchor=anchor,
        gpu_count=pod_gpus,
    )

    assert result.passed is False
    assert result.event.reason_code == MachineSpecMessages.SCRAPE_FAILED_DRIVER.reason
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["state"] == "fault" and call.kwargs["faults"] == faults
    assert call.kwargs["visible_gpu_count"] == 0 and call.kwargs["nvml_gpu_count"] == count
    services.backend.report_rented_gpu_drop.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_scrape_that_lists_no_gpu_on_an_idle_node_posts_nothing(context_factory):
    services = _services()

    result = await _scrape_listing_no_gpu(
        context_factory, services, count=8, scrape_error="NVMLError(999)", status="STOPPED"
    )

    assert result.passed is False
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.calls == 0


@pytest.mark.asyncio
async def test_an_nvml_error_with_every_card_listed_is_reported(context_factory):
    services = _services()

    result = await _run(
        context_factory, services, listed=UUIDS, scrape_error="NVMLError_GpuIsLost(15)"
    )

    assert result.event.reason_code == Msg.DROP.reason
    assert result.event.what_we_saw["faults"] == [FAULT_NVML_ERROR]
    services.backend.report_rented_gpu_drop.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_node_that_is_not_rented_is_ignored(context_factory):
    services = _services()

    result = await _run(
        context_factory, services, listed=UUIDS[:5], scrape_error="NVMLError(999)", rented=False
    )

    assert result.event.reason_code == Msg.NOT_RENTED.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.calls == 0


@pytest.mark.asyncio
async def test_a_pod_that_is_not_running_is_ignored(context_factory):
    services = _services()

    result = await _run(context_factory, services, listed=UUIDS[:5], status="REBOOT_PENDING")

    assert result.event.reason_code == Msg.NOT_RENTED.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_same_incident_is_posted_once(context_factory):
    services = _services()

    for _ in range(3):
        result = await _run(
            context_factory, services, listed=UUIDS[:5], scrape_error="NVMLError(999)"
        )

    services.backend.report_rented_gpu_drop.assert_awaited_once()
    pod = result.event.what_we_saw["pods"][0]
    assert pod["consecutive_cycles"] == 3 and pod["reported"] is True and pod["posted"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_answer",
    [
        None,
        RentedGpuDropResponse(recorded=True, delivery="notify_failed"),
        RentedGpuDropResponse(recorded=False, delivery="disabled"),
        RentedGpuDropResponse(recorded=True, delivery="capped"),
    ],
    ids=["no_answer", "notify_failed", "backend_flag_off", "unknown_delivery"],
)
async def test_an_unacknowledged_report_is_posted_again_next_cycle(context_factory, first_answer):
    services = _services(answer=first_answer)
    await _run(context_factory, services, listed=UUIDS[:5])
    services.backend.report_rented_gpu_drop.return_value = NOTIFIED

    await _run(context_factory, services, listed=UUIDS[:5])
    await _run(context_factory, services, listed=UUIDS[:5])

    assert services.backend.report_rented_gpu_drop.await_count == 2
    first, second = services.backend.report_rented_gpu_drop.await_args_list
    assert first.kwargs["first_seen_at"] == second.kwargs["first_seen_at"]


@pytest.mark.asyncio
async def test_a_backend_error_does_not_fail_the_cycle_and_is_retried(context_factory):
    services = _services()
    services.backend.report_rented_gpu_drop.side_effect = [RuntimeError("boom"), NOTIFIED]

    first = await _run(context_factory, services, listed=UUIDS[:5])
    await _run(context_factory, services, listed=UUIDS[:5])

    assert first.passed is True and first.event.reason_code == Msg.DROP.reason
    assert services.backend.report_rented_gpu_drop.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault_answer", "states", "second_reason"),
    [
        (NOTIFIED, ["fault", "recovered"], Msg.RECOVERED.reason),
        (None, ["fault", "recovered"], Msg.RECOVERED.reason),
        (RentedGpuDropResponse(recorded=False, delivery="disabled"), ["fault"], Msg.OK.reason),
    ],
    ids=["notified", "fault_answer_lost", "never_recorded"],
)
async def test_recovery_is_posted_once_for_an_incident_the_backend_may_hold(
    context_factory, fault_answer, states, second_reason
):
    services = _services(answer=fault_answer)
    await _run(context_factory, services, listed=UUIDS[:5], scrape_error="NVMLError(999)")
    services.backend.report_rented_gpu_drop.return_value = NOTIFIED

    second = await _run(context_factory, services, listed=UUIDS)
    third = await _run(context_factory, services, listed=UUIDS)

    assert (second.event.reason_code, third.event.reason_code) == (second_reason, Msg.OK.reason)
    assert _states(services) == states
    assert services.redis.store == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recovery_answer",
    [
        None,
        RentedGpuDropResponse(recorded=True, delivery="notify_failed"),
        RentedGpuDropResponse(recorded=False, delivery="disabled"),
    ],
    ids=["no_answer", "notify_failed", "disabled"],
)
async def test_an_unacknowledged_recovery_is_posted_again(context_factory, recovery_answer):
    services = _services()
    await _run(context_factory, services, listed=UUIDS[:5])
    services.backend.report_rented_gpu_drop.return_value = recovery_answer
    await _run(context_factory, services, listed=UUIDS)
    assert services.redis.store != {}
    services.backend.report_rented_gpu_drop.return_value = NOTIFIED
    await _run(context_factory, services, listed=UUIDS)
    await _run(context_factory, services, listed=UUIDS)

    assert _states(services) == ["fault", "recovered", "recovered"]
    assert services.redis.store == {}


@pytest.mark.asyncio
async def test_a_fault_after_a_posted_recovery_starts_a_new_incident(context_factory):
    services = _services()
    clock = _Clock()

    with patch.object(rented_gpu_drop, "datetime", clock):
        await _run(context_factory, services, listed=UUIDS[:5])
        services.backend.report_rented_gpu_drop.return_value = RentedGpuDropResponse(
            recorded=True, delivery="notify_failed"
        )
        clock.at = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
        await _run(context_factory, services, listed=UUIDS)
        services.backend.report_rented_gpu_drop.return_value = NOTIFIED
        clock.at = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
        await _run(context_factory, services, listed=UUIDS[:5])

    assert _states(services) == ["fault", "recovered", "fault"]
    first, _, again = services.backend.report_rented_gpu_drop.await_args_list
    assert first.kwargs["first_seen_at"] == "2026-01-01T00:00:00+00:00"
    assert again.kwargs["first_seen_at"] == "2026-01-01T00:30:00+00:00"
    assert again.kwargs["consecutive_cycles"] == 1


@pytest.mark.asyncio
async def test_dry_run_logs_the_drop_and_posts_nothing(context_factory):
    services = _services()

    with patch.object(rented_gpu_drop.settings, "DRY_RUN", True):
        result = await _run(context_factory, services, listed=UUIDS[:5])
        cleared = await _run(context_factory, services, listed=UUIDS)

    assert result.event.reason_code == Msg.DROP.reason
    assert cleared.event.reason_code == Msg.OK.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_incident_through_a_redis_outage_is_posted_once_and_recovered(context_factory):
    services = _services(redis=FakeRedis(failing=True))

    first = await _run(context_factory, services, listed=UUIDS[:5])
    await _run(context_factory, services, listed=UUIDS[:5])
    recovered = await _run(context_factory, services, listed=UUIDS)

    assert first.passed is True and first.event.reason_code == Msg.DROP.reason
    assert recovered.event.reason_code == Msg.RECOVERED.reason
    assert _states(services) == ["fault", "recovered"]
    assert rented_gpu_drop._LOCAL_MARKS == {}


@pytest.mark.asyncio
async def test_the_pipeline_check_with_default_settings_posts_nothing(context_factory, monkeypatch):
    monkeypatch.delenv("RENTED_GPU_DROP_CHECK_ENABLED", raising=False)
    defaults = type(rented_gpu_drop.settings)(_env_file=None)
    check = next(c for c in PipelineFactory.build_checks() if isinstance(c, RentedGpuDropCheck))
    services = _services()

    with patch.object(rented_gpu_drop, "settings", defaults):
        result = await check.run(
            _ctx(context_factory, services, listed=UUIDS[:5], scrape_error="NVMLError(999)")
        )

    assert result.passed and result.event.reason_code == Msg.DISABLED.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.calls == 0 and services.redis.store == {}


HEALTHY_CYCLES = [
    # (cards the scrape listed, gpu_scrape_error, anchored UUIDs): a healthy rented node over several cycles,
    # including an optional NVML query that answers NOT_SUPPORTED and a scrape that lists one card twice
    # (8 detail rows, 7 distinct UUIDs, anchored as the fingerprint check stores it)
    (UUIDS, None, UUIDS),
    (UUIDS, None, UUIDS),
    (UUIDS, "NVMLError_NotSupported(3)", UUIDS),
    (list(reversed(UUIDS)), None, UUIDS),
    (DUPLICATED, None, UUIDS[:7]),
    (UUIDS, None, UUIDS),
]


@pytest.mark.asyncio
async def test_a_healthy_rented_node_stays_quiet_over_many_cycles(context_factory):
    services = _services()

    reasons = [
        (
            await _run(context_factory, services, listed=listed, scrape_error=error, anchor=anchor)
        ).event.reason_code
        for listed, error, anchor in HEALTHY_CYCLES
    ]

    assert reasons == [Msg.OK.reason] * len(HEALTHY_CYCLES)
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.store == {}


def _split_pods() -> list[RentedPod]:
    return [
        RentedPod(pod_id=f"pod-{n}", container_name=f"container_{n}", gpu_count=n, status="RUNNING")
        for n in (3, 5)
    ]


@pytest.mark.asyncio
async def test_a_split_node_sends_each_pod_its_own_gpu_count_and_the_executor_totals(
    context_factory,
):
    services = _services()

    result = await _run(context_factory, services, listed=UUIDS[:7], pods=_split_pods())

    sent = {
        call.args[0]: call.kwargs
        for call in services.backend.report_rented_gpu_drop.await_args_list
    }
    assert {pod: kwargs["pod_gpu_count"] for pod, kwargs in sent.items()} == {
        "pod-3": 3,
        "pod-5": 5,
    }
    assert all(
        kwargs["rented_gpu_count"] == 8
        and kwargs["nvml_gpu_count"] == 8
        and kwargs["expected_gpu_count"] == 8
        and kwargs["visible_gpu_count"] == 7
        for kwargs in sent.values()
    )
    assert result.event.what_we_saw["rented_gpu_count"] == 8
    assert [pod["gpu_count"] for pod in result.event.what_we_saw["pods"]] == [3, 5]


@pytest.mark.asyncio
async def test_a_split_node_that_loses_another_card_is_reported_again(context_factory):
    services = _services()

    for listed in (UUIDS[:7], UUIDS[:7], UUIDS[:6], UUIDS[:6]):
        await _run(context_factory, services, listed=listed, pods=_split_pods())

    calls = services.backend.report_rented_gpu_drop.await_args_list
    assert [(call.args[0], call.kwargs["visible_gpu_count"]) for call in calls] == [
        ("pod-3", 7),
        ("pod-5", 7),
        ("pod-3", 6),
        ("pod-5", 6),
    ]
    assert calls[-1].kwargs["missing_uuids"] == UUIDS[6:]
    assert calls[0].kwargs["first_seen_at"] == calls[-1].kwargs["first_seen_at"]


@pytest.mark.asyncio
async def test_counts_above_the_backend_cap_are_clamped(context_factory):
    services = _services()
    big = [f"GPU-b0b{i:03d}-0000-0000-0000-000000000000" for i in range(80)]
    ctx = _ctx(context_factory, services, listed=big[:70], count=80, gpu_count=80)
    ctx.verified["uuids"] = ",".join(big)

    result = await RentedGpuDropCheck().run(ctx)

    assert result.event.reason_code == Msg.DROP.reason
    sent = services.backend.report_rented_gpu_drop.await_args.kwargs
    assert sent["expected_gpu_count"] == rented_gpu_drop.MAX_REPORTED_GPU_COUNT
    assert sent["visible_gpu_count"] == rented_gpu_drop.MAX_REPORTED_GPU_COUNT
    assert sent["pod_gpu_count"] == sent["rented_gpu_count"] == sent["nvml_gpu_count"]
    assert sent["nvml_gpu_count"] == rented_gpu_drop.MAX_REPORTED_GPU_COUNT


@pytest.mark.parametrize(
    "build",
    [
        PipelineFactory.build_checks,
        PipelineFactory.build_dry_run_checks,
        PipelineFactory.build_fast_path_checks,
    ],
)
def test_the_check_runs_after_the_scrape_and_before_the_fatal_gpu_checks(build):
    kinds = [type(check) for check in build()]

    at = kinds.index(RentedGpuDropCheck)
    assert kinds[at - 1] is MachineSpecScrapeCheck
    assert at < kinds.index(GpuModelValidCheck) < kinds.index(TenantEnforcementCheck)
    assert RentedGpuDropCheck.fatal is False


# --- an incident replayed ----------------------------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.at = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self, tz=None):
        return self.at


DROP_CYCLES = [
    # (validator check time, cards the scrape listed, gpu_scrape_error)
    ("2026-01-01T00:00:00+00:00", UUIDS[:5], "NVMLError(999)"),
    ("2026-01-01T00:15:00+00:00", UUIDS[:5], "NVMLError(999)"),
    ("2026-01-01T01:30:00+00:00", UUIDS, None),
]


@pytest.mark.asyncio
async def test_incident_replay_alerts_on_the_first_cycle_once_and_recovers_once(context_factory):
    services = _services()
    clock = _Clock()
    seen = []

    with patch.object(rented_gpu_drop, "datetime", clock):
        for at, listed, error in DROP_CYCLES:
            clock.at = datetime.fromisoformat(at)
            result = await _run(context_factory, services, listed=listed, scrape_error=error)
            seen.append(
                (at, result.event.reason_code, services.backend.report_rented_gpu_drop.await_count)
            )

    assert seen == [
        ("2026-01-01T00:00:00+00:00", "RENTED_GPU_DROP", 1),
        ("2026-01-01T00:15:00+00:00", "RENTED_GPU_DROP", 1),
        ("2026-01-01T01:30:00+00:00", "RENTED_GPU_RECOVERED", 2),
    ]
    fault, recovered = services.backend.report_rented_gpu_drop.await_args_list
    assert fault.kwargs["first_seen_at"] == "2026-01-01T00:00:00+00:00"
    assert recovered.kwargs["state"] == "recovered"
    assert recovered.kwargs["first_seen_at"] == "2026-01-01T00:00:00+00:00"
    assert recovered.kwargs["consecutive_cycles"] == 2
