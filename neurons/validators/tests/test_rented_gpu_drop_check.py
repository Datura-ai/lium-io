"""RENTED_GPU_DROP: a rented node that lost a GPU is reported the cycle it is seen, once per incident."""

from datetime import UTC, datetime
from pathlib import Path
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
from neurons.validators.src.services.task.messages import RentedGpuDropMessages as Msg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedGpuDropResponse,
    RentedPod,
)

from tests.helpers import FakeRedis, build_services, build_state, default_executor

MODEL = "NVIDIA GeForce RTX 5090"
UUIDS = [f"GPU-a0a{i}-0000-0000-0000-00000000000{i}" for i in range(8)]
POD_ID = "00000000-0000-4000-8000-000000000001"
NOTIFIED = RentedGpuDropResponse(recorded=True, delivery="notified")


@pytest.fixture(autouse=True)
def _check_on():
    with patch.object(rented_gpu_drop.settings, "RENTED_GPU_DROP_CHECK_ENABLED", True):
        yield


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


# --- the pure rule ------------------------------------------------------------------------------------------------


def test_nvml_error_code_reads_the_scrape_repr():
    assert nvml_error_code("NVMLError(999)") == 999
    assert nvml_error_code("NVMLError_GpuIsLost(15)") == 15
    assert nvml_error_code("OSError('libnvidia-ml.so.1: cannot open')") is None
    assert nvml_error_code(None) is None


def test_a_count_drop_names_every_fault_and_the_unlisted_uuids():
    drop = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=8,
        listed_uuids=UUIDS[:5],
        listed_count=5,
        scrape_error="NVMLError(999)",
    )

    assert drop is not None
    assert drop.expected == 8 and drop.visible == 5
    assert drop.faults == [
        FAULT_BELOW_RENTED,
        FAULT_DETAILS_SHORT,
        FAULT_ANCHORED_MISSING,
        FAULT_NVML_ERROR,
    ]
    assert drop.missing_uuids == UUIDS[5:]
    assert drop.nvml_error_code == 999


def test_an_nvml_loss_code_is_a_fault_even_with_every_card_listed():
    drop = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=8,
        listed_uuids=UUIDS,
        listed_count=8,
        scrape_error="NVMLError(15)",
    )

    assert drop is not None and drop.faults == [FAULT_NVML_ERROR] and drop.visible == 8


def test_an_nvml_error_that_is_not_a_loss_code_with_every_card_listed_is_not_a_fault():
    assert (
        judge_rented_gpus(
            rented_gpu_count=8,
            anchor_uuids=UUIDS,
            nvml_count=8,
            listed_uuids=UUIDS,
            listed_count=8,
            scrape_error="NVMLError(3)",
        )
        is None
    )


def test_a_split_node_missing_a_card_outside_the_rental_is_still_reported_by_its_anchor():
    drop = judge_rented_gpus(
        rented_gpu_count=4,
        anchor_uuids=UUIDS,
        nvml_count=7,
        listed_uuids=UUIDS[:7],
        listed_count=7,
        scrape_error=None,
    )

    assert (
        drop is not None
        and drop.faults == [FAULT_ANCHORED_MISSING]
        and drop.missing_uuids == [UUIDS[7]]
    )


def test_a_card_listed_twice_is_counted_by_its_rows_not_its_distinct_uuids():
    duplicated = UUIDS[:7] + [UUIDS[0]]

    assert (
        judge_rented_gpus(
            rented_gpu_count=8,
            anchor_uuids=UUIDS[:7],
            nvml_count=8,
            listed_uuids=duplicated,
            listed_count=8,
            scrape_error=None,
        )
        is None
    )
    drop = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=8,
        listed_uuids=duplicated,
        listed_count=8,
        scrape_error=None,
    )
    assert drop is not None and drop.faults == [FAULT_ANCHORED_MISSING] and drop.visible == 8


# --- the check ----------------------------------------------------------------------------------------------------


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
async def test_an_nvml_error_with_every_card_listed_is_reported(context_factory):
    services = _services()

    result = await _run(
        context_factory, services, listed=UUIDS, scrape_error="NVMLError_GpuIsLost(15)"
    )

    assert result.event.reason_code == Msg.DROP.reason
    assert result.event.what_we_saw["faults"] == [FAULT_NVML_ERROR]
    services.backend.report_rented_gpu_drop.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_healthy_rented_node_posts_nothing_and_writes_no_mark(context_factory):
    services = _services()

    result = await _run(context_factory, services, listed=UUIDS)

    assert result.event.reason_code == Msg.OK.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.store == {}


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
    ],
    ids=["no_answer", "notify_failed", "backend_flag_off"],
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
async def test_recovery_is_posted_once_when_every_card_is_back(context_factory):
    services = _services()
    await _run(context_factory, services, listed=UUIDS[:5], scrape_error="NVMLError(999)")
    services.backend.report_rented_gpu_drop.return_value = RentedGpuDropResponse(
        recorded=True, delivery="notified"
    )

    recovered = await _run(context_factory, services, listed=UUIDS)
    after = await _run(context_factory, services, listed=UUIDS)

    assert recovered.event.reason_code == Msg.RECOVERED.reason
    assert after.event.reason_code == Msg.OK.reason
    states = [
        call.kwargs["state"] for call in services.backend.report_rented_gpu_drop.await_args_list
    ]
    assert states == ["fault", "recovered"]
    assert services.redis.store == {}


@pytest.mark.asyncio
async def test_a_refused_recovery_notice_is_posted_again(context_factory):
    services = _services()
    await _run(context_factory, services, listed=UUIDS[:5])
    services.backend.report_rented_gpu_drop.return_value = RentedGpuDropResponse(
        recorded=True, delivery="notify_failed"
    )
    await _run(context_factory, services, listed=UUIDS)
    services.backend.report_rented_gpu_drop.return_value = NOTIFIED
    await _run(context_factory, services, listed=UUIDS)
    await _run(context_factory, services, listed=UUIDS)

    states = [
        call.kwargs["state"] for call in services.backend.report_rented_gpu_drop.await_args_list
    ]
    assert states == ["fault", "recovered", "recovered"]


@pytest.mark.asyncio
async def test_no_recovery_is_posted_for_an_incident_the_backend_never_recorded(context_factory):
    services = _services(answer=RentedGpuDropResponse(recorded=False, delivery="disabled"))
    await _run(context_factory, services, listed=UUIDS[:5])

    await _run(context_factory, services, listed=UUIDS)

    states = [
        call.kwargs["state"] for call in services.backend.report_rented_gpu_drop.await_args_list
    ]
    assert states == ["fault"]
    assert services.redis.store == {}


@pytest.mark.asyncio
async def test_dry_run_logs_the_drop_and_posts_nothing(context_factory):
    services = _services()

    with patch.object(rented_gpu_drop.settings, "DRY_RUN", True):
        result = await _run(context_factory, services, listed=UUIDS[:5])

    assert result.event.reason_code == Msg.DROP.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_down_still_reports_the_drop_every_cycle(context_factory):
    services = _services(redis=FakeRedis(failing=True))

    first = await _run(context_factory, services, listed=UUIDS[:5])
    await _run(context_factory, services, listed=UUIDS[:5])

    assert first.passed is True and first.event.reason_code == Msg.DROP.reason
    assert services.backend.report_rented_gpu_drop.await_count == 2


def test_a_scrape_cut_short_by_a_non_loss_nvml_error_waits_for_confirmation():
    cut = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=8,
        listed_uuids=UUIDS[:3],
        listed_count=3,
        scrape_error="NVMLError_Timeout(10)",
    )
    lost = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=8,
        listed_uuids=UUIDS[:5],
        listed_count=5,
        scrape_error="NVMLError(999)",
    )
    driver_short = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=7,
        listed_uuids=UUIDS[:7],
        listed_count=7,
        scrape_error="NVMLError_NotSupported(3)",
    )
    cut_and_driver_short = judge_rented_gpus(
        rented_gpu_count=8,
        anchor_uuids=UUIDS,
        nvml_count=7,
        listed_uuids=UUIDS[:5],
        listed_count=5,
        scrape_error="NVMLError_Timeout(10)",
    )

    assert cut is not None and cut.confirm_first is True
    assert lost is not None and lost.confirm_first is False
    assert driver_short is not None and driver_short.confirm_first is False
    assert cut_and_driver_short is not None and cut_and_driver_short.confirm_first is False


@pytest.mark.asyncio
async def test_a_one_off_scrape_timeout_posts_nothing_and_a_second_one_posts(context_factory):
    services = _services()
    timeout = {"listed": UUIDS[:3], "scrape_error": "NVMLError_Timeout(10)"}

    first = await _run(context_factory, services, **timeout)
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert first.event.reason_code == Msg.DROP.reason
    assert first.event.what_we_saw["pods"][0]["held"] is True

    await _run(context_factory, services, **timeout)
    services.backend.report_rented_gpu_drop.assert_awaited_once()
    assert services.backend.report_rented_gpu_drop.await_args.kwargs["consecutive_cycles"] == 2


@pytest.mark.asyncio
async def test_a_scrape_timeout_with_the_driver_count_below_the_rental_posts_on_the_first_cycle(
    context_factory,
):
    services = _services()

    result = await _run(
        context_factory, services, listed=UUIDS[:5], count=7, scrape_error="NVMLError_Timeout(10)"
    )

    assert result.event.what_we_saw["pods"][0]["held"] is False
    services.backend.report_rented_gpu_drop.assert_awaited_once()
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["consecutive_cycles"] == 1
    assert call.kwargs["nvml_gpu_count"] == 7 and call.kwargs["visible_gpu_count"] == 5
    assert call.kwargs["rented_gpu_count"] == 8


@pytest.mark.asyncio
async def test_a_one_off_scrape_timeout_that_clears_is_never_posted(context_factory):
    services = _services()

    await _run(context_factory, services, listed=UUIDS[:3], scrape_error="NVMLError_Timeout(10)")
    await _run(context_factory, services, listed=UUIDS)

    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.store == {}


def test_the_check_does_not_import_the_rented_pod_ssh_module():
    assert "rented_pod_ssh" not in Path(rented_gpu_drop.__file__).read_text()


@pytest.mark.asyncio
async def test_the_check_off_does_nothing(context_factory):
    services = _services()

    with patch.object(rented_gpu_drop.settings, "RENTED_GPU_DROP_CHECK_ENABLED", False):
        result = await _run(context_factory, services, listed=UUIDS[:5])

    assert result.event.reason_code == Msg.DISABLED.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    assert services.redis.calls == 0


def test_the_check_is_off_by_default():
    field = type(rented_gpu_drop.settings).model_fields["RENTED_GPU_DROP_CHECK_ENABLED"]
    assert field.default is False


HEALTHY_CYCLES = [
    # (cards the scrape listed, gpu_scrape_error, anchored UUIDs): a healthy rented node over several cycles,
    # including an optional NVML query that answers NOT_SUPPORTED and a scrape that lists one card twice
    # (8 detail rows, 7 distinct UUIDs, anchored as the fingerprint check stores it)
    (UUIDS, None, UUIDS),
    (UUIDS, None, UUIDS),
    (UUIDS, "NVMLError_NotSupported(3)", UUIDS),
    (list(reversed(UUIDS)), None, UUIDS),
    (UUIDS[:7] + [UUIDS[0]], None, UUIDS[:7]),
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


@pytest.mark.asyncio
async def test_a_split_node_sends_each_pod_its_own_gpu_count_and_the_executor_totals(
    context_factory,
):
    services = _services()
    pods = [
        RentedPod(pod_id=f"pod-{n}", container_name=f"container_{n}", gpu_count=n, status="RUNNING")
        for n in (3, 5)
    ]

    result = await _run(context_factory, services, listed=UUIDS[:7], pods=pods)

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
    "build", [PipelineFactory.build_checks, PipelineFactory.build_dry_run_checks]
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
