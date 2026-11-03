"""RENTED_GPU_DROP: a rented node that lost a GPU is reported the cycle it is seen, once per incident."""

import json
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.checks import (
    MachineSpecScrapeCheck,
    RentedGpuDropCheck,
    rented_gpu_drop,
)
from neurons.validators.src.services.task.checks.rented_gpu_drop import (
    FAULT_ANCHORED_MISSING,
    FAULT_BELOW_RENTED,
    FAULT_DETAILS_SHORT,
    FAULT_ECC_UNCORRECTED,
    FAULT_NVML_ERROR,
    judge_rented_gpus,
)
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
from tests.test_gpu_fault_probe_check import FakeRunner
from tests.test_gpu_memory_health_check import _report
from tests.test_machine_spec_scrape_check import DummySSHCommandRunner, make_command_result

MODEL = "NVIDIA GeForce RTX 5090"
UUIDS = [f"GPU-a0a{i}-0000-0000-0000-00000000000{i}" for i in range(8)]
POD_ID = "00000000-0000-4000-8000-000000000001"
NOTIFIED = RentedGpuDropResponse(recorded=True, delivery="notified")


@pytest.fixture(autouse=True)
def _clear_local_marks():
    rented_gpu_drop._LOCAL_MARKS.clear()
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
    owner: str = "miner-hotkey",
) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey=owner,
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


def _states(services) -> list[str]:
    return [
        call.kwargs["state"] for call in services.backend.report_rented_gpu_drop.await_args_list
    ]


# --- the pure rule ------------------------------------------------------------------------------------------------


BR, DS, AM, NV = FAULT_BELOW_RENTED, FAULT_DETAILS_SHORT, FAULT_ANCHORED_MISSING, FAULT_NVML_ERROR


# --- the check ----------------------------------------------------------------------------------------------------


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
async def test_a_scrape_that_lists_no_gpu_on_an_idle_node_posts_nothing(context_factory):
    services = _services()

    result = await _scrape_listing_no_gpu(
        context_factory, services, count=8, scrape_error="NVMLError(999)", status="STOPPED"
    )

    assert result.passed is False
    services.backend.report_rented_gpu_drop.assert_not_awaited()
    # only the GPU loss hold (DAH-4457) is written; no rented-GPU-drop mark
    assert [key.split(":")[0] for key in services.redis.store] == ["gpu_loss_hold"]


async def _run_with_memory_read(context_factory, services, stdout: str):
    ctx = _ctx(context_factory, services, listed=UUIDS).model_copy(
        update={"runner": FakeRunner(stdout)}
    )
    return await RentedGpuDropCheck().run(ctx), ctx


@pytest.mark.asyncio
async def test_a_listed_card_with_uncorrected_ecc_is_reported_as_lost(context_factory):
    services = _services()
    flags = [f"gpu 3 ({UUIDS[3]}): 1 uncorrected ECC error(s) since the driver loaded"]

    result, ctx = await _run_with_memory_read(
        context_factory, services, _report("flagged", flags=flags)
    )

    assert result.event.reason_code == Msg.DROP.reason
    assert result.event.what_we_saw["faults"] == [FAULT_ECC_UNCORRECTED]
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["faults"] == [FAULT_ECC_UNCORRECTED]
    assert call.kwargs["missing_uuids"] == [UUIDS[3]]
    assert call.kwargs["expected_gpu_count"] == call.kwargs["visible_gpu_count"] == 8
    # GpuMemoryHealthCheck reuses this read instead of a second SSH run
    assert result.updates["state"].memory_health.report["flags"] == flags
    assert len(ctx.runner.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_memory_read_does_not_recover_an_uncorrected_ecc_incident(context_factory):
    services = _services()
    flags = [f"gpu 3 ({UUIDS[3]}): 1 uncorrected ECC error(s) since the driver loaded"]
    await _run_with_memory_read(context_factory, services, _report("flagged", flags=flags))

    unread, _ = await _run_with_memory_read(
        context_factory, services, _report("error", flags=[], nvml={"available": False})
    )
    assert unread.event.reason_code == Msg.OK.reason
    assert _states(services) == ["fault"]

    healthy = {"available": True, "gpus": [{"index": 3, "uuid": UUIDS[3], "ecc_uncorrected": 0}]}
    healed, _ = await _run_with_memory_read(
        context_factory, services, _report("ok", flags=[], nvml=healthy)
    )
    assert healed.event.reason_code == Msg.RECOVERED.reason
    assert _states(services) == ["fault", "recovered"]


ECC_FLAGS = [f"gpu 3 ({UUIDS[3]}): 1 uncorrected ECC error(s) since the driver loaded"]


@pytest.mark.asyncio
async def test_a_broken_card_on_a_partly_rented_node_is_named(context_factory):
    # counts intact: a report naming no card is refused by the backend, which maps the UUID to its renter
    services = _services()
    ctx = _ctx(context_factory, services, listed=UUIDS, gpu_count=1).model_copy(
        update={"runner": FakeRunner(_report("flagged", flags=ECC_FLAGS))}
    )

    result = await RentedGpuDropCheck().run(ctx)

    assert result.event.what_we_saw["faults"] == [FAULT_ECC_UNCORRECTED]
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["faults"] == [FAULT_ECC_UNCORRECTED]
    assert call.kwargs["missing_uuids"] == [UUIDS[3]]
    assert call.kwargs["expected_gpu_count"] == call.kwargs["visible_gpu_count"] == 8


def test_a_card_off_the_bus_and_a_broken_card_are_both_named_on_a_part_rental():
    # 8 anchored cards, one fell off the bus so NVML and the scrape list 7, and the rental holds 7
    drop = judge_rented_gpus(
        rented_gpu_count=7,
        anchor_uuids=UUIDS,
        nvml_count=7,
        listed_uuids=UUIDS[:7],
        listed_count=7,
        scrape_error=None,
        broken_uuids=[UUIDS[3]],
    )

    assert drop.missing_uuids == [UUIDS[7], UUIDS[3]]


UNREAD_MEMORY = [
    pytest.param(
        _report("flagged", flags=["gpu 3: no uuid in this flag"]), id="flag-names-no-card"
    ),
    pytest.param(
        _report(
            "ok",
            flags=[],
            nvml={
                "available": True,
                "gpus": [{"index": 3, "uuid": UUIDS[3], "ecc_uncorrected": None}],
            },
        ),
        id="ecc-read-failed-on-a-card",
    ),
    pytest.param(
        _report("ok", flags=[], nvml={"available": True, "gpus": []}), id="nvml-lists-no-card"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("unread_stdout", UNREAD_MEMORY)
async def test_a_memory_read_that_proves_nothing_keeps_an_ecc_incident_open(
    context_factory, unread_stdout
):
    redis = FakeRedis()
    services = _services(redis=redis)
    key = f"rented_gpu_drop:{POD_ID}"
    await _run_with_memory_read(context_factory, services, _report("flagged", flags=ECC_FLAGS))
    redis.ttl.pop(key)

    unread, _ = await _run_with_memory_read(context_factory, services, unread_stdout)

    assert unread.event.reason_code == Msg.OK.reason
    assert _states(services) == ["fault"]
    # the held mark is written again, so its TTL does not run out while NVML stays unread
    assert redis.ttl[key] == rented_gpu_drop.settings.RENTED_GPU_DROP_STATE_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_healthy_memory_read_reports_nothing(context_factory):
    services = _services()

    result, _ = await _run_with_memory_read(context_factory, services, _report("ok"))

    assert result.event.reason_code == Msg.OK.reason
    services.backend.report_rented_gpu_drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_pipeline_check_with_default_settings_reports_the_drop(context_factory):
    defaults = type(rented_gpu_drop.settings)(_env_file=None)
    check = next(c for c in PipelineFactory.build_checks() if isinstance(c, RentedGpuDropCheck))
    services = _services()

    with patch.object(rented_gpu_drop, "settings", defaults):
        result = await check.run(_ctx(context_factory, services, listed=UUIDS[:5]))

    assert result.passed and result.event.reason_code == Msg.DROP.reason
    services.backend.report_rented_gpu_drop.assert_awaited_once()


# --- an incident replayed ----------------------------------------------------------------------------------------------


# --- a whole set of other cards: the post carries them, so the backend can move the pod at once -----------------------


@pytest.mark.asyncio
async def test_a_whole_set_of_other_cards_is_posted_with_the_new_cards(context_factory):
    services = _services()
    new = [f"GPU-b0b{i}-0000-0000-0000-00000000000{i}" for i in range(8)]
    ctx = _ctx(context_factory, services, listed=new).model_copy(
        update={"runner": FakeRunner(_report("ok"))}
    )

    await RentedGpuDropCheck().run(ctx)

    call = services.backend.report_rented_gpu_drop.await_args
    assert FAULT_ANCHORED_MISSING in call.kwargs["faults"]
    assert call.kwargs["missing_uuids"] == sorted(UUIDS)
    assert call.kwargs["gpu_details"] == _details(new)


@pytest.mark.asyncio
async def test_a_fault_with_every_anchored_card_listed_posts_no_cards(context_factory):
    services = _services()
    flags = [f"gpu 3 ({UUIDS[3]}): 1 uncorrected ECC error(s) since the driver loaded"]

    await _run_with_memory_read(context_factory, services, _report("flagged", flags=flags))

    assert services.backend.report_rented_gpu_drop.await_args.kwargs["gpu_details"] is None


def _at(*isos: str):
    """Patch the check's clock to return these cycle times in turn."""
    from datetime import datetime as real

    clock = patch.object(rented_gpu_drop, "datetime")
    mocked = clock.start()
    mocked.now.side_effect = [real.fromisoformat(iso) for iso in isos]
    return clock


@pytest.mark.asyncio
async def test_reports_carry_the_observation_time_and_a_retried_recovery_keeps_it(context_factory):
    t_fault, t_clean, t_retry = (
        "2026-11-02T10:00:00+00:00",
        "2026-11-02T10:05:00+00:00",
        "2026-11-02T10:10:00+00:00",
    )
    services = _services()
    clock = _at(t_fault, t_clean, t_retry)
    try:
        await RentedGpuDropCheck().run(_ctx(context_factory, services, listed=UUIDS[:7], count=7))
        services.backend.report_rented_gpu_drop.return_value = None  # the recovery goes unanswered
        await RentedGpuDropCheck().run(_ctx(context_factory, services, listed=UUIDS))
        services.backend.report_rented_gpu_drop.return_value = NOTIFIED
        await RentedGpuDropCheck().run(_ctx(context_factory, services, listed=UUIDS))
    finally:
        clock.stop()

    calls = services.backend.report_rented_gpu_drop.await_args_list
    assert [(c.kwargs["state"], c.kwargs["observed_at"]) for c in calls] == [
        ("fault", t_fault),
        ("recovered", t_clean),
        ("recovered", t_clean),
    ]


@pytest.mark.asyncio
async def test_the_backend_client_posts_observed_at():
    from unittest.mock import AsyncMock

    from neurons.validators.src.clients.backend_client import BackendClient

    client = BackendClient.__new__(BackendClient)
    client.post = AsyncMock(return_value=None)
    kwargs = dict(
        state="recovered",
        executor_id="e",
        first_seen_at="t0",
        consecutive_cycles=1,
        expected_gpu_count=0,
        visible_gpu_count=0,
        missing_uuids=[],
        nvml_error_code=None,
        faults=[],
        pod_gpu_count=8,
        rented_gpu_count=8,
        nvml_gpu_count=8,
    )
    await client.report_rented_gpu_drop(POD_ID, observed_at="t1", **kwargs)
    assert client.post.await_args.kwargs["json_data"]["observed_at"] == "t1"
    await client.report_rented_gpu_drop(POD_ID, **kwargs)
    assert "observed_at" not in client.post.await_args.kwargs["json_data"]
