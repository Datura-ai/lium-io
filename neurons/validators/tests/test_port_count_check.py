from __future__ import annotations

import pytest

from neurons.validators.src.services.task.checks.finalize import FinalizeCheck
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import FinalizeMessages, PortCountMessages as Msg
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.const import MIN_PORT_COUNT

from tests.helpers import build_state




@pytest.mark.asyncio
async def test_port_count_sufficient_passes(context_factory):
    """Check passes when port count >= MIN_PORT_COUNT."""
    ctx = context_factory(state=build_state(verified_port_count=MIN_PORT_COUNT + 1))

    # Act
    result = await PortCountCheck().run(ctx)

    # Assert
    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == MIN_PORT_COUNT + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("port_count", [0, MIN_PORT_COUNT - 1])
async def test_port_count_insufficient_fails_when_not_rented(context_factory, port_count):
    """Check fails when port count < MIN_PORT_COUNT and executor is not rented."""
    ctx = context_factory(state=build_state(verified_port_count=port_count))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.updates["port_count"] == port_count
    assert result.updates["state"].specs["available_port_count"] == port_count


@pytest.mark.asyncio
async def test_port_count_insufficient_passes_when_rented(context_factory):
    """Check passes when port count < MIN_PORT_COUNT but executor is rented."""
    rented_data = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="test-miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[8080, 8081])],
            )
        },
    )
    ctx = context_factory(state=build_state(verified_port_count=MIN_PORT_COUNT - 1, rented_data=rented_data))

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    assert result.updates["port_count"] == MIN_PORT_COUNT - 1
    assert result.event.severity == "warning"
    assert result.event.impact.endswith("; the rented portion is scored as rented")
    assert result.event.what_we_saw["exempt_because_rented"] is True
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == 0


@pytest.mark.asyncio
async def test_insufficient_ports_names_the_orphaned_container_holding_them(context_factory):
    """DAH-2991: the shortfall caused by an orphan the cleanup could not remove is named, not a bare count."""
    name = "pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7"
    ctx = context_factory(state=build_state(verified_port_count=2, orphaned_containers=[name]))

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.INSUFFICIENT_PORTS.reason
    assert result.event.what_we_saw["held_by_orphaned_containers"] == [name]
    assert name in result.event.remediation


EXECUTOR_UUID = "executor-123"
ANSWERED_PAIRS = [(9001, 40001), (9002, 40002)]
BACKGROUND_JOB_PORTS = [40010, 40011]


def _background_job_data(
    *,
    owner: str | None = "lium",
    filler_ports: list[int] | None = None,
    executor_uuid: str = EXECUTOR_UUID,
    renter_pods: list[RentedPod] | None = None,
) -> RentedExecutorsResponse:
    executors = {}
    if renter_pods is not None:
        executors[EXECUTOR_UUID] = RentedExecutor(
            miner_hotkey="test-miner",
            executor_ip_address="127.0.0.1",
            executor_ip_port="22",
            pods=renter_pods,
        )
    return RentedExecutorsResponse(
        executors=executors,
        filler_ports_by_executor={executor_uuid: BACKGROUND_JOB_PORTS if filler_ports is None else filler_ports},
        default_job_owner_by_executor={executor_uuid: owner} if owner is not None else {},
    )


def _state(rented_data: RentedExecutorsResponse | None, pairs=ANSWERED_PAIRS):
    return build_state(verified_port_count=len(pairs), verified_port_pairs=list(pairs), rented_data=rented_data)


@pytest.mark.asyncio
async def test_background_job_ports_lift_an_unrented_host_and_finalize_does_not_report_it_hidden(context_factory):
    ctx = context_factory(state=_state(_background_job_data()), score=1.0, job_score=1.0)
    listed_only_if = (
        "Listed only if the platform counts ports held by preemptible background jobs: "
        f"{len(ANSWERED_PAIRS)} verified ports plus {len(BACKGROUND_JOB_PORTS)} held, need {MIN_PORT_COUNT}"
    )

    result = await PortCountCheck().run(ctx)
    final_result = await FinalizeCheck().run(ctx.model_copy(update=result.updates))

    assert result.passed is True
    assert result.event.reason_code == Msg.PORT_COUNT_RECORDED.reason
    # the published figures stay the answered count: the platform adds these ports itself
    assert result.updates["port_count"] == len(ANSWERED_PAIRS)
    assert result.updates["state"].specs["available_port_count"] == len(ANSWERED_PAIRS)
    assert result.updates["state"].preemptible_background_job_port_count == len(BACKGROUND_JOB_PORTS)
    # passed under the published floor: listed only while the platform counts the held ports too
    # (its count_preemptible_filler_ports_as_free setting), so neither event claims the node is hidden
    assert result.event.severity == "warning"
    assert result.event.impact == (
        f"{listed_only_if}; scored with {len(BACKGROUND_JOB_PORTS)} ports held by preemptible background jobs"
    )
    assert result.event.what_we_saw["listing_hidden"] is None
    assert result.event.what_we_saw["exempt_because_rented"] is False
    assert final_result.event.reason_code == FinalizeMessages.COMPLETED.reason
    assert final_result.event.impact.startswith(f"{listed_only_if}.")
    assert final_result.event.what_we_saw["port_floor"]["listing_hidden"] is None
    # the firewall fix is offered only for the case where the platform does not list the node
    assert final_result.event.remediation.startswith("If the node is not listed for renters: ")


RENTER_POD = RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[40020])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rented_data", "pairs", "passed", "held"),
    [
        pytest.param(_background_job_data(filler_ports=[40010, 40011, 40012]), [], False, 0, id="no-port-answered"),
        pytest.param(_background_job_data(), ANSWERED_PAIRS[:1], True, 2, id="1-answered-plus-2-held-is-the-floor"),
        pytest.param(_background_job_data(filler_ports=[40010]), ANSWERED_PAIRS[:1], False, 1, id="1-plus-1-short"),
        pytest.param(
            _background_job_data(filler_ports=[ANSWERED_PAIRS[0][1]]), ANSWERED_PAIRS, False, 0, id="answered-and-held"
        ),
        pytest.param(None, ANSWERED_PAIRS, False, 0, id="no-backend-snapshot"),
        pytest.param(_background_job_data(owner="miner"), ANSWERED_PAIRS, False, 0, id="miner-default-job"),
        pytest.param(_background_job_data(owner=None), ANSWERED_PAIRS, False, 0, id="owner-not-reported"),
        pytest.param(_background_job_data(owner="future-owner"), ANSWERED_PAIRS, False, 0, id="unknown-owner"),
        pytest.param(
            _background_job_data(executor_uuid="another-executor"), ANSWERED_PAIRS, False, 0, id="another-executor"
        ),
        # the backend drops stale and terminal filler rows before it lists ports (daos/filler_run.py)
        pytest.param(_background_job_data(filler_ports=[]), ANSWERED_PAIRS, False, 0, id="no-live-job-rows"),
        # main passes a rented host whatever its count; the background-job ports are never added to it
        pytest.param(_background_job_data(renter_pods=[RENTER_POD]), ANSWERED_PAIRS, True, 0, id="renter-pod"),
        pytest.param(_background_job_data(renter_pods=[]), ANSWERED_PAIRS, True, 2, id="entry-without-pods"),
    ],
)
async def test_which_ports_count_toward_the_floor(context_factory, rented_data, pairs, passed, held):
    ctx = context_factory(state=_state(rented_data, pairs=pairs))

    result = await PortCountCheck().run(ctx)

    assert result.passed is passed
    assert result.event.what_we_saw["held_by_preemptible_background_jobs"] == held
    assert result.updates["port_count"] == len(pairs)
    assert result.updates["state"].specs["available_port_count"] == len(pairs)
