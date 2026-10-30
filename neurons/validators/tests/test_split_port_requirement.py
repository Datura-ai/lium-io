"""DAH-3698 — a node offered as GPU splits lists only with MIN_PORT_COUNT free verified ports for each of
its gpu_count // gpu_splitting_min_count splits; below that PortCountCheck fails INSUFFICIENT_PORTS."""

from __future__ import annotations

import pytest

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import MIN_PORT_COUNT

from tests.helpers import build_state, make_context


def _ctx(
    ports: int,
    *,
    gpu_count: int = 8,
    min_count: int | None = 1,
    rented: bool = False,
    declared: int | None = None,
):
    rented_data = None
    if rented:
        rented_data = RentedExecutorsResponse(
            executors={
                "executor-123": RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address="127.0.0.1",
                    executor_ip_port="8001",
                    pods=[
                        RentedPod(
                            pod_id="pod-1", container_name="container_pod-1", rented_ports=[65000]
                        )
                    ],
                )
            }
        )
    state = build_state(
        verified_port_count=ports,
        declared_port_count=declared,
        gpu_count=gpu_count,
        supports_gpu_splitting=min_count is not None,
        gpu_splitting_min_count=min_count,
        rented_data=rented_data,
    )
    ctx = make_context(state=state)
    return ctx.model_copy(
        update={"executor": ctx.executor.model_copy(update={"uuid": "executor-123"})}
    )


@pytest.mark.asyncio
async def test_exactly_enough_ports_for_every_split_passes():
    result = await PortCountCheck().run(_ctx(8 * MIN_PORT_COUNT))
    assert result.passed is True
    assert result.event.reason_code == PortCountMessages.PORT_COUNT_RECORDED.reason


@pytest.mark.asyncio
async def test_one_port_short_fails_with_the_split_count_and_the_ports_to_open():
    result = await PortCountCheck().run(_ctx(8 * MIN_PORT_COUNT - 1))
    assert result.passed is False
    assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    what = result.event.what_we_saw
    assert (what["available_port_count"], what["required"], what["max_splits"]) == (23, 24, 8)
    assert "Open at least 24 renting ports for this node's 8 GPU splits" in result.event.remediation
    assert result.updates["state"].specs["available_port_count"] == 23


@pytest.mark.asyncio
async def test_two_gpu_splits_need_ports_for_four_splits():
    assert (await PortCountCheck().run(_ctx(4 * MIN_PORT_COUNT, min_count=2))).passed is True
    short = await PortCountCheck().run(_ctx(4 * MIN_PORT_COUNT - 1, min_count=2))
    assert short.passed is False and short.event.what_we_saw["max_splits"] == 4


@pytest.mark.asyncio
async def test_non_split_node_needs_only_the_plain_floor():
    result = await PortCountCheck().run(_ctx(MIN_PORT_COUNT, min_count=None))
    assert result.passed is True


@pytest.mark.asyncio
async def test_rented_split_node_keeps_its_exemption():
    result = await PortCountCheck().run(_ctx(MIN_PORT_COUNT, rented=True))
    assert result.passed is True


def _rented_now(*, pods: bool = False, filler_ports: list[int] | None = None) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8001",
                pods=[RentedPod(pod_id="pod-1", container_name="container_pod-1", rented_ports=[65000])]
                if pods
                else [],
            )
        },
        filler_ports_by_executor={"executor-123": filler_ports} if filler_ports else {},
    )


@pytest.mark.asyncio
async def test_a_rent_that_started_after_the_snapshot_keeps_the_rented_exemption():
    ctx = _ctx(8 * MIN_PORT_COUNT - 3)
    ctx.services.backend.get_rented_executors_now.return_value = _rented_now(pods=True)
    result = await PortCountCheck().run(ctx)
    assert result.passed is True
    assert result.updates["state"].rented_data.executors["executor-123"].pods


@pytest.mark.asyncio
async def test_a_filler_that_started_after_the_snapshot_counts_its_ports():
    ctx = _ctx(8 * MIN_PORT_COUNT - 3)
    ctx.services.backend.get_rented_executors_now.return_value = _rented_now(filler_ports=[61000, 61001, 61002])
    result = await PortCountCheck().run(ctx)
    assert result.passed is True
    assert result.updates["port_count"] == 8 * MIN_PORT_COUNT


@pytest.mark.asyncio
async def test_no_fresh_rental_data_keeps_the_failure():
    ctx = _ctx(8 * MIN_PORT_COUNT - 3)  # the helper's backend answers None
    assert (await PortCountCheck().run(ctx)).passed is False


@pytest.mark.asyncio
async def test_a_listed_split_node_is_hidden_at_23_ports_and_listed_again_at_24():
    from neurons.validators.src.services.task.availability import availability_errors

    def errors(result):
        return [e.reason_code for e in availability_errors([result.event])]

    # the backend skips a zero-score report, so only an availability entry takes the listing down
    assert errors(await PortCountCheck().run(_ctx(24))) == []
    short = await PortCountCheck().run(_ctx(23))
    assert errors(short) == [PortCountMessages.INSUFFICIENT_PORTS.reason]
    assert short.event.what_we_saw["reach_target"] == "executor_ports"
    assert errors(await PortCountCheck().run(_ctx(24))) == []


async def _guarded(declared: int) -> list:
    """Six listed nodes verified at 23 ports, declaring `declared`, through the outage guard."""
    from neurons.validators.src.services.task.availability import (
        availability_errors,
        silence_availability_errors_on_our_own_outage,
    )
    from services.task.models import JobResult

    results = []
    for _ in range(6):
        event = (await PortCountCheck().run(_ctx(23, declared=declared))).event
        entries = [e.model_dump(mode="json") for e in availability_errors([event])]
        results.append(JobResult.model_construct(availability_errors=entries))
    silence_availability_errors_on_our_own_outage(results)
    return results


@pytest.mark.asyncio
async def test_the_outage_guard_counts_and_silences_a_split_probe_shortfall():
    # enough ports declared, too few verified: a fleet-wide probe failure on our side must not hide every split node
    assert all(result.availability_errors is None for result in await _guarded(declared=24))


@pytest.mark.asyncio
async def test_the_outage_guard_keeps_a_split_node_that_declares_too_few_ports():
    # too few ports declared is the node's configuration, not our probe: the listing comes down
    results = await _guarded(declared=23)
    assert all(
        [e["reason_code"] for e in result.availability_errors] == ["INSUFFICIENT_PORTS"]
        for result in results
    )


def _rented_by(miner_hotkey: str, address: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey=miner_hotkey,
                executor_ip_address=address,
                executor_ip_port="22",  # the helper executor's port
                pods=[RentedPod(pod_id="pod-1", container_name="container_pod-1", rented_ports=[65000])],
            )
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "miner_hotkey, address, rented",
    [
        ("miner-hotkey", "127.0.0.1", True),  # same miner
        ("other-hotkey", "127.0.0.1", True),  # another miner, same endpoint: the renter's pod is on this box
        ("other-hotkey", "10.0.0.9", False),  # another miner, another endpoint: not this node's rental
    ],
)
async def test_the_split_requirement_sees_the_rental_the_tenant_check_judges(miner_hotkey, address, rented):
    from dataclasses import replace

    from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
    from neurons.validators.src.services.task.checks.rented_machine import _rental_of_validated_miner

    ctx = _ctx(MIN_PORT_COUNT)
    ctx = ctx.model_copy(update={"state": replace(ctx.state, rented_data=_rented_by(miner_hotkey, address))})
    rental = ctx.state.rented_data.executors["executor-123"]
    assert (_rental_of_validated_miner(ctx, rental) is not None) is rented

    result = await PortCountCheck().run(ctx)
    assert result.passed is rented
    if not rented:
        assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason

    from unittest.mock import AsyncMock

    probe = AsyncMock()
    probe.verify_ports.side_effect = RuntimeError("stop after the probe call")
    ctx = ctx.model_copy(update={"services": replace(ctx.services, connectivity=probe)})
    with pytest.raises(RuntimeError):
        await PortConnectivityCheck().run(ctx)
    assert probe.verify_ports.call_args.kwargs["min_ports"] == (
        MIN_PORT_COUNT if rented else 8 * MIN_PORT_COUNT
    )
