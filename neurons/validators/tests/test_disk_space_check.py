"""DAH-4961: the listing's disk floor as a validation failure of an idle node."""

from __future__ import annotations

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import MAX_DISK_UTILIZATION_PERCENT
from services.task.checks.disk_space import KB_PER_GB, DiskSpaceCheck
from services.task.messages import DiskSpaceMessages as Msg

from core.config import settings
from tests.helpers import build_services, build_state
from tests.test_port_floor_listing import PerPodBackendClient, PerPodSSHClient


def _disk(utilization: float, free_gb: float) -> dict:
    return {"hard_disk": {"utilization": utilization, "free": free_gb * KB_PER_GB}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "specs, reason",
    [
        (_disk(50, 500), Msg.OK.reason),
        (_disk(MAX_DISK_UTILIZATION_PERCENT + 1, 500), Msg.TOO_FULL.reason),
        (_disk(50, settings.MIN_FREE_DISK_GB - 1), Msg.FREE_TOO_LOW.reason),
        ({}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": 10}}, Msg.NOT_REPORTED.reason),
        (_disk(10, 50 - 1 / KB_PER_GB), Msg.FREE_TOO_LOW.reason),
        (_disk(10, 50), Msg.OK.reason),
        (_disk(10, 50 + 1 / KB_PER_GB), Msg.OK.reason),
        (_disk(89.999, 100), Msg.OK.reason),
        (_disk(90, 100), Msg.OK.reason),
        (_disk(90.001, 100), Msg.TOO_FULL.reason),
        ({"hard_disk": {"utilization": float("nan"), "free": 500 * KB_PER_GB}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": "NaN", "free": 500 * KB_PER_GB}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": 95, "free": float("inf")}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": 50, "free": "Infinity"}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": "-Infinity", "free": 500 * KB_PER_GB}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": None, "free": 500 * KB_PER_GB}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": 10, "free": None}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": "n/a", "free": 500 * KB_PER_GB}}, Msg.NOT_REPORTED.reason),
        ({"hard_disk": {"utilization": 10, "free": "lots"}}, Msg.NOT_REPORTED.reason),
    ],
)
async def test_idle_node_fails_the_listing_disk_floor(context_factory, specs, reason):
    ctx = context_factory(state=build_state(specs=specs))

    result = await DiskSpaceCheck().run(ctx)

    assert result.event.reason_code == reason
    assert result.passed is (reason == Msg.OK.reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner, address, port, live, exempt",
    [
        ("miner-hotkey", "127.0.0.1", "8080", True, True),  # this miner's rental
        ("other-hotkey", "127.0.0.1", "22", True, True),  # another miner, same endpoint: the pod is here
        ("other-hotkey", "10.0.0.1", "22", True, False),  # another miner elsewhere
        (None, None, None, False, False),  # no rental
    ],
)
async def test_live_rental_is_not_failed_for_its_disk(
    context_factory, owner, address, port, live, exempt
):
    """Tenant enforcement and disk enforcement agree whether the rental belongs on this node."""
    executors = {}
    if owner is not None:
        executors["executor-123"] = RentedExecutor(
            miner_hotkey=owner,
            executor_ip_address=address,
            executor_ip_port=port,
            pods=[RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[])],
        )
    rented_data = RentedExecutorsResponse(executors=executors)
    ctx = context_factory(
        state=build_state(specs=_disk(95, 500), rented_data=rented_data), rented=live
    )

    result = await DiskSpaceCheck().run(ctx)

    assert result.passed is exempt
    assert result.event.reason_code == Msg.TOO_FULL.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pod_ids, running, live, exempt",
    [
        (["pod-a", "pod-b"], {"pod-b"}, {"pod-b"}, True),  # split node: A ended, B still rented
        (["pod-a", "pod-b"], {"pod-a"}, {"pod-a"}, True),
        (["pod-a", "pod-b"], set(), set(), False),  # every listed pod ended
        (["pod-a"], set(), set(), False),  # the only rental ended since the cycle started
        (["pod-a", "pod-b"], {"pod-b"}, set(), False),  # both ended, B's container not torn down yet
    ],
)
async def test_ended_rental_exempts_only_while_a_listed_rental_is_active(
    context_factory, pod_ids, running, live, exempt
):
    rented_data = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=[
                    RentedPod(pod_id=p, container_name=f"container_{p}", rented_ports=[])
                    for p in pod_ids
                ],
            )
        }
    )
    ctx = context_factory(
        state=build_state(specs=_disk(95, 500), rented_data=rented_data),
        services=build_services(backend=PerPodBackendClient(active=live)),
        ssh=PerPodSSHClient(running={f"container_{p}" for p in running}),
        rented=False,
    )

    result = await DiskSpaceCheck().run(ctx)

    assert result.passed is exempt
    assert result.event.reason_code == Msg.TOO_FULL.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"default_job_owner_by_executor": {"executor-123": "lium"}},
        {"all_filler_containers_by_executor": {"executor-123": ["filler_run-1"]}},
        {"default_job_owner_by_executor": {"executor-123": "miner"}},
    ],
)
async def test_default_job_or_filler_does_not_exempt_a_full_disk(context_factory, fields):
    """Only a live rental is exempt; an idle node with a Default Job or filler still fails."""
    rented_data = RentedExecutorsResponse(executors={}, **fields)
    ctx = context_factory(state=build_state(specs=_disk(95, 500), rented_data=rented_data))

    result = await DiskSpaceCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.TOO_FULL.reason
