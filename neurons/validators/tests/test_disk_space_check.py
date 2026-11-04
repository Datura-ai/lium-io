"""DAH-4961: the listing's disk floor as a validation failure of an idle node."""

from __future__ import annotations

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import FRESH_SIZING_HEADROOM_GB, MAX_DISK_UTILIZATION_PERCENT
from services.task.checks.disk_space import KB_PER_GB, DiskSpaceCheck, SplitDiskCheck
from services.task.checks.rented_machine import TenantEnforcementCheck
from services.task.pipeline_factory import PipelineFactory
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


def _split_rental(*pod_gpu_counts: int | None) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=[
                    RentedPod(pod_id=f"pod-{i}", container_name=f"c{i}", rented_ports=[], gpu_count=n)
                    for i, n in enumerate(pod_gpu_counts)
                ],
            )
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gpu_count, min_count, pods, free_gb, free_splits, listable",
    [
        # 2 GPUs, 1 rented: under the whole-host floor, but enough for the one free split
        (2, 1, (1,), settings.MIN_FREE_DISK_GB * 0.5 + FRESH_SIZING_HEADROOM_GB, 1, 1),
        (2, 1, (1,), settings.MIN_FREE_DISK_GB * 0.4, 1, 0),
        # 8 GPUs in 2-GPU splits, 2 rented: 3 free splits, room for 2 of them
        (8, 2, (2,), 2 * (settings.MIN_FREE_DISK_GB / 4 + FRESH_SIZING_HEADROOM_GB), 3, 2),
    ],
)
async def test_split_rented_node_sizes_disk_per_free_split(
    context_factory, gpu_count, min_count, pods, free_gb, free_splits, listable
):
    state = build_state(
        specs=_disk(50, free_gb),
        rented_data=_split_rental(*pods),
        gpu_count=gpu_count,
        supports_gpu_splitting=True,
        gpu_splitting_min_count=min_count,
    )
    ctx = context_factory(state=state, rented=True)

    result = await DiskSpaceCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FREE_TOO_LOW.reason
    split = result.updates["state"].specs["split_disk"]
    assert split["free_split_count"] == free_splits
    assert split["listable_split_count"] == listable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supports_splitting, min_count, pods",
    [
        (False, None, (1,)),  # unsplit node: the whole-host floor only
        (True, 1, (None,)),  # a pod with no GPU count: free GPUs unknown
        (True, 1, (2,)),  # fully rented: no free split
    ],
)
async def test_unsplit_or_unknown_node_keeps_the_whole_host_floor(
    context_factory, supports_splitting, min_count, pods
):
    state = build_state(
        specs=_disk(50, settings.MIN_FREE_DISK_GB - 1),
        rented_data=_split_rental(*pods),
        gpu_count=2,
        supports_gpu_splitting=supports_splitting,
        gpu_splitting_min_count=min_count,
    )
    ctx = context_factory(state=state, rented=True)

    result = await DiskSpaceCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FREE_TOO_LOW.reason
    assert "split_disk" not in result.updates.get("state", state).specs


@pytest.mark.asyncio
async def test_idle_split_node_still_fails_the_whole_host_floor(context_factory):
    state = build_state(
        specs=_disk(50, settings.MIN_FREE_DISK_GB - 1),
        gpu_count=2,
        supports_gpu_splitting=True,
        gpu_splitting_min_count=1,
    )

    result = await DiskSpaceCheck().run(context_factory(state=state))

    assert result.passed is False
    assert "updates" not in result.model_fields_set


@pytest.mark.parametrize(
    "build",
    [PipelineFactory.build_checks, PipelineFactory.build_fast_path_checks, PipelineFactory.build_dry_run_checks],
)
def test_split_disk_is_sized_before_the_rented_halt_in_every_pipeline(build):
    # TenantEnforcementCheck halts a rented node's cycle, so DiskSpaceCheck never sees a normal partly rented node
    ids = [check.check_id for check in build()]

    assert ids.index(SplitDiskCheck.check_id) < ids.index(TenantEnforcementCheck.check_id)


def _split_state(free_gb: float, utilization: float = 50, **kwargs):
    defaults = dict(gpu_count=8, supports_gpu_splitting=True, gpu_splitting_min_count=2)
    rented_data = _split_rental(*kwargs.pop("pods", (2,)))
    return build_state(specs=_disk(utilization, free_gb), rented_data=rented_data, **{**defaults, **kwargs})


@pytest.mark.asyncio
async def test_split_disk_check_publishes_the_sizing_on_a_partly_rented_node(context_factory):
    # 8 GPUs in 2-GPU splits, 2 rented: 3 free splits, room for 2 of them
    ctx = context_factory(
        state=_split_state(2 * (settings.MIN_FREE_DISK_GB / 4 + FRESH_SIZING_HEADROOM_GB)), rented=True
    )

    result = await SplitDiskCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].specs["split_disk"] == {
        "free_split_count": 3,
        "listable_split_count": 2,
        "listable_gpu_count": 4,
        "required_free_gb_per_split": round(settings.MIN_FREE_DISK_GB / 4 + FRESH_SIZING_HEADROOM_GB, 1),
    }


@pytest.mark.asyncio
async def test_split_disk_check_writes_nothing_on_an_idle_node(context_factory):
    state = build_state(specs=_disk(50, 1), gpu_count=8, supports_gpu_splitting=True, gpu_splitting_min_count=2)

    result = await SplitDiskCheck().run(context_factory(state=state))

    assert result.passed is True
    assert "updates" not in result.model_fields_set


@pytest.mark.asyncio
async def test_leftover_gpus_ride_with_the_last_listable_split(context_factory):
    # 8 GPUs in 2-GPU splits, 1 rented: 7 free GPUs = 3 splits + 1 leftover, all with room
    free_gb = 3 * (settings.MIN_FREE_DISK_GB / 4 + FRESH_SIZING_HEADROOM_GB)
    ctx = context_factory(state=_split_state(free_gb, pods=(1,)), rented=True)

    split = (await SplitDiskCheck().run(ctx)).updates["state"].specs["split_disk"]

    assert (split["listable_split_count"], split["listable_gpu_count"]) == (3, 7)


@pytest.mark.asyncio
@pytest.mark.parametrize("check", [SplitDiskCheck, DiskSpaceCheck])
async def test_a_zero_floor_asks_no_disk_of_a_split(context_factory, monkeypatch, check):
    # MIN_FREE_DISK_GB=0 is accepted; an over-90% rented split node must not divide by zero
    monkeypatch.setattr(settings, "MIN_FREE_DISK_GB", 0)
    ctx = context_factory(state=_split_state(1, utilization=MAX_DISK_UTILIZATION_PERCENT + 5), rented=True)

    result = await check().run(ctx)

    assert result.passed is True
    split = result.updates["state"].specs["split_disk"]
    assert (split["listable_split_count"], split["required_free_gb_per_split"]) == (3, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("free_gb, listable", [(6.25, 0), (16.25, 1)])
async def test_a_free_split_also_needs_the_headroom_fresh_sizing_keeps(context_factory, monkeypatch, free_gb, listable):
    # a 1-of-8 split needs its 6.25 GB share of a 50 GB floor plus the 10 GB a rental's sizing leaves free;
    # with the share alone the rental got the 1 GB floor volume
    monkeypatch.setattr(settings, "MIN_FREE_DISK_GB", 50)
    ctx = context_factory(state=_split_state(free_gb, gpu_splitting_min_count=1, pods=(1, 1, 1, 1, 1, 1)), rented=True)

    split = (await SplitDiskCheck().run(ctx)).updates["state"].specs["split_disk"]

    assert (split["free_split_count"], split["listable_split_count"]) == (2, listable)
    assert split["required_free_gb_per_split"] == round(16.25, 1)
