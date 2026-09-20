"""E-187 (DAH-3706 family): a filler never starts on GPUs a RUNNING customer pod holds — the validator's half.

The platform judges the launch against its rows; this is the host-truth check right before the
filler's `docker run`: `docker inspect` every RUNNING `pod_*` container for the GPUs it was given
(HostConfig.DeviceRequests) and refuse the filler when the sets intersect — BEFORE the create's
container sweep, so the customer's container is never the thing removed to make room. The refusal
is a FailedContainerRequest at step `filler_live_pod_guard` (the platform keys on that step) and the
typed event FILLER_START_REFUSED_LIVE_POD (executor uuid, pod names, gpu overlap).

Failing-first: the create tests read only pre-existing names — on main the filler is CREATED beside
the pod. The parser / overlap unit tests import the fix's module inside the test body.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated, FailedContainerRequest, WorkloadKind
from services.docker_service import DockerService

from tests.test_deploy_optimizations import (
    _created_run_spec,
    _patch_happy,
    _payload,
    _run,
    _ssh_client,
    _ssh_run_cmds,
)


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


EVENT = "FILLER_START_REFUSED_LIVE_POD"
GUARD_STEP = "filler_live_pod_guard"

# What `docker inspect --format '{{.Name}}\t{{json .HostConfig.DeviceRequests}}'` prints per container.
POD_ON_0_1 = '/pod_cust1\t[{"Driver":"nvidia","Count":0,"DeviceIDs":["GPU-0","GPU-1"],"Capabilities":[["gpu"]],"Options":{}}]'
POD_WHOLE_HOST = '/pod_whole\t[{"Driver":"nvidia","Count":-1,"DeviceIDs":null,"Capabilities":[["gpu"]],"Options":{}}]'


def _host_with_running_pods(*inspect_lines: str):
    """An ssh client whose live-pod listing answers with these inspect lines (empty stdout otherwise)."""
    client = _ssh_client()

    def _side(cmd, *args, **kwargs):
        result = AsyncMock()
        result.exit_status = 0
        result.stderr = ""
        result.stdout = (
            "\n".join(inspect_lines) + "\n" if "DeviceRequests" in cmd and inspect_lines else ""
        )
        return result

    client.run = AsyncMock(side_effect=_side)
    return client


def _filler_payload(gpu_uuids: list[str]):
    return _payload(
        workload_kind=WorkloadKind.FILLER, gpu_uuids=gpu_uuids, active_container_names=[]
    )


def _events(caplog) -> list[dict]:
    return [
        record.msg.extra
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event") == EVENT
    ]


def _live_pod_listing_cmds(ssh_client) -> list[str]:
    return [cmd for cmd in _ssh_run_cmds(ssh_client) if "DeviceRequests" in cmd]


# ---------------------------------------------------------------------------------------------------
# create_container
# ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filler_is_refused_when_a_running_pod_holds_one_of_its_gpus(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)
    payload = _filler_payload(["GPU-1", "GPU-2"])

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP
    assert result.pod_id == payload.pod_id and result.workload_kind == WorkloadKind.FILLER
    assert "pod_cust1" in (result.detail or "") and "GPU-1" in (result.detail or "")
    assert svc.rental_docker_client_factory.client.run_specs == [], "no docker run"
    # the customer's container is never swept to make room for a filler
    svc.clean_existing_containers.assert_not_awaited()
    [event] = _events(caplog)
    assert event["executor_id"] == payload.executor_id
    assert event["filler_pod_id"] == payload.pod_id
    assert event["pod_containers"] == ["pod_cust1"]
    assert event["gpu_overlap"] == ["GPU-1"]


@pytest.mark.asyncio
async def test_filler_starts_on_the_gpus_no_running_pod_holds(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)
    payload = _filler_payload([f"GPU-{index}" for index in range(2, 8)])

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert _created_run_spec(svc) is not None
    assert _events(caplog) == []
    assert len(_live_pod_listing_cmds(ssh)) == 1, "the guard read the host exactly once"


@pytest.mark.asyncio
async def test_a_whole_host_pod_refuses_every_filler(svc, monkeypatch):
    ssh = _host_with_running_pods(POD_WHOLE_HOST)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _filler_payload(["GPU-5"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP


@pytest.mark.asyncio
async def test_a_whole_node_filler_is_refused_by_any_running_pod(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload([]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP
    [event] = _events(caplog)
    assert event["gpu_overlap"] == ["GPU-0", "GPU-1"]


@pytest.mark.asyncio
async def test_a_host_with_no_running_pod_lets_the_filler_through(svc, monkeypatch):
    ssh = _host_with_running_pods()
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _filler_payload(["GPU-0"]))

    assert isinstance(result, ContainerCreated), result


@pytest.mark.asyncio
async def test_a_customer_create_never_runs_the_filler_guard(svc, monkeypatch):
    # The rent path has its own rule (lium-io#1417 removes every filler_*); a pod create must not be
    # refused, or even slowed, by a listing meant for fillers.
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _payload(gpu_uuids=["GPU-1"]))

    assert isinstance(result, ContainerCreated), result
    assert _live_pod_listing_cmds(ssh) == []


@pytest.mark.asyncio
async def test_a_listing_that_cannot_be_read_refuses_the_filler(svc, monkeypatch, caplog):
    # Fail closed: a filler is revenue for Lium, a pod is a paying customer. Unknown host state means
    # no filler this cycle, not a filler beside a possibly live pod.
    ssh = _ssh_client()

    def _side(cmd, *args, **kwargs):
        result = AsyncMock()
        result.exit_status = 1 if "DeviceRequests" in cmd else 0
        result.stdout = ""
        result.stderr = "Cannot connect to the Docker daemon" if "DeviceRequests" in cmd else ""
        return result

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _filler_payload(["GPU-0"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP
    assert _events(caplog) == [], "an unreadable host is not an overlap event"


# ---------------------------------------------------------------------------------------------------
# parser / overlap (pure)
# ---------------------------------------------------------------------------------------------------


def test_parse_live_pod_gpu_sets_reads_pinned_whole_host_and_skips_non_pods():
    from services.filler_live_pod_guard import parse_live_pod_gpu_sets

    listing = "\n".join(
        [
            POD_ON_0_1,
            POD_WHOLE_HOST,
            '/filler_abc\t[{"Driver":"nvidia","Count":0,"DeviceIDs":["GPU-7"],"Capabilities":[["gpu"]]}]',
            "/container_health_check\t[]",
            "/pod_nogpu\tnull",
            "",
        ]
    )

    pods = parse_live_pod_gpu_sets(listing)

    assert pods == {"pod_cust1": frozenset({"GPU-0", "GPU-1"}), "pod_whole": None}


def test_parse_live_pod_gpu_sets_tolerates_a_malformed_line():
    from services.filler_live_pod_guard import parse_live_pod_gpu_sets

    assert parse_live_pod_gpu_sets("/pod_broken\tnot-json\n" + POD_ON_0_1) == {
        "pod_broken": None,  # unreadable device set = treat as whole host (fail closed)
        "pod_cust1": frozenset({"GPU-0", "GPU-1"}),
    }


def test_find_live_pod_gpu_overlap_pinned_sets():
    from services.filler_live_pod_guard import find_live_pod_gpu_overlap

    pods = {"pod_a": frozenset({"GPU-0", "GPU-1"}), "pod_b": frozenset({"GPU-4"})}
    assert find_live_pod_gpu_overlap(["GPU-2", "GPU-3"], pods) is None
    overlap = find_live_pod_gpu_overlap(["GPU-1", "GPU-4"], pods)
    assert overlap is not None
    assert overlap.pod_containers == ["pod_a", "pod_b"]
    assert overlap.gpu_overlap == ["GPU-1", "GPU-4"]


def test_find_live_pod_gpu_overlap_whole_host_on_either_side():
    from services.filler_live_pod_guard import find_live_pod_gpu_overlap

    assert find_live_pod_gpu_overlap(["GPU-5"], {"pod_w": None}).pod_containers == ["pod_w"]
    whole_filler = find_live_pod_gpu_overlap([], {"pod_a": frozenset({"GPU-0"})})
    assert whole_filler is not None and whole_filler.gpu_overlap == ["GPU-0"]
    assert find_live_pod_gpu_overlap([], {}) is None
