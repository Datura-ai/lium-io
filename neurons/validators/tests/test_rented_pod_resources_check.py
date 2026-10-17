"""DAH-4564 — RentedPodResourcesCheck: a host script that `docker update`s a renter's pod onto fewer
cores or a smaller quota fails only under enforcement; a raised oom_score_adj is recorded, never
failed; an inability to measure never fails."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import asyncssh
import pytest
from neurons.validators.src.services.task.checks.rented_pod_resources import RentedPodResourcesCheck
from neurons.validators.src.services.task.messages import RentedPodResourcesMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import RENTAL_CPU_COUNT_LABEL

from tests.helpers import build_state

SETTINGS = "neurons.validators.src.services.task.checks.rented_pod_resources.settings"


def _container(cpuset="", nano=8_000_000_000, oom=500, label="8"):
    return {
        "name": "/pod_1",
        "cpuset": cpuset,
        "nano": nano,
        "oom": oom,
        "pid": 4242,
        "labels": {RENTAL_CPU_COUNT_LABEL: label} if label else None,
    }


def _ssh(container=None, init_oom=500, raise_exc=None):
    async def run(cmd, *args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if cmd.startswith("/usr/bin/docker inspect"):
            out = json.dumps(container) if container else ""
            return SimpleNamespace(exit_status=0 if container else 1, stdout=out, stderr="")
        return SimpleNamespace(exit_status=0, stdout=f"4242 {init_oom}\n", stderr="")

    return SimpleNamespace(run=run)


def _ctx(context_factory, ssh):
    rented = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="1", container_name="pod_1", status="RUNNING")],
            )
        }
    )
    state = build_state(specs={"cpu": {"count": 64}}, rented_data=rented)
    return context_factory(state=state, ssh=ssh)


async def _run(ctx, enforce=False):
    with patch(SETTINGS) as s:
        s.RENTED_POD_RESOURCES_CHECK_ENABLED = True
        s.RENTED_POD_RESOURCES_ENFORCEMENT_ENABLED = enforce
        return await RentedPodResourcesCheck().run(ctx)


def _findings(result):
    return result.event.what_we_saw["pods"][0]["findings"]


@pytest.mark.asyncio
async def test_clean_pod_passes(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container())), enforce=True)
    assert result.passed is True
    assert result.event.reason_code == Msg.RESOURCES_OK.reason
    assert _findings(result) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
async def test_pinned_cpuset_fails_only_when_enforced(context_factory, enforce):
    # executor_protect.sh: 8 cores rented, pinned to 6 of them
    result = await _run(_ctx(context_factory, _ssh(_container(cpuset="2-7"))), enforce=enforce)
    assert result.event.reason_code == Msg.RESOURCES_REDUCED.reason
    assert _findings(result) == ["cpuset_pinned"]
    assert result.passed is (not enforce)
    assert result.updates == ({"rented_pod_resources_passed": False} if enforce else {})


@pytest.mark.asyncio
async def test_cpuset_covering_the_entitlement_is_not_pinned(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container(cpuset="0-7"))), enforce=True)
    assert result.passed is True


@pytest.mark.asyncio
async def test_cpuset_without_label_or_quota_is_judged_against_host_cpus(context_factory):
    result = await _run(
        _ctx(context_factory, _ssh(_container(cpuset="0-31", nano=0, label=None))), enforce=True
    )
    assert _findings(result) == ["cpuset_pinned"]


@pytest.mark.asyncio
async def test_reduced_nano_cpus_fails_under_enforcement(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container(nano=4_000_000_000))), enforce=True)
    assert result.passed is False
    assert _findings(result) == ["cpu_quota_reduced"]


@pytest.mark.asyncio
async def test_raised_oom_score_is_recorded_never_failed(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container(), init_oom=1000)), enforce=True)
    assert result.passed is True
    assert result.updates == {}
    assert _findings(result) == ["oom_priority_raised"]
    assert result.event.what_we_saw["pods"][0]["oom_score_adj_init"] == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh", [_ssh(raise_exc=asyncssh.Error(1, "boom")), _ssh(raise_exc=OSError("down")), _ssh(None)]
)
async def test_inability_to_measure_passes(context_factory, ssh):
    result = await _run(_ctx(context_factory, ssh), enforce=True)
    assert result.passed is True
    assert result.event.reason_code == Msg.RESOURCES_UNKNOWN.reason


@pytest.mark.asyncio
async def test_disabled_by_default_skips(context_factory):
    result = await RentedPodResourcesCheck().run(
        _ctx(context_factory, _ssh(_container(cpuset="2")))
    )
    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
