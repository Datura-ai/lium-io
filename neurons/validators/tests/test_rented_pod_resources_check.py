"""DAH-4564 — RentedPodResourcesCheck: a host script that `docker update`s a renter's pod onto fewer
cores or a smaller quota fails; a raised oom_score_adj is recorded, never
failed; an inability to measure never fails."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import ANY, patch

import asyncssh
import pytest
from neurons.validators.src.services.task.checks.rented_pod_resources import (
    _INIT_READ_SCRIPT,
    RentedPodResourcesCheck,
    pod_verdict_updates,
)
from neurons.validators.src.services.task.messages import RentedPodResourcesMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import RENTAL_CPU_COUNT_LABEL

from tests.helpers import build_state

SETTINGS = "neurons.validators.src.services.task.checks.rented_pod_resources.settings"


def _container(cpuset="", nano=8_000_000_000, oom=500, label="8", quota=0, period=0):
    return {
        "name": "/pod_1",
        "cpuset": cpuset,
        "nano": nano,
        "quota": quota,
        "period": period,
        "oom": oom,
        "pid": 4242,
        "labels": {RENTAL_CPU_COUNT_LABEL: label} if label else None,
    }


def _ssh(
    container=None, init_oom=500, raise_exc=None, cgroup_cpuset="", cpu_max="", proc_err="", runtime_fault="1"
):
    async def run(cmd, *args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if cmd.startswith("/usr/bin/docker inspect"):
            out = json.dumps(container) if container else ""
            return SimpleNamespace(exit_status=0 if container else 1, stdout=out, stderr="")
        return SimpleNamespace(
            exit_status=0,
            stdout=f"4242|{init_oom}|{cgroup_cpuset}|{cpu_max}|{runtime_fault}|{proc_err}\n",
            stderr="",
        )

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


async def _run(ctx):
    with patch(SETTINGS) as s:
        s.RENTED_POD_RESOURCES_CHECK_ENABLED = True
        s.DRY_RUN = False
        return await RentedPodResourcesCheck().run(ctx)


def _findings(result):
    return result.event.what_we_saw["pods"][0]["findings"]


@pytest.mark.asyncio
async def test_clean_pod_passes(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container())))
    assert result.passed is True
    assert result.event.reason_code == Msg.RESOURCES_OK.reason
    assert _findings(result) == []


@pytest.mark.asyncio
async def test_pinned_cpuset_fails_and_zeroes_the_score(context_factory):
    # executor_protect.sh: 8 cores rented, pinned to 6 of them
    result = await _run(_ctx(context_factory, _ssh(_container(cpuset="2-7"))))
    assert result.event.reason_code == Msg.RESOURCES_REDUCED.reason
    assert _findings(result) == ["cpuset_pinned"]
    assert result.passed is False
    assert result.updates == {"rented_pod_resources_passed": False, "state": ANY}
    assert result.updates["state"].rented_pod_faults == {"1": []}


@pytest.mark.asyncio
async def test_cpuset_covering_the_entitlement_is_not_pinned(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container(cpuset="0-7"))))
    assert result.passed is True


@pytest.mark.asyncio
async def test_cpuset_without_label_or_quota_is_judged_against_host_cpus(context_factory):
    result = await _run(
        _ctx(context_factory, _ssh(_container(cpuset="0-31", nano=0, label=None)))
    )
    assert _findings(result) == ["cpuset_pinned"]


@pytest.mark.asyncio
async def test_reduced_nano_cpus_fails(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container(nano=4_000_000_000))))
    assert result.passed is False
    assert _findings(result) == ["cpu_quota_reduced"]


@pytest.mark.asyncio
async def test_raised_oom_score_is_recorded_never_failed(context_factory):
    result = await _run(_ctx(context_factory, _ssh(_container(), init_oom=1000)))
    assert result.passed is True
    assert result.updates["state"].rented_pod_faults == {"1": []}
    assert _findings(result) == ["oom_priority_raised"]
    assert result.event.what_we_saw["pods"][0]["oom_score_adj_init"] == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh", [_ssh(raise_exc=asyncssh.Error(1, "boom")), _ssh(raise_exc=OSError("down")), _ssh(None)]
)
async def test_inability_to_measure_passes(context_factory, ssh):
    result = await _run(_ctx(context_factory, ssh))
    assert result.passed is True
    assert result.event.reason_code == Msg.RESOURCES_UNKNOWN.reason


@pytest.mark.asyncio
async def test_disabled_skips(context_factory):
    ctx = _ctx(context_factory, _ssh(_container(cpuset="2")))
    with patch(SETTINGS) as s:
        s.RENTED_POD_RESOURCES_CHECK_ENABLED = False
        result = await RentedPodResourcesCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason


def test_check_on_by_default():
    from core.config import Settings

    assert Settings.model_fields["RENTED_POD_RESOURCES_CHECK_ENABLED"].default is True


# 100 advertised cores; the owner's floor is 95% (MIN_RENTAL_CPU_SHARE) in both cpuset and quota.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "container, findings",
    [
        (_container(cpuset="0-99", nano=0, label="100"), []),
        (_container(cpuset="0-95", nano=0, label="100"), []),
        (_container(cpuset="0-94", nano=0, label="100"), []),
        (_container(cpuset="0-93", nano=0, label="100"), ["cpuset_pinned"]),
        (_container(nano=100_000_000_000, label="100"), []),
        (_container(nano=96_000_000_000, label="100"), []),
        (_container(nano=95_000_000_000, label="100"), []),
        (_container(nano=94_000_000_000, label="100"), ["cpu_quota_reduced"]),
        (_container(nano=0, quota=9_600_000, period=100_000, label="100"), []),
        (_container(nano=0, quota=9_500_000, period=100_000, label="100"), []),
        (_container(nano=0, quota=9_400_000, period=100_000, label="100"), ["cpu_quota_reduced"]),
        (_container(nano=0, quota=4_700_000, label="100"), ["cpu_quota_reduced"]),  # default period
        (_container(nano=0, quota=-1, period=100_000, label="100"), []),  # unlimited
        (_container(nano=0, quota=0, label="100"), []),  # unlimited
        (
            _container(cpuset="0-93", nano=94_000_000_000, label="100"),
            ["cpuset_pinned", "cpu_quota_reduced"],
        ),
    ],
)
async def test_95_percent_floor(context_factory, container, findings):
    result = await _run(_ctx(context_factory, _ssh(container)))
    assert _findings(result) == findings
    assert result.passed is (not findings)


# A host that writes the pod's cgroup directly leaves HostConfig untouched; the init PID's cgroup shows it.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cgroup_cpuset, cpu_max, findings",
    [
        ("0-63", ";800000 100000;max 100000", []),  # the pod's own --cpus 8, host CPUs all usable
        ("2-7", ";800000 100000", ["cpuset_pinned"]),  # cpuset.cpus written on the pod's cgroup
        (
            "0-63",
            ";800000 100000;400000 100000",
            ["cpu_quota_reduced"],
        ),  # a parent slice capped at 4
        ("", "", []),  # cgroup v1 or unreadable: nothing seen, nothing judged
    ],
)
async def test_cgroup_written_directly_is_seen(context_factory, cgroup_cpuset, cpu_max, findings):
    ssh = _ssh(_container(), cgroup_cpuset=cgroup_cpuset, cpu_max=cpu_max)
    result = await _run(_ctx(context_factory, ssh))
    assert _findings(result) == findings
    assert result.passed is (not findings)


@pytest.mark.asyncio
async def test_memory_guard_coverage_is_counted_in_the_result(context_factory):
    ssh = _ssh(_container())
    inner_run = ssh.run

    async def run(cmd, *args, **kwargs):
        if "MemoryMin" in cmd:
            return SimpleNamespace(exit_status=0, stdout="memory_min_skipped no_systemd\n", stderr="")
        return await inner_run(cmd, *args, **kwargs)

    result = await _run(_ctx(context_factory, SimpleNamespace(run=run)))

    assert result.event.what_we_saw["memory_guard"] == {"set": 0, "failed": ["memory_min no_systemd"]}


@pytest.mark.asyncio
async def test_dry_run_validator_writes_no_memory_guard_on_the_host(context_factory):
    ctx = _ctx(context_factory, _ssh(_container()))
    with patch(SETTINGS) as s, patch(
        "neurons.validators.src.services.task.checks.rented_pod_resources.apply_host_memory_guard"
    ) as guard:
        s.RENTED_POD_RESOURCES_CHECK_ENABLED = True
        s.DRY_RUN = True
        result = await RentedPodResourcesCheck().run(ctx)

    guard.assert_not_called()
    assert result.event.what_we_saw["memory_guard"] == {"set": 0, "failed": []}


@pytest.mark.asyncio
async def test_dead_proc_fuse_mount_is_reported_never_failed(context_factory):
    err = "cat: /proc/4242/root/proc/sys/kernel/hostname: Transport endpoint is not connected"
    result = await _run(_ctx(context_factory, _ssh(_container(), proc_err=err)))
    assert result.passed is True
    # reported per pod so the platform can tell the renter
    assert result.updates["state"].rented_pod_faults == {"1": [Msg.PROC_FUSE_DISCONNECTED.reason]}
    assert result.event.reason_code == Msg.PROC_FUSE_DISCONNECTED.reason
    assert _findings(result) == ["proc_fuse_disconnected"]


@pytest.mark.asyncio
async def test_dead_proc_under_a_sysbox_fs_older_than_the_pod_is_not_a_host_fault(context_factory):
    # the renter's own FUSE mount stacked on /proc/sys and killed: sysbox-fs never restarted under the pod
    err = "Transport endpoint is not connected"
    result = await _run(_ctx(context_factory, _ssh(_container(), proc_err=err, runtime_fault="0")))
    assert result.event.reason_code == Msg.RESOURCES_OK.reason
    assert result.updates["state"].rented_pod_faults == {"1": []}


@pytest.mark.asyncio
async def test_scored_finding_wins_over_dead_proc_fuse_mount(context_factory):
    err = "Transport endpoint is not connected"
    result = await _run(_ctx(context_factory, _ssh(_container(cpuset="2-7"), proc_err=err)))
    assert result.passed is False
    assert result.event.reason_code == Msg.RESOURCES_REDUCED.reason


def test_a_later_checks_verdict_keeps_the_earlier_checks_codes(context_factory):
    ctx = _ctx(context_factory, ssh=None)
    first = pod_verdict_updates(ctx, {"pod_1": [Msg.PROC_FUSE_DISCONNECTED.reason]})["state"]
    ctx = context_factory(state=first, ssh=None)
    # another check finds the pod clean, and a container not on the rented list is not judged
    second = pod_verdict_updates(ctx, {"pod_1": [], "pod_x": ["CODE"]})["state"]
    assert second.rented_pod_faults == {"1": [Msg.PROC_FUSE_DISCONNECTED.reason]}
    assert pod_verdict_updates(context_factory(state=build_state(), ssh=None), {"pod_1": []}) == {}


def test_the_newest_sysbox_fs_is_timed_against_the_pod():
    # an old sysbox-fs left running beside a restarted one would read as older than the pod and hide the restart
    assert "pgrep -xn sysbox-fs" in _INIT_READ_SCRIPT
    # the script runs in the executor container, which has no sysbox-fs binary: the host's is seen through PID 1
    assert "[ -x /proc/1/root/usr/bin/sysbox-fs ]" in _INIT_READ_SCRIPT
    assert "command -v sysbox-fs" not in _INIT_READ_SCRIPT



def _ssh_without_init_line(container):
    inspect = _ssh(container)

    async def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker inspect"):
            return await inspect.run(cmd)
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    return SimpleNamespace(run=run)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh",
    [
        _ssh(_container(), proc_err="cat: /proc/4242/root/proc/sys/kernel/hostname: Permission denied"),
        _ssh_without_init_line(_container()),
    ],
)
async def test_a_failed_proc_read_sends_no_verdict_so_an_earlier_fault_stands(context_factory, ssh):
    # another error, or no line for the pod's init, says nothing about the pod: it must not be sent as [] (clean),
    # which the platform reads as "the fault is gone"
    result = await _run(_ctx(context_factory, ssh))
    assert result.passed is True
    assert "state" not in result.updates


def test_dead_proc_fuse_text_claims_no_cause_and_asks_no_runtime_restart():
    """the read sees only ENOTCONN, never why: it must not tell a provider sysbox-fs restarted (a provider's
    sysbox-fs was still the 13/14 Nov process after our 17 Nov notice said it restarted), nor to restart sysbox"""
    text = Msg.PROC_FUSE_DISCONNECTED.remediation
    assert "stopped or restarted" not in text
    assert "restart it" not in text
    assert "RENTED_POD_OWNERSHIP_RUNTIME_RESTARTED" in text
