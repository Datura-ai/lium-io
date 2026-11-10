"""DAH-5951: on a host whose Docker cannot limit a container's disk, a rented pod filling the Docker disk is
paused (never deleted) and resumed once the disk has room; a host with the limit is never touched."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import asyncssh
import pytest
from neurons.validators.src.services.task.checks.rented_pod_disk import (
    PodLayer,
    RentedPodDiskCheck,
    parse_reading,
    plan,
    read_command,
)
from neurons.validators.src.services.task.messages import RentedPodDiskMessages as Msg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import FakeRedis, build_services, build_state

GB = 1024**3
DISK = 1000 * GB


def _ssh(stdout="", raise_exc=None):
    calls = []

    async def run(cmd, *args, **kwargs):
        calls.append(cmd)
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(exit_status=0, stdout=stdout if "inspect" in cmd else "", stderr="")

    return SimpleNamespace(run=run, calls=calls)


def _ctx(context_factory, ssh, supported=False, redis=None):
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
    state = build_state(specs={"storage_limit_supported": supported}, rented_data=rented)
    return context_factory(
        state=state, ssh=ssh, services=build_services(redis=redis or FakeRedis())
    )


def _reading(free_gb, *pods):
    return "\n".join([f"DF {DISK} {free_gb * GB}", *pods])


def test_a_full_disk_pauses_only_a_pod_holding_a_large_share_of_it_volume_included():
    pods = [
        PodLayer("big_layer", "running", 300 * GB, 0),
        PodLayer("big_volume", "running", 1 * GB, 400 * GB),
        PodLayer("small", "running", 20 * GB, 30 * GB),
    ]
    assert plan(DISK, 40 * GB, pods, set()) == (["big_layer", "big_volume"], [])


def test_a_disk_the_host_filled_itself_pauses_nobody():
    assert plan(DISK, 10 * GB, [PodLayer("pod_1", "running", 5 * GB, 50 * GB)], set()) == ([], [])


def test_a_paused_pod_resumes_only_once_a_tenth_of_the_disk_is_free():
    pods = [PodLayer("pod_1", "paused", 300 * GB, 0)]
    assert plan(DISK, 70 * GB, pods, {"pod_1"}) == ([], [])
    assert plan(DISK, 100 * GB, pods, {"pod_1"}) == ([], ["pod_1"])
    assert plan(DISK, 100 * GB, pods, set()) == ([], [])


@pytest.mark.asyncio
async def test_a_host_with_the_disk_limit_is_never_touched(context_factory):
    ssh = _ssh()
    result = await RentedPodDiskCheck().run(_ctx(context_factory, ssh, supported=True))
    assert result.event.reason_code == Msg.LIMITED.reason
    assert ssh.calls == []


@pytest.mark.asyncio
async def test_a_pod_filling_a_no_limit_host_is_paused_not_removed(context_factory):
    ssh = _ssh(_reading(20, f"/pod_1 running {500 * GB} 0"))
    result = await RentedPodDiskCheck().run(_ctx(context_factory, ssh))
    assert result.passed is True
    assert result.event.reason_code == Msg.PAUSED.reason
    assert ssh.calls[-1] == "/usr/bin/docker pause pod_1"
    assert not any("rm" in c.split() for c in ssh.calls)


@pytest.mark.asyncio
async def test_a_room_left_pod_is_resumed_and_an_unreadable_host_passes(context_factory):
    redis = FakeRedis()
    await RentedPodDiskCheck().run(
        _ctx(context_factory, _ssh(_reading(20, f"/pod_1 running {500 * GB} 0")), redis=redis)
    )
    ssh = _ssh(_reading(200, f"/pod_1 paused {500 * GB} 0"))
    result = await RentedPodDiskCheck().run(_ctx(context_factory, ssh, redis=redis))
    assert result.event.reason_code == Msg.RESUMED.reason
    assert ssh.calls[-1] == "/usr/bin/docker unpause pod_1"

    for ssh in (_ssh(""), _ssh(raise_exc=asyncssh.ConnectionLost("gone"))):
        result = await RentedPodDiskCheck().run(_ctx(context_factory, ssh))
        assert (result.passed, result.event.reason_code) == (True, Msg.UNKNOWN.reason)
        assert len(ssh.calls) == 1


def test_the_read_command_reports_the_disk_and_counts_the_local_volume(tmp_path):
    volume = tmp_path / "volumes" / "v1"
    volume.mkdir(parents=True)
    (volume / "data").write_bytes(b"x" * 3 * 1024 * 1024)
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f'[ "$1" = info ] && {{ echo {tmp_path}; exit 0; }}\n'
        f'echo "/pod_1 running 4096 {volume}"\n'
    )
    docker.chmod(0o755)
    out = subprocess.run(
        ["sh", "-c", read_command(["pod_1"]).replace("/usr/bin/docker", str(docker))],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ["PATH"]},
    ).stdout
    size, free, pods = parse_reading(out)
    assert size and free is not None and free <= size
    assert len(pods) == 1 and pods[0].container == "pod_1" and pods[0].size_rw == 4096
    assert pods[0].volume_bytes >= 3 * 1024 * 1024


def test_runs_in_the_validation_pipelines_but_never_in_the_dry_run():
    def has(checks):
        return any(isinstance(c, RentedPodDiskCheck) for c in checks)

    assert has(PipelineFactory.build_checks())
    assert has(PipelineFactory.build_fast_path_checks())
    assert not has(PipelineFactory.build_dry_run_checks())


@pytest.mark.asyncio
async def test_the_backend_is_told_of_each_pause_and_resume_so_a_paused_pod_is_not_billed(
    context_factory,
):
    redis = FakeRedis()
    paused = _ctx(context_factory, _ssh(_reading(20, f"/pod_1 running {500 * GB} 0")), redis=redis)
    still = _ctx(context_factory, _ssh(_reading(70, f"/pod_1 paused {500 * GB} 0")), redis=redis)
    resumed = _ctx(context_factory, _ssh(_reading(200, f"/pod_1 paused {500 * GB} 0")), redis=redis)

    for ctx in (paused, still, resumed):
        await RentedPodDiskCheck().run(ctx)

    def told(ctx):
        return [
            c.kwargs["paused"] for c in ctx.services.backend.report_pod_disk_pause.call_args_list
        ]

    assert (told(paused), told(still), told(resumed)) == ([True], [True], [False])
    assert paused.services.backend.report_pod_disk_pause.call_args.args == ("1",)
    assert (
        paused.services.backend.report_pod_disk_pause.call_args.kwargs["executor_id"]
        == "executor-123"
    )


@pytest.mark.asyncio
async def test_a_pod_paused_for_another_reason_is_never_resumed_or_reported(context_factory):
    for free_gb in (70, 200):
        ssh = _ssh(_reading(free_gb, f"/pod_1 paused {500 * GB} 0"))
        ctx = _ctx(context_factory, ssh)
        result = await RentedPodDiskCheck().run(ctx)
        assert result.event.reason_code == Msg.OK.reason
        assert not any("unpause" in c for c in ssh.calls)
        ctx.services.backend.report_pod_disk_pause.assert_not_called()


@pytest.mark.asyncio
async def test_a_redis_outage_acts_on_nothing(context_factory):
    ssh = _ssh(_reading(20, f"/pod_1 running {500 * GB} 0"))
    result = await RentedPodDiskCheck().run(
        _ctx(context_factory, ssh, redis=FakeRedis(failing=True))
    )
    assert (result.passed, result.event.reason_code) == (True, Msg.UNKNOWN.reason)
    assert not any("pause" in c.split() for c in ssh.calls)


@pytest.mark.asyncio
async def test_the_pause_record_is_saved_before_the_pause_and_a_failed_save_pauses_nothing(
    context_factory,
):
    redis = FakeRedis()
    ssh = _ssh(_reading(20, f"/pod_1 running {500 * GB} 0"))
    real_run = ssh.run

    async def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker pause"):
            assert json.loads(redis.store["rented_pod_disk:paused:executor-123"]) == ["pod_1"]
        return await real_run(cmd, *args, **kwargs)

    ssh.run = run
    await RentedPodDiskCheck().run(_ctx(context_factory, ssh, redis=redis))
    assert ssh.calls[-1] == "/usr/bin/docker pause pod_1"

    redis = FakeRedis()
    redis.fail_next_set_of.add("rented_pod_disk:paused:executor-123")
    ssh = _ssh(_reading(20, f"/pod_1 running {500 * GB} 0"))
    ctx = _ctx(context_factory, ssh, redis=redis)
    result = await RentedPodDiskCheck().run(ctx)
    assert (result.passed, result.event.reason_code) == (True, Msg.UNKNOWN.reason)
    assert not any("pause" in c.split() for c in ssh.calls)
    ctx.services.backend.report_pod_disk_pause.assert_not_called()


@pytest.mark.asyncio
async def test_billing_stops_only_for_a_pod_seen_paused_and_a_pod_seen_running_again_is_reported_resumed(
    context_factory,
):
    key = "rented_pod_disk:paused:executor-123"
    # recorded, but missing from this read: kept, and not reported either way
    redis = FakeRedis()
    redis.store[key] = json.dumps(["pod_1"])
    ctx = _ctx(context_factory, _ssh(_reading(70)), redis=redis)
    await RentedPodDiskCheck().run(ctx)
    ctx.services.backend.report_pod_disk_pause.assert_not_called()
    assert json.loads(redis.store[key]) == ["pod_1"]

    # recorded, seen running again: reported resumed and dropped
    ctx = _ctx(context_factory, _ssh(_reading(70, f"/pod_1 running {500 * GB} 0")), redis=redis)
    await RentedPodDiskCheck().run(ctx)
    assert [
        c.kwargs["paused"] for c in ctx.services.backend.report_pod_disk_pause.call_args_list
    ] == [False]
    assert key not in redis.store


@pytest.mark.asyncio
async def test_a_record_of_a_pod_no_longer_rented_is_cleared(context_factory):
    key = "rented_pod_disk:paused:executor-123"
    redis = FakeRedis()
    redis.store[key] = json.dumps(["old_pod", "pod_1"])
    await RentedPodDiskCheck().run(
        _ctx(context_factory, _ssh(_reading(70, f"/pod_1 paused {500 * GB} 0")), redis=redis)
    )
    assert json.loads(redis.store[key]) == ["pod_1"]


@pytest.mark.asyncio
async def test_a_recorded_pod_seen_running_and_paused_again_is_reported_paused_only(
    context_factory,
):
    key = "rented_pod_disk:paused:executor-123"
    redis = FakeRedis()
    redis.store[key] = json.dumps(["pod_1"])
    ctx = _ctx(context_factory, _ssh(_reading(20, f"/pod_1 running {500 * GB} 0")), redis=redis)
    await RentedPodDiskCheck().run(ctx)
    assert [
        c.kwargs["paused"] for c in ctx.services.backend.report_pod_disk_pause.call_args_list
    ] == [True]
    assert json.loads(redis.store[key]) == ["pod_1"]
