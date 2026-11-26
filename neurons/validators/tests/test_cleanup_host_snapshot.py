"""DAH-7109: the stale-container cleanup's always-run reads come from one SSH call until the host changes."""

import json
import stat
import subprocess
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.docker_utils import DockerCommand
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.container_cleanup import (
    FREE_DISK_UNDER_DOCKER_ROOT,
    HOST_SNAPSHOT,
    LIST_CREATED_CONTAINERS,
    LIST_VOLUME_NAMES,
    ContainerCleanup,
    parse_host_snapshot,
)

NOW = 1_800_000_000
LIVE_POD = "pod_00000000-0000-4000-8000-000000000001"
STRAY = "00000000-0000-4000-8000-0000000000a1"
CACHE = "dphn_cache_hf_x"
GB = 1024**3


def _created_at(age_min: float) -> str:
    return datetime.fromtimestamp(NOW - age_min * 60, UTC).strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


class Host:
    """The executor: answers the snapshot from its sections and records every other command."""

    def __init__(self, rental=(), dangling=(), created=(), vol=(), filler=(), rc=None, free_gb=500):
        self.sections = {
            "RENTAL": [json.dumps(row) for row in rental],
            "DANGLING": list(dangling),
            "CREATED": list(created),
            "VOL": list(vol),
            "FILLER": list(filler),
        }
        self.rc = rc or {}
        self.free_gb = free_gb
        self.commands: list[str] = []

    def _snapshot(self) -> str:
        lines = [f"NOW\t{NOW}"]
        for tag, rows in self.sections.items():
            if not self.rc.get(tag):
                lines += [f"{tag}\t{row}" for row in rows]
            lines.append(f"{tag}_RC\t{self.rc.get(tag, 0)}")
        return "\n".join(lines)

    async def run(self, command, *args, **kwargs):
        self.commands.append(command)
        if command == HOST_SNAPSHOT:
            return SimpleNamespace(exit_status=0, stdout=self._snapshot(), stderr="")
        reads = parse_host_snapshot(self._snapshot())  # a read sent on its own gets the same answer
        if command in reads:
            return reads[command]
        if command == FREE_DISK_UNDER_DOCKER_ROOT:
            avail = int(self.free_gb * GB)
            stdout = f"Filesystem 1B-blocks Used Available Capacity\n/dev/sda1 1 1 {avail} 1% /hostfs"
            return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")


def _ctx(host: Host):
    from tests.helpers import build_services, build_state, make_context

    redis = AsyncMock()
    redis.sadd.return_value = 0  # seen before: no first-sight grace
    redis.hgetall.return_value = {}
    rented = RentedExecutorsResponse(
        executors={
            "another-node": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=[RentedPod(pod_id=LIVE_POD.removeprefix("pod_"), container_name=LIVE_POD)],
            )
        },
    )
    return make_context(
        ssh=host,
        services=build_services(container_cleanup=ContainerCleanup(), redis=redis),
        state=build_state(rented_data=rented, gpu_processes=[]),
    )


async def _run(host: Host, hourly: bool = True):
    from services.task.checks.stale_container_cleanup import StaleContainerCleanupCheck

    ctx = _ctx(host)
    check = StaleContainerCleanupCheck()
    if not hourly:
        check._last_sweep_at[ctx.executor.uuid] = time.monotonic()
    result = await check.run(ctx)
    return result, host.commands[1:]  # what ran after the snapshot


@pytest.mark.asyncio
async def test_a_node_with_nothing_to_remove_and_no_cache_makes_one_call_hourly_passes_included():
    host = Host(rental=[[LIVE_POD, LIVE_POD.removeprefix("pod_"), _created_at(600)]], vol=["volume_x"])

    _, after = await _run(host, hourly=True)

    assert host.commands[0] == HOST_SNAPSHOT
    assert after == []


@pytest.mark.asyncio
async def test_a_cache_with_a_running_filler_makes_one_call():
    host = Host(vol=[CACHE], filler=["filler_x"])

    _, after = await _run(host, hourly=False)

    assert after == []


@pytest.mark.asyncio
async def test_a_cache_with_no_running_filler_adds_only_the_one_disk_measurement():
    host = Host(vol=[CACHE])

    _, after = await _run(host, hourly=False)

    assert after == [FREE_DISK_UNDER_DOCKER_ROOT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tag, work",
    [
        ("RENTAL", {"rental": [[f"pod_{STRAY}", STRAY, _created_at(600)]]}),
        ("DANGLING", {"dangling": ["a" * 64]}),
        ("CREATED", {"created": [f"{'b' * 64} lium_capability_check_1_abcdef"]}),
        ("VOL", {"vol": [CACHE]}),
        ("FILLER", {"vol": [CACHE]}),  # a failed filler read counts as a running filler: no reclaim
    ],
)
async def test_a_failed_section_stops_only_its_own_step_and_sends_nothing_to_the_host(tag, work):
    host = Host(**work, rc={tag: 1}, free_gb=1)

    _, after = await _run(host, hourly=False)

    assert after == []


@pytest.mark.asyncio
async def test_age_comes_from_the_snapshot_an_old_candidate_goes_a_fresh_or_unreadable_one_stays():
    old, fresh, unreadable = (f"00000000-0000-4000-8000-0000000000b{i}" for i in (1, 2, 3))
    host = Host(
        rental=[
            [f"pod_{old}", old, _created_at(600)],
            [f"pod_{fresh}", fresh, _created_at(1)],
            [f"pod_{unreadable}", unreadable, "yesterday"],
        ]
    )

    _, after = await _run(host, hourly=False)

    removals = [command for command in after if "docker rm" in command]
    assert any(f"pod_{old}" in command for command in removals)
    assert not any(fresh in command or unreadable in command for command in removals)


@pytest.mark.asyncio
async def test_after_a_removal_every_later_read_goes_to_the_host():
    host = Host(rental=[[f"pod_{STRAY}", STRAY, _created_at(600)]], vol=[CACHE], free_gb=500)

    _, after = await _run(host, hourly=True)

    assert any(f"pod_{STRAY}" in command for command in after)
    assert after.count(DockerCommand.volume_ls_dangling()) == 2  # the prune and the hourly orphan pass
    assert LIST_CREATED_CONTAINERS in after
    assert after.count(LIST_VOLUME_NAMES) == 2  # the hourly sweep and the reclaim
    assert FREE_DISK_UNDER_DOCKER_ROOT in after


def test_a_section_cut_short_is_left_to_the_host_and_a_forged_line_is_ignored():
    stdout = f"NOW\t{NOW}\nDANGLING\tv1\nDANGLING_RC\tx\nVOL\tv2\nVOL_RC\t0\nVOL_RC2\t0\nCREATED\tz"

    answers = parse_host_snapshot(stdout)

    assert set(answers) == {LIST_VOLUME_NAMES}
    assert answers[LIST_VOLUME_NAMES].stdout == "v2\n"


def test_the_snapshot_runs_under_sh_and_reports_each_section_with_its_status(tmp_path):
    # dash or busybox sh, as on an executor: a fake docker prints each section's tagged line from its
    # own --format (the tag before the tab), and the dangling listing fails
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$0.log"\n'
        "for last; do :; done\n"
        'case "$*" in *dangling=true*) exit 1 ;; esac\n'
        'tag=${last%%\\\\t*}\n'
        'printf \'%s\\t%s\\n\' "$tag" row\n'
    )
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    command = HOST_SNAPSHOT.replace("/usr/bin/docker", str(docker))

    out = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, check=True)
    answers = parse_host_snapshot(out.stdout)

    assert answers[DockerCommand.volume_ls_dangling()].exit_status == 1
    assert answers[LIST_VOLUME_NAMES].stdout == "row\n"
    assert answers[LIST_CREATED_CONTAINERS].stdout == "row\n"
    rental = next(a for a in answers.values() if a.stdout.startswith(tuple("0123456789")))
    assert rental.stdout.splitlines()[1] == "row"
    assert int(rental.stdout.splitlines()[0]) > 0  # the host clock
    assert len(answers) == 5
    assert "\\t" in (tmp_path / "docker.log").read_text()  # docker gets the tab escape, not a raw tab
