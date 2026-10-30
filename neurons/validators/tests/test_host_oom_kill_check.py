from __future__ import annotations

from types import SimpleNamespace

import pytest
from services.task.checks.host_oom_kill import HostOomKillCheck, parse_global_oom_kills

POD_ID = "6f1d2c3b4a5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8"
# ticket-0903, 29 Oct: a global OOM killed the renter's python on a 62 GiB host
GLOBAL_POD_KILL = (
    "[ 9000.123456] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=docker-"
    + POD_ID
    + ".scope,"
    "mems_allowed=0,global_oom,task_memcg=/system.slice/docker-"
    + POD_ID
    + ".scope,task=python,pid=48213,uid=0\n"
    "[ 9000.123499] Out of memory: Killed process 48213 (python) total-vm:52346880kB, anon-rss:41000000kB, "
    "file-rss:0kB, shmem-rss:0kB, UID:0 pgtables:81000kB oom_score_adj:0\n"
)
GLOBAL_HOST_KILL = (
    "[ 9500.000001] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,"
    "task_memcg=/user.slice/user-1000.slice/session-3.scope,task=miner-xmr,pid=777,uid=1000\n"
    "[ 9500.000050] Out of memory: Killed process 777 (miner-xmr) total-vm:9000000kB, anon-rss:8388608kB, "
    "file-rss:0kB, shmem-rss:0kB, UID:1000 pgtables:100kB oom_score_adj:0\n"
)
# the container hit its own limit: not a host OOM
MEMCG_KILL = (
    "[ 9600.000000] oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=docker-"
    + POD_ID
    + ".scope,"
    "mems_allowed=0,oom_memcg=/system.slice/docker-"
    + POD_ID
    + ".scope,task_memcg=/system.slice/docker-"
    + POD_ID
    + ".scope,task=python,pid=50000,uid=0\n"
)


def test_parser_keeps_global_kills_with_container_size_and_age():
    kills = parse_global_oom_kills(
        "9600.50 12345.00\n" + GLOBAL_POD_KILL + GLOBAL_HOST_KILL + MEMCG_KILL
    )
    assert kills == [
        {"seconds_ago": 600, "task": "python", "container_id": POD_ID, "anon_rss_kb": 41000000},
        {"seconds_ago": 100, "task": "miner-xmr", "container_id": None, "anon_rss_kb": 8388608},
    ]


def test_parser_reads_cgroupfs_paths_and_survives_junk():
    line = GLOBAL_POD_KILL.replace("/system.slice/docker-" + POD_ID + ".scope", "/docker/" + POD_ID)
    assert parse_global_oom_kills("9001 1\n" + line)[0]["container_id"] == POD_ID
    assert parse_global_oom_kills("9001 1\nnot a kill line\n") == []


class FakeShell:
    def __init__(self, dmesg: str, name: str = "", inspect_status: int = 0):
        self.outputs = {"cat /proc/uptime": (dmesg, 0), "docker inspect": (name, inspect_status)}
        self.commands: list[str] = []

    async def run(self, command: str, check: bool = False):
        self.commands.append(command)
        out, status = next(v for k, v in self.outputs.items() if k in command)
        return SimpleNamespace(stdout=out, exit_status=status)


async def _run(context_factory, shell):
    ctx = context_factory(ssh=shell)
    return await HostOomKillCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_rental_pod_kill_names_only_the_container(context_factory):
    shell = FakeShell(
        "9600.5 1\n" + GLOBAL_POD_KILL, name="/pod_0b6e0f4e-1111-2222-3333-444455556666\n"
    )
    result = await _run(context_factory, shell)
    assert result.passed
    assert result.updates["default_extra"]["host_oom_kill"] == {
        "kind": "rental_pod",
        "container_id": POD_ID[:12],
        "container_name": "pod_0b6e0f4e-1111-2222-3333-444455556666",
        "task": None,  # the renter's process name never leaves
        "memory_gib": 39.1,
        "minutes_ago": 10,
    }
    assert result.event.reason_code == "HOST_OOM_KILL"


@pytest.mark.asyncio
async def test_a_host_process_kill_names_the_process(context_factory):
    result = await _run(context_factory, FakeShell("9600 1\n" + GLOBAL_POD_KILL + GLOBAL_HOST_KILL))
    summary = result.updates["default_extra"]["host_oom_kill"]
    assert (summary["kind"], summary["task"], summary["container_id"], summary["memory_gib"]) == (
        "host_process",
        "miner-xmr",
        None,
        8.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "status", "kind", "task"),
    [
        ("/executor-executor-1", 0, "lium_container", "python"),
        ("/my-own-db", 0, "other_container", "python"),
        ("", 1, "removed_container", None),
    ],
)
async def test_other_containers(context_factory, name, status, kind, task):
    summary = (
        await _run(context_factory, FakeShell("9600 1\n" + GLOBAL_POD_KILL, name, status))
    ).updates["default_extra"]["host_oom_kill"]
    assert (summary["kind"], summary["task"]) == (kind, task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dmesg", ["200000 1\n" + GLOBAL_POD_KILL, "9600 1\n" + MEMCG_KILL, "9600 1\n"]
)
async def test_no_recent_global_kill_leaves_the_context_alone(context_factory, dmesg):
    result = await _run(context_factory, FakeShell(dmesg))
    assert result.passed and not result.updates
    assert result.event.reason_code == "HOST_OOM_KILL_NONE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dmesg", ["9600 1\nDMESG_UNREADABLE\n", "not uptime\n" + GLOBAL_POD_KILL, ""]
)
async def test_an_unreadable_kernel_log_is_unread_not_quiet(context_factory, dmesg):
    result = await _run(context_factory, FakeShell(dmesg))
    assert result.passed and not result.updates
    assert result.event.reason_code == "HOST_OOM_KILL_UNREAD"


@pytest.mark.asyncio
async def test_an_ssh_error_never_fails_the_node(context_factory):
    class Broken:
        async def run(self, command, check=False):
            raise OSError("connection lost")

    result = await _run(context_factory, Broken())
    assert result.passed and not result.updates
    assert result.event.reason_code == "HOST_OOM_KILL_UNREAD"
