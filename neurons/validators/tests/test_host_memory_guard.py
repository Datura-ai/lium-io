"""DAH-6379: the host memory guard sets memory.oom.group on the pod's cgroup and MemoryMin on the host
services, in a script that parses, and never fails the caller."""

from __future__ import annotations

import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from services.host_memory_guard import (
    EXECUTOR_MEMORY_MIN_MIB,
    MIB,
    UNIT_MEMORY_MIN_MIB,
    _OOM_GROUP_SCRIPT,
    _host_memory_min_script,
    apply_host_memory_guard,
    host_memory_guard_script,
)


def test_script_is_valid_sh_and_targets_each_pod_cgroup():
    script = host_memory_guard_script(["pod_1", "pod_$(x)"])

    assert subprocess.run(["sh", "-n", "-c", script], check=False).returncode == 0
    assert "'pod_$(x)'" in script
    assert "system.slice/docker-$id.scope" in script and "memory.oom.group" in script


def test_system_slice_min_covers_every_unit_once():
    script = host_memory_guard_script(["pod_1"])
    total = sum(UNIT_MEMORY_MIN_MIB.values()) - UNIT_MEMORY_MIN_MIB["ssh.service"] + EXECUTOR_MEMORY_MIN_MIB

    assert f"set_min system.slice {total * MIB}" in script
    assert total * MIB < 4 * 1024 * MIB


@pytest.mark.asyncio
async def test_apply_never_raises_on_ssh_failure():
    async def run(*args, **kwargs):
        raise OSError("connection lost")

    await apply_host_memory_guard(SimpleNamespace(run=run), ["pod_1"])


def _fake_bin(tmp_path, name, body):
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_oom_group_is_set_once_on_the_pod_scope(tmp_path):
    scope = tmp_path / "root/sys/fs/cgroup/system.slice/docker-abc.scope"
    scope.mkdir(parents=True)
    (scope / "memory.oom.group").write_text("0\n")
    docker = _fake_bin(tmp_path, "docker", "echo abc\n")
    script = _OOM_GROUP_SCRIPT % {"names": "pod_1"}
    script = script.replace("/proc/1/root", str(tmp_path / "root")).replace("/usr/bin/docker", str(docker))

    first = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=False)
    second = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=False)

    assert (scope / "memory.oom.group").read_text().strip() == "1"
    assert first.stdout.split() == ["oom_group_set", "abc"]
    assert second.stdout == ""


def test_memory_min_is_set_on_active_units_and_system_slice_once(tmp_path):
    state = tmp_path / "sd"
    state.mkdir()
    _fake_bin(
        tmp_path,
        "systemctl",
        f'case "$1" in show) cat "{state}/$5" 2>/dev/null || echo 0;; '
        'is-active) [ "$3" = docker.service ] || [ "$3" = ssh.service ];; '
        f'set-property) echo "${{4#MemoryMin=}}" > "{state}/$3";; esac\n',
    )
    inner = shlex.split(_host_memory_min_script().split("sh -c ", 1)[1])[0]
    inner = inner.replace("[ -d /run/systemd/system ] && [ -f /sys/fs/cgroup/cgroup.controllers ] || exit 0; ", "")
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}

    first = subprocess.run(["sh", "-c", inner], capture_output=True, text=True, env=env, check=False)
    second = subprocess.run(["sh", "-c", inner], capture_output=True, text=True, env=env, check=False)

    assert sorted(p.name for p in state.iterdir()) == ["docker.service", "ssh.service", "system.slice"]
    assert (state / "docker.service").read_text().strip() == str(UNIT_MEMORY_MIN_MIB["docker.service"] * MIB)
    assert "memory_min_set system.slice" in first.stdout
    assert second.stdout == ""
