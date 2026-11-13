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
    parse_guard_output,
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

    result = await apply_host_memory_guard(SimpleNamespace(run=run), ["pod_1"])

    assert result["failed"] == ["ssh OSError('connection lost')"]


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
    inner = inner.split("[ -f /sys/fs/cgroup/cgroup.controllers ] || { echo memory_min_skipped no_cgroup_v2; exit 0; }; ", 1)[1]
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}

    first = subprocess.run(["sh", "-c", inner], capture_output=True, text=True, env=env, check=False)
    second = subprocess.run(["sh", "-c", inner], capture_output=True, text=True, env=env, check=False)

    assert sorted(p.name for p in state.iterdir()) == ["docker.service", "ssh.service", "system.slice"]
    assert (state / "docker.service").read_text().strip() == str(UNIT_MEMORY_MIN_MIB["docker.service"] * MIB)
    assert "memory_min_set system.slice" in first.stdout
    assert parse_guard_output(second.stdout)["set"] == []


def test_oom_group_reports_a_pod_without_a_cgroup_v2_scope(tmp_path):
    (tmp_path / "root/sys/fs/cgroup").mkdir(parents=True)
    docker = _fake_bin(tmp_path, "docker", "echo abc\n")
    script = _OOM_GROUP_SCRIPT % {"names": "pod_1"}
    script = script.replace("/proc/1/root", str(tmp_path / "root")).replace("/usr/bin/docker", str(docker))

    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=False).stdout

    assert parse_guard_output(out) == {"set": [], "failed": ["oom_group abc no_cgroup_v2_scope"]}


def test_memory_min_reports_a_refused_set_property(tmp_path):
    _fake_bin(
        tmp_path,
        "systemctl",
        'case "$1" in show) echo 0;; is-active) [ "$3" = docker.service ];; set-property) exit 1;; esac\n',
    )
    inner = shlex.split(_host_memory_min_script().split("sh -c ", 1)[1].rsplit(" 2>/dev/null", 1)[0])[0]
    inner = inner.split("exit 0; }; ", 2)[2]
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}

    out = subprocess.run(["sh", "-c", inner], capture_output=True, text=True, env=env, check=False).stdout

    failed = parse_guard_output(out)["failed"]
    assert "memory_min docker.service set_property_failed" in failed
    assert "memory_min system.slice set_property_failed" in failed


@pytest.mark.asyncio
async def test_unapplied_settings_log_a_warning_with_the_node_and_reason(caplog):
    async def run(*args, **kwargs):
        return SimpleNamespace(stdout="oom_group_set abc\nmemory_min_skipped no_systemd\n")

    with caplog.at_level("WARNING"):
        result = await apply_host_memory_guard(SimpleNamespace(run=run), ["pod_1"], {"executor_uuid": "node-1"})

    assert result == {"set": ["oom_group abc"], "failed": ["memory_min no_systemd"]}
    warning = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning) == 1
    extra = warning[0].msg.extra
    assert extra["executor_uuid"] == "node-1" and extra["failed"] == ["memory_min no_systemd"]
