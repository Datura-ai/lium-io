"""DAH-6379: the host memory guard sets MemoryMin on the host services, in a script that parses, and never
fails the caller."""

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
    _host_memory_min_script,
    apply_host_memory_guard,
    host_memory_guard_script,
    parse_guard_output,
)


def test_script_is_valid_sh_and_writes_only_memory_min():
    script = host_memory_guard_script()

    assert subprocess.run(["sh", "-n", "-c", script], check=False).returncode == 0
    assert _host_memory_min_script() in script
    assert "memory.oom.group" not in script


@pytest.mark.asyncio
async def test_apply_never_raises_on_ssh_failure():
    async def run(*args, **kwargs):
        raise OSError("connection lost")

    result = await apply_host_memory_guard(SimpleNamespace(run=run))

    assert result["failed"] == ["ssh OSError('connection lost')"]


def _fake_bin(tmp_path, name, body):
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


UNITS_MIB = sum(UNIT_MEMORY_MIN_MIB.values()) - UNIT_MEMORY_MIN_MIB["ssh.service"]
EX = EXECUTOR_MEMORY_MIN_MIB * MIB


def _run_memory_min(tmp_path, executor_cgroup: str, set_property: str = 'echo "${4#MemoryMin=}" > "$S/$3"'):
    """Runs the memory.min script against a fake systemctl (docker.service and ssh.service active), a fake
    cgroup tree and a fake /proc/self/cgroup placing this shell at executor_cgroup. Returns (state, cgroup root)."""
    state, root = tmp_path / "sd", tmp_path / "cg"
    state.mkdir(exist_ok=True)
    (root / executor_cgroup.lstrip("/")).mkdir(parents=True, exist_ok=True)
    (tmp_path / "self_cgroup").write_text(f"0::{executor_cgroup}\n")
    _fake_bin(
        tmp_path,
        "systemctl",
        f'S="{state}"; case "$1" in show) cat "$S/$5" 2>/dev/null || echo 0;; '
        'is-active) [ "$3" = docker.service ] || [ "$3" = ssh.service ];; '
        f"set-property) {set_property};; esac\n",
    )
    inner = shlex.split(_host_memory_min_script().split("sh -c ", 1)[1])[0]
    inner = inner.split("exit 0; }; ", 2)[2]
    inner = inner.replace("/proc/self/cgroup", str(tmp_path / "self_cgroup")).replace("/sys/fs/cgroup", str(root))
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    return lambda: subprocess.run(["sh", "-c", inner], capture_output=True, text=True, env=env, check=False).stdout


def _read(path):
    return int(path.read_text().strip())


def test_executor_in_a_systemd_scope_gets_memory_min_via_systemctl(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")

    first, second = run(), run()

    state = tmp_path / "sd"
    assert sorted(p.name for p in state.iterdir()) == [
        "docker-abc.scope", "docker.service", "ssh.service", "system.slice"
    ]
    assert _read(state / "docker.service") == UNIT_MEMORY_MIN_MIB["docker.service"] * MIB
    assert _read(state / "docker-abc.scope") == EX
    assert _read(state / "system.slice") == UNITS_MIB * MIB + EX
    assert "memory_min_set system.slice" in first
    assert parse_guard_output(second)["set"] == []


def test_executor_on_a_cgroupfs_host_gets_memory_min_written_with_its_parent(tmp_path):
    run = _run_memory_min(tmp_path, "/docker/abc")

    first, second = run(), run()

    root, state = tmp_path / "cg", tmp_path / "sd"
    assert _read(root / "docker/abc/memory.min") == EX
    assert _read(root / "docker/memory.min") == EX
    assert _read(state / "system.slice") == UNITS_MIB * MIB
    assert "memory_min /docker/abc" in parse_guard_output(first)["set"]
    assert parse_guard_output(second)["set"] == []


def test_cgroupfs_parent_with_more_protection_is_not_lowered(tmp_path):
    run = _run_memory_min(tmp_path, "/docker/abc")
    (tmp_path / "cg/docker/memory.min").write_text(f"{EX * 2}\n")

    run()

    assert _read(tmp_path / "cg/docker/memory.min") == EX * 2


def test_systemd_unit_with_more_protection_is_not_lowered(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")
    (tmp_path / "sd/system.slice").write_text(f"{UNITS_MIB * MIB * 4}\n")
    (tmp_path / "sd/docker.service").write_text("infinity\n")

    out = run()

    assert _read(tmp_path / "sd/system.slice") == UNITS_MIB * MIB * 4
    assert (tmp_path / "sd/docker.service").read_text().strip() == "infinity"
    assert "memory_min_set system.slice" not in out


def test_protected_total_stays_under_the_pod_reserve():
    assert UNITS_MIB * MIB + EX < 4 * 1024 * MIB


def test_memory_min_reports_a_refused_set_property_and_an_unknown_executor_path(tmp_path):
    run = _run_memory_min(tmp_path, "/user.slice/x", set_property="exit 1")

    failed = parse_guard_output(run())["failed"]

    assert "memory_min docker.service set_property_failed" in failed
    assert "memory_min system.slice set_property_failed" in failed
    assert "memory_min executor no_host_cgroup_path" in failed


@pytest.mark.asyncio
async def test_unapplied_settings_log_a_warning_with_the_node_and_reason(caplog):
    async def run(*args, **kwargs):
        return SimpleNamespace(stdout="memory_min_set docker.service\nmemory_min_skipped no_systemd\n")

    with caplog.at_level("WARNING"):
        result = await apply_host_memory_guard(SimpleNamespace(run=run), {"executor_uuid": "node-1"})

    assert result == {"set": ["memory_min docker.service"], "failed": ["memory_min no_systemd"]}
    warning = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning) == 1
    extra = warning[0].msg.extra
    assert extra["executor_uuid"] == "node-1" and extra["failed"] == ["memory_min no_systemd"]
