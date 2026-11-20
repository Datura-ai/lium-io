"""DAH-6379: the host memory guard sets MemoryMin on the host services, in a script that parses, and never
fails the caller."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
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


def _run_memory_min(
    tmp_path,
    executor_cgroup: str,
    set_property: str = 'echo "${4#MemoryMin=}" > "$S/$3"',
    systemd: bool = True,
    procs: tuple = (),
    active: tuple = ("docker.service", "ssh.service"),
):
    """Runs the memory.min script against a fake systemctl (`active` units active, MemoryMin from the state dir),
    a fake cgroup tree, a fake /proc placing this shell at executor_cgroup and holding `procs` ((comm, cgroup[, PID
    namespace[, mount namespace]]) tuples; PID 1 and the default namespaces are the host's), and a host with or
    without systemd. Returns a callable giving the script's stdout."""
    state, root, proc = tmp_path / "sd", tmp_path / "cg", tmp_path / "proc"
    state.mkdir(exist_ok=True)
    (proc / "self").mkdir(parents=True, exist_ok=True)
    (proc / "self" / "cgroup").write_text(f"0::{executor_cgroup}\n")
    for pid, (comm, cgroup, *ns) in enumerate([("init", "/init"), *procs], 1):
        (proc / str(pid) / "ns").mkdir(parents=True, exist_ok=True)
        (proc / str(pid) / "comm").write_text(f"{comm}\n")
        (proc / str(pid) / "cgroup").write_text(f"0::{cgroup}\n")
        for kind, name in zip(("pid", "mnt"), (*ns, "host", "host")):
            if not (proc / str(pid) / "ns" / kind).is_symlink():
                (proc / str(pid) / "ns" / kind).symlink_to(f"{kind}:[{name}]")
    for cgroup in [executor_cgroup, *(p[1] for p in procs)]:
        if cgroup.strip("/") and not cgroup.startswith("/.."):
            (root / cgroup.lstrip("/")).mkdir(parents=True, exist_ok=True)
    run_systemd = tmp_path / "run_systemd"
    if systemd:
        run_systemd.mkdir(exist_ok=True)
    _fake_bin(
        tmp_path,
        "systemctl",
        f'S="{state}"; case "$1" in show) cat "$S/$5" 2>/dev/null || echo 0;; '
        f'is-active) case " {" ".join(active)} " in *" $3 "*) ;; *) exit 1;; esac;; '
        f"set-property) {set_property};; esac\n",
    )
    inner = shlex.split(_host_memory_min_script().split("sh -c ", 1)[1])[0]
    inner = inner.split("exit 0; }; ", 1)[1]
    inner = (
        inner.replace("/sys/fs/cgroup", str(root))
        .replace("/run/systemd/system", str(run_systemd))
        .replace("/run/lium-host-memory-guard", str(tmp_path / "guard-base"))
        .replace("/proc/", f"{proc}/")
    )
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


@pytest.mark.parametrize("existing", [f"{EX * 2}", "max"])
def test_cgroupfs_parent_with_more_protection_is_not_lowered(tmp_path, existing):
    run = _run_memory_min(tmp_path, "/docker/abc")
    (tmp_path / "cg/docker/memory.min").write_text(f"{existing}\n")

    run()

    assert (tmp_path / "cg/docker/memory.min").read_text().strip() == existing


def test_systemd_unit_with_more_protection_is_not_lowered(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")
    (tmp_path / "sd/system.slice").write_text(f"{UNITS_MIB * MIB * 4}\n")
    (tmp_path / "sd/docker.service").write_text("infinity\n")

    out = run()

    assert (tmp_path / "sd/docker.service").read_text().strip() == "infinity"
    # the slice holding an unlimited unit is made unlimited too, so it never caps the unit's protection
    assert (tmp_path / "sd/system.slice").read_text().strip() == "infinity"
    assert "memory_min_set docker.service" not in out


def test_protected_total_stays_under_the_pod_reserve():
    assert UNITS_MIB * MIB + EX < 4 * 1024 * MIB


def test_memory_min_reports_a_refused_set_property(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope", set_property="exit 1")

    failed = parse_guard_output(run())["failed"]

    assert "memory_min docker.service set_property_failed" in failed
    assert "memory_min system.slice set_property_failed" in failed


def test_executor_on_another_cgroup_path_gets_memory_min_written_with_its_parents(tmp_path):
    run = _run_memory_min(tmp_path, "/user.slice/x")

    out = parse_guard_output(run())

    root = tmp_path / "cg"
    assert _read(root / "user.slice/x/memory.min") == EX
    assert _read(root / "user.slice/memory.min") == EX
    assert out["failed"] == []


@pytest.mark.parametrize("existing_mib", [0, 2048, "infinity"])
def test_executor_under_system_slice_counts_in_the_slice_set_by_systemd(tmp_path, existing_mib):
    active = ("docker.service", "ssh.service", "containerd.service")
    run = _run_memory_min(tmp_path, "/system.slice/containerd.service/x", active=active)
    root, state = tmp_path / "cg", tmp_path / "sd"
    if existing_mib == "infinity":
        # systemd writes an infinite MemoryMin to the unit cgroup as max
        (state / "containerd.service").write_text("infinity\n")
        (root / "system.slice/containerd.service/memory.min").write_text("max\n")

        run()

        assert _read(root / "system.slice/containerd.service/x/memory.min") == EX
        assert (root / "system.slice/containerd.service/memory.min").read_text().strip() == "max"
        assert (state / "containerd.service").read_text().strip() == "infinity"
        assert (state / "system.slice").read_text().strip() == "infinity"
        return
    if existing_mib:
        (state / "containerd.service").write_text(f"{existing_mib * MIB}\n")
    share = max(existing_mib, UNIT_MEMORY_MIN_MIB["containerd.service"]) * MIB

    run()

    assert _read(root / "system.slice/containerd.service/x/memory.min") == EX
    assert _read(root / "system.slice/containerd.service/memory.min") == EX + share
    assert _read(state / "containerd.service") == share
    assert not (root / "system.slice/memory.min").exists()
    extra = share - UNIT_MEMORY_MIN_MIB["containerd.service"] * MIB
    assert _read(state / "system.slice") == UNITS_MIB * MIB + extra + EX


def test_executor_with_a_larger_existing_reservation_under_system_slice_counts_in_full(tmp_path):
    executor = "/system.slice/custom.slice/executor"
    run = _run_memory_min(tmp_path, executor)
    (tmp_path / "cg" / executor.lstrip("/") / "memory.min").write_text(f"{4096 * MIB}\n")

    run()

    assert _read(tmp_path / "cg/system.slice/custom.slice/memory.min") == 4096 * MIB
    assert _read(tmp_path / "sd/system.slice") == UNITS_MIB * MIB + 4096 * MIB


@pytest.mark.parametrize(("existing", "slice_total"), [(4096 * MIB, UNITS_MIB * MIB + 4096 * MIB), ("infinity", "infinity")])
def test_executor_scope_with_a_larger_existing_memory_min_counts_in_full_in_the_slice(tmp_path, existing, slice_total):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")
    (tmp_path / "sd/docker-abc.scope").write_text(f"{existing}\n")

    run()

    assert (tmp_path / "sd/docker-abc.scope").read_text().strip() == str(existing)
    assert (tmp_path / "sd/system.slice").read_text().strip() == str(slice_total)


def test_executor_directly_in_system_slice_adds_its_share_to_the_slice(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice")

    run()

    assert _read(tmp_path / "sd/system.slice") == UNITS_MIB * MIB + EX


@pytest.mark.parametrize(
    ("path", "reason"),
    [("/", "root_cgroup"), ("/../../docker/abc", "outside_host_cgroup_namespace")],
)
def test_executor_path_that_cannot_be_protected_is_reported(tmp_path, path, reason):
    run = _run_memory_min(tmp_path, path)

    assert f"memory_min executor {reason}" in parse_guard_output(run())["failed"]


def test_host_without_systemd_protects_each_service_cgroup_and_the_executor(tmp_path):
    procs = (
        ("sshd", "/openrc.sshd"),
        ("sshd", "/openrc.sshd"),
        ("dockerd", "/openrc.docker"),
        ("containerd", "/openrc.docker"),
        ("sysbox-mgr", "/"),
        ("bash", "/openrc.sshd"),
    )
    run = _run_memory_min(tmp_path, "/docker/abc", systemd=False, procs=procs)

    first, second = parse_guard_output(run()), parse_guard_output(run())

    root = tmp_path / "cg"
    mib = UNIT_MEMORY_MIN_MIB
    assert _read(root / "openrc.sshd/memory.min") == mib["sshd.service"] * MIB
    assert _read(root / "openrc.docker/memory.min") == (mib["docker.service"] + mib["containerd.service"]) * MIB
    assert _read(root / "docker/abc/memory.min") == EX
    assert _read(root / "docker/memory.min") == EX
    assert first["failed"] == ["memory_min sysbox-mgr root_cgroup"]
    assert not list((tmp_path / "sd").iterdir())
    assert second["set"] == []


@pytest.mark.parametrize(
    ("host_proc", "host_cgroup", "unit"),
    [
        (("sshd", "/host/ssh"), "host/ssh", "sshd.service"),
        # a host daemon in its own mount namespace is still a host service
        (("dockerd", "/host/docker", "host", "own"), "host/docker", "docker.service"),
    ],
)
def test_host_without_systemd_protects_the_host_service_not_a_rental_or_executor_one(tmp_path, host_proc, host_cgroup, unit):
    name = host_proc[0]
    procs = ((name, "/rental/svc", "rental", "rental"), (name, "/docker/abc/svc"), host_proc)
    run = _run_memory_min(tmp_path, "/docker/abc", systemd=False, procs=procs)

    out = parse_guard_output(run())

    root = tmp_path / "cg"
    assert _read(root / host_cgroup / "memory.min") == UNIT_MEMORY_MIN_MIB[unit] * MIB
    assert not (root / "rental/memory.min").exists()
    assert _read(root / "docker/abc/memory.min") == EX
    assert out["failed"] == []


def test_host_without_systemd_reports_a_service_whose_pid_namespace_is_unreadable(tmp_path):
    run = _run_memory_min(tmp_path, "/docker/abc", systemd=False, procs=(("sshd", "/host/ssh"),))
    (tmp_path / "proc/2/ns/pid").unlink()

    out = parse_guard_output(run())

    assert out["failed"] == ["memory_min sshd pid_namespace_unreadable"]
    assert not (tmp_path / "cg/host/ssh/memory.min").exists()


def test_host_without_systemd_sums_system_slice_itself(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/executor", systemd=False, procs=(("sshd", "/system.slice/sshd"),))

    run()

    sshd = UNIT_MEMORY_MIN_MIB["sshd.service"] * MIB
    assert _read(tmp_path / "cg/system.slice/memory.min") == EX + sshd


def test_executor_path_with_a_space_gets_memory_min_with_its_parent(tmp_path):
    run = _run_memory_min(tmp_path, "/tenant/executor one")

    first, second = parse_guard_output(run()), parse_guard_output(run())

    root = tmp_path / "cg"
    assert _read(root / "tenant/executor one/memory.min") == EX
    assert _read(root / "tenant/memory.min") == EX
    assert first["failed"] == [] and second["set"] == []


@pytest.mark.parametrize("existing", [2048 * MIB, 4096 * MIB, "max"])
@pytest.mark.parametrize("nested", [False, True])
def test_host_without_systemd_counts_a_larger_existing_service_reservation_in_its_parents(tmp_path, existing, nested):
    """A sibling service's existing memory.min adds to the parent's; on a service holding the executor it is the
    whole subtree's, so it only raises that cgroup. Either way a repeat run changes nothing."""
    executor = "/host/containerd/executor" if nested else "/host/executor"
    run = _run_memory_min(tmp_path, executor, systemd=False, procs=(("containerd", "/host/containerd"),))
    (tmp_path / "cg/host/containerd/memory.min").write_text(f"{existing}\n")

    first, second = run(), run()

    root, own = tmp_path / "cg", UNIT_MEMORY_MIN_MIB["containerd.service"] * MIB
    assert _read(root / executor.lstrip("/") / "memory.min") == EX
    if existing == "max":
        assert (root / "host/memory.min").read_text().strip() == "max"
    elif nested:
        assert _read(root / "host/containerd/memory.min") == max(existing, own + EX)
        assert _read(root / "host/memory.min") == max(existing, own + EX)
    else:
        assert _read(root / "host/containerd/memory.min") == existing
        assert _read(root / "host/memory.min") == existing + EX
    assert parse_guard_output(second)["set"] == [], first


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


def test_executor_nested_under_a_docker_scope_is_written_with_its_parent(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope/worker.scope")

    out = parse_guard_output(run())

    root, state = tmp_path / "cg", tmp_path / "sd"
    assert _read(root / "system.slice/docker-abc.scope/worker.scope/memory.min") == EX
    assert _read(root / "system.slice/docker-abc.scope/memory.min") == EX
    assert not (state / "docker-abc.scope/worker.scope").exists()
    assert _read(state / "system.slice") == UNITS_MIB * MIB + EX
    assert out["failed"] == []


@pytest.mark.parametrize(("granted", "sibling", "parent"), [
    (4096 * MIB, 4096 * MIB, 4096 * MIB + EX),  # a reservation the parent already grants keeps it, the executor adds
    (0, "max", EX),  # a claim the parent never granted (a rental's own) does not raise the parent
    (2048 * MIB, 4096 * MIB, 2048 * MIB + EX),  # only the granted part counts
])
def test_parent_counts_the_protection_its_other_children_already_hold(tmp_path, granted, sibling, parent):
    run = _run_memory_min(tmp_path, "/tenant/executor", systemd=False)
    root = tmp_path / "cg"
    (root / "tenant/other").mkdir(parents=True)
    (root / "tenant/other/memory.min").write_text(f"{sibling}\n")
    (root / "tenant/memory.min").write_text(f"{granted}\n")

    run()
    run()  # a later run counts the same grant, not the guard's own earlier increment

    assert _read(root / "tenant/memory.min") == parent
    assert _read(root / "tenant/executor/memory.min") == EX
    assert (root / "tenant/other/memory.min").read_text().strip() == str(sibling)


@pytest.mark.parametrize(("granted", "other", "counted"), [
    (4096 * MIB, 4096 * MIB, 4096 * MIB),  # a reservation the slice already grants keeps it
    (0, "max", 0),  # a claim the slice never granted (a rental's own scope) does not raise the slice
    (2048 * MIB, 4096 * MIB, 2048 * MIB),  # only the granted part counts
])
def test_system_slice_counts_the_protection_its_other_children_already_hold(tmp_path, granted, other, counted):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")
    root, state = tmp_path / "cg", tmp_path / "sd"
    (root / "system.slice/other.service").mkdir(parents=True)
    (root / "system.slice/other.service/memory.min").write_text(f"{other}\n")
    (state / "system.slice").write_text(f"{granted}\n")

    run()
    first = _read(state / "system.slice")
    run()  # a later run counts the same grant, not the guard's own earlier increment

    assert first == max(granted, UNITS_MIB * MIB + EX + counted)
    assert _read(state / "system.slice") == first


def test_other_children_count_together_up_to_what_the_parent_granted(tmp_path):
    """Two rentals claiming max under a parent granting 2 GiB add 2 GiB in total, not 2 GiB each."""
    run = _run_memory_min(tmp_path, "/tenant/executor", systemd=False)
    root = tmp_path / "cg"
    for name in ("a", "b"):
        (root / "tenant" / name).mkdir(parents=True)
        (root / "tenant" / name / "memory.min").write_text("max\n")
    (root / "tenant/memory.min").write_text(f"{2048 * MIB}\n")

    run()
    run()

    assert _read(root / "tenant/memory.min") == 2048 * MIB + EX


def test_system_slice_other_children_count_together_up_to_what_the_slice_granted(tmp_path):
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")
    root, state = tmp_path / "cg", tmp_path / "sd"
    for name in ("a.scope", "b.scope"):
        (root / "system.slice" / name).mkdir(parents=True)
        (root / "system.slice" / name / "memory.min").write_text("max\n")
    (state / "system.slice").write_text(f"{2048 * MIB}\n")

    run()
    run()

    assert _read(state / "system.slice") == UNITS_MIB * MIB + EX + 2048 * MIB


def test_parent_baseline_is_taken_before_the_first_raise_even_without_siblings(tmp_path):
    """A sibling that appears after the first run is not credited with the executor's grant."""
    run = _run_memory_min(tmp_path, "/tenant/executor", systemd=False)
    root = tmp_path / "cg"

    run()  # no sibling yet: the parent's baseline (0) is recorded before it is raised
    (root / "tenant/other").mkdir()
    (root / "tenant/other/memory.min").write_text("max\n")
    run()

    assert _read(root / "tenant/memory.min") == EX


def test_system_slice_reads_one_baseline_record_when_two_were_written(tmp_path):
    """Two first runs that raced before the lock each appended a baseline; a later run reads the first one."""
    run = _run_memory_min(tmp_path, "/system.slice/docker-abc.scope")
    root, state = tmp_path / "cg", tmp_path / "sd"
    (root / "system.slice/other.scope").mkdir(parents=True)
    (root / "system.slice/other.scope/memory.min").write_text("max\n")
    (state / "system.slice").write_text("0\n")
    (tmp_path / "guard-base.").write_text("0 /system.slice\n0 /system.slice\n")

    out = parse_guard_output(run())

    assert out["failed"] == []
    assert _read(state / "system.slice") == UNITS_MIB * MIB + EX


def test_runs_wait_for_each_other(tmp_path):
    """A second run waits for the lock, so a baseline is recorded once, before the first raise."""
    run = _run_memory_min(tmp_path, "/tenant/executor", systemd=False)
    lock = tmp_path / "guard-base..lock"
    holder = subprocess.Popen(["flock", str(lock), "sh", "-c", "sleep 1.5"])
    try:
        time.sleep(0.3)
        start = time.monotonic()
        out = run()
        waited = time.monotonic() - start
    finally:
        holder.wait()
    assert waited >= 1.0
    assert "memory_min_set /tenant/executor" in out
