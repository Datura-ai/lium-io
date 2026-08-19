"""The supervisor is independent of cvmd, and cvmd can tell truthfully when it is gone.

Both properties were learned on hardware. On au11 the supervisor started as cvmd's direct child,
and cvmd never calls `wait()` on it — so a stopped supervisor sat as a **zombie**. A zombie is
still a member of its process group, so `killpg(pgid, 0)` kept succeeding: every teardown ran
the whole signal ladder (measured: 260s) and ended by reporting "still present after SIGKILL"
about a guest that had powered off gracefully minutes earlier. `os.kill(pid, 0)` shares the
blind spot, and that is exactly what `shutdown_instance` waits on.

These tests spawn the real `spawn()` against a stand-in that calls the real `_detach`, so the
shipping code is what gets asserted — only the part that would start QEMU is replaced.
"""

import os
import signal
import time
from pathlib import Path

import pytest
from cvmd.cvm import supervisor

FIXTURE_MODULE = "fixtures.sleepy_supervisor"
SETTLE_SECONDS = 10


def process_field(pid: int, field: str) -> int | None:
    """Read one field out of /proc/<pid>/stat. None if the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The comm field is parenthesised and may contain spaces, so split after it.
    after_comm = raw[raw.rindex(")") + 2 :].split()
    # stat fields from `state` (index 0 here) — ppid is 1, pgrp 2, session 3.
    return {
        "state": after_comm[0],
        "ppid": int(after_comm[1]),
        "pgid": int(after_comm[2]),
        "sid": int(after_comm[3]),
    }[field]


def wait_until(predicate, seconds: int = SETTLE_SECONDS) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def detached(tmp_path, monkeypatch):
    """Spawn the stand-in supervisor and yield its pid, killing it afterwards whatever happens."""
    monkeypatch.setattr(supervisor, "CHILD_MODULE", FIXTURE_MODULE)
    # Prepended, not replaced: the subprocess still has to find `cvmd` the same way this
    # process did, and the stand-in imports the real `_detach` from it.
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [str(Path(__file__).parent), existing]))
    )

    vm_dir = tmp_path / "instance"
    vm_dir.mkdir()
    pid = supervisor.spawn(scripts_dir=tmp_path, vm_dir=vm_dir, kp_port=3443)
    try:
        yield pid, vm_dir
    finally:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class TestDetachment:
    def test_the_supervisor_is_not_cvmd_s_child(self, detached):
        """Orphaned to init, so nothing has to reap it and nothing can leave it a zombie."""
        pid, _ = detached
        assert process_field(pid, "ppid") == 1

    def test_it_leads_its_own_session_and_process_group(self, detached):
        """QEMU lands in a group whose id is this pid, which is how teardown reaches both."""
        pid, _ = detached
        assert process_field(pid, "sid") == pid
        assert process_field(pid, "pgid") == pid

    def test_spawn_returns_the_pid_that_holds_the_vm(self, detached):
        """Popen sees the process that forks and exits, never the one that matters."""
        pid, vm_dir = detached
        assert int((vm_dir / supervisor.PID_FILE).read_text().strip()) == pid
        assert process_field(pid, "state") is not None

    def test_it_survives_a_signal_aimed_at_cvmd_s_own_group(self, detached):
        """A CVM must outlive a cvmd restart, so it must not share cvmd's process group."""
        pid, _ = detached
        assert process_field(pid, "pgid") != os.getpgid(0)

    def test_cvmd_is_left_with_no_zombie(self, detached):
        """The defect this file exists for.

        Before the double-fork, a stopped supervisor stayed a zombie in its own process group,
        so `killpg(pgid, 0)` never reported it gone and teardown ran the whole signal ladder
        before declaring a failure that had not happened.
        """
        pid, _ = detached
        os.killpg(pid, signal.SIGKILL)

        assert wait_until(lambda: process_field(pid, "state") in (None, "X")), (
            f"pid {pid} is still {process_field(pid, 'state')} — a zombie here means cvmd is "
            f"its parent again, and every teardown will burn its full timeout"
        )

    def test_the_process_group_reads_as_gone_once_it_is(self, detached):
        """The predicate teardown actually uses, against a real killed process."""
        pid, _ = detached
        os.killpg(pid, signal.SIGKILL)

        assert wait_until(lambda: supervisor._process_group_gone(pid))


class TestSpawnOverASurvivingDirectory:
    def test_a_stale_pid_file_from_a_previous_life_never_wins(self, tmp_path, monkeypatch):
        """DAH-2679: a renter relaunch spawns over a directory that still holds the OLD
        life's pid file, and `_await_pid_file` returns the first pid it can parse — so
        `spawn` must remove the stale file before the child that writes the new one exists.
        Without the unlink, the very first poll would return the dead (or recycled) pid."""
        monkeypatch.setattr(supervisor, "CHILD_MODULE", FIXTURE_MODULE)
        existing = os.environ.get("PYTHONPATH", "")
        monkeypatch.setenv(
            "PYTHONPATH", os.pathsep.join(filter(None, [str(Path(__file__).parent), existing]))
        )
        vm_dir = tmp_path / "instance"
        vm_dir.mkdir()
        stale = 999_999_999
        (vm_dir / supervisor.PID_FILE).write_text(f"{stale}\n")

        pid = supervisor.spawn(scripts_dir=tmp_path, vm_dir=vm_dir, kp_port=3443)
        try:
            assert pid != stale
            assert int((vm_dir / supervisor.PID_FILE).read_text().strip()) == pid
            assert process_field(pid, "state") is not None
        finally:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class TestTheUnitLetsItSurvive:
    """Detaching is not enough on its own — systemd kills by cgroup, not by session.

    Measured on au11: with the default `KillMode=control-group`, `systemctl restart cvmd` logged
    `cvmd.service: Killing process 218807 (qemu-system-x86) with signal SIGKILL` and the node
    came back FAILED holding a CVM directory and no CVM. The double-fork puts the supervisor in
    its own *session*; nothing in a process's control moves it out of its unit's *cgroup*.

    Asserted against the packaged unit because no test host has systemd, and because the
    property is entirely a property of that file.
    """

    UNIT = Path(__file__).resolve().parents[1] / "packaging" / "cvmd.service"

    def test_the_unit_stops_only_the_daemon(self):
        assert "KillMode=process" in self.UNIT.read_text(), (
            "without KillMode=process, restarting cvmd kills the node's CVM — systemd's default "
            "kills every process in the unit's cgroup, and detaching does not leave it"
        )

    def test_the_reason_is_recorded_beside_it(self):
        """A one-word setting whose reason is off-page is a setting someone tidies away."""
        text = self.UNIT.read_text()
        assert "cgroup" in text and "RECONCILING" in text


class TestConsoleLog:
    def test_the_child_s_output_is_not_held_in_a_buffer(self, detached):
        """`-u`, and why it matters.

        Python block-buffers stdout when it is a file. Without `-u` the QEMU command line that
        `run_instance` prints sits in the child's buffer for the VM's whole life, while QEMU
        writes past it through the same descriptor — so the one line saying what was launched is
        missing from the log for exactly as long as anyone would want to read it.
        """
        _pid, vm_dir = detached
        log = supervisor.console_log_path(vm_dir)

        assert wait_until(
            lambda: "detached" in log.read_text()
        ), f"the console log is empty while the supervisor runs: {log.read_text()!r}"
