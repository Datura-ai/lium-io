"""The four release conditions, and the loop that waits on them.

These read `/proc` and `/proc/meminfo`, so every test here points them at a fixture tree it
built itself. That is the only way to assert what a zombie, a held VFIO descriptor or a host
still short of memory look like from a test process — the real conditions are staged against
real hardware in the acceptance run, and the two are meant to be read together.
"""

import os
import socket
from pathlib import Path

import pytest
from cvmd.cvm import ports, release

ONE_GIB_KIB = 1024 * 1024


def make_proc(tmp_path, processes: dict) -> Path:
    """A fake `/proc`. `processes` maps pid -> (comm, state, pgrp, [fd targets])."""
    root = tmp_path / "proc"
    root.mkdir()
    for pid, (comm, state, pgrp, fds) in processes.items():
        entry = root / str(pid)
        entry.mkdir()
        (entry / "stat").write_text(f"{pid} ({comm}) {state} 1 {pgrp} 0 0 0 0 0\n")
        fd_dir = entry / "fd"
        fd_dir.mkdir()
        for number, target in enumerate(fds):
            (fd_dir / str(number)).symlink_to(target)
    return root


def make_meminfo(tmp_path, **values) -> Path:
    path = tmp_path / "meminfo"
    path.write_text("".join(f"{name}:{value:>12} kB\n" for name, value in values.items()))
    return path


class TestGroupMembers:
    def test_it_finds_the_members_of_the_group_and_no_others(self, tmp_path):
        proc = make_proc(
            tmp_path,
            {
                100: ("python3", "S", 100, []),
                101: ("qemu-system-x86_64", "S", 100, []),
                200: ("sshd", "S", 200, []),
            },
        )

        assert release.group_members(100, proc=proc) == [
            "pid 100 (python3)",
            "pid 101 (qemu-system-x86_64)",
        ]

    def test_a_zombie_is_a_member_and_is_named_as_one(self, tmp_path):
        """The DAH-2576 bug, as a condition. `killpg(pgid, 0)` succeeds against a zombie, so a
        teardown that trusts it reports a failure that never happened — or, worse, a success."""
        proc = make_proc(tmp_path, {101: ("qemu-system-x86_64", "Z", 100, [])})

        assert release.group_members(100, proc=proc) == ["pid 101 (qemu-system-x86_64, zombie)"]

    def test_an_empty_group_is_an_empty_list(self, tmp_path):
        proc = make_proc(tmp_path, {200: ("sshd", "S", 200, [])})

        assert release.group_members(100, proc=proc) == []

    def test_a_command_name_with_spaces_and_parentheses_still_parses(self, tmp_path):
        """`comm` is not escaped in /proc/<pid>/stat, so the fields after it are found from the
        LAST `)`. Getting this wrong reads the state and the group from the command name."""
        proc = make_proc(tmp_path, {7: ("qemu (tdx) x86", "Z", 7, [])})

        assert release.group_members(7, proc=proc) == ["pid 7 (qemu (tdx) x86, zombie)"]


class TestVfioHolders:
    def test_a_process_holding_a_vfio_group_is_named_with_the_device(self, tmp_path):
        proc = make_proc(
            tmp_path,
            {
                100: ("qemu-system-x86_64", "S", 100, ["/dev/vfio/42", "/dev/null"]),
                200: ("sshd", "S", 200, ["/dev/null"]),
            },
        )

        holders, unreadable = release.vfio_holders(proc=proc)

        assert holders == ["pid 100 (qemu-system-x86_64) holds /dev/vfio/42"]
        assert unreadable == []

    def test_a_host_with_nothing_open_reports_nothing(self, tmp_path):
        proc = make_proc(tmp_path, {200: ("sshd", "S", 200, ["/dev/null", "/etc/hosts"])})

        assert release.vfio_holders(proc=proc) == ([], [])

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read every /proc/<pid>/fd")
    def test_descriptors_that_cannot_be_read_are_reported_rather_than_ignored(self, tmp_path):
        """Could not look is not the same as found nothing. Reporting it is what keeps this
        check fail-closed on a host where cvmd is not root."""
        proc = make_proc(tmp_path, {100: ("qemu-system-x86_64", "S", 100, ["/dev/vfio/42"])})
        (proc / "100" / "fd").chmod(0o000)
        try:
            holders, unreadable = release.vfio_holders(proc=proc)
        finally:
            (proc / "100" / "fd").chmod(0o700)

        assert holders == []
        assert unreadable == [100]


class TestMemorySizes:
    @pytest.mark.parametrize(
        ("spec", "kib"),
        [
            ("1T", 1024 * 1024 * 1024),
            ("2G", 2 * 1024 * 1024),
            ("512M", 512 * 1024),
            ("2048", 2048 * 1024),
        ],
    )
    def test_it_reads_a_size_the_way_dstack_does(self, spec, kib):
        """A bare number is mebibytes to `DStackManager._convert_memory_to_mb`, which is what
        turned this string into the number QEMU was started with."""
        assert release.memory_kib(spec) == kib

    def test_a_size_dstack_would_not_accept_is_refused(self):
        with pytest.raises(release.MemoryUnreadable):
            release.memory_kib("lots")

    def test_meminfo_is_read_as_kibibytes(self, tmp_path):
        path = make_meminfo(tmp_path, MemTotal=100, MemAvailable=42)

        assert release.meminfo(path=path)["MemAvailable"] == 42
        assert release.mem_available_kib(path=path) == 42

    def test_a_meminfo_without_memavailable_is_an_error_not_a_zero(self, tmp_path):
        path = make_meminfo(tmp_path, MemTotal=100)

        with pytest.raises(release.MemoryUnreadable):
            release.mem_available_kib(path=path)


class TestTheMemoryCondition:
    def _checks(self, tmp_path, **overrides):
        return release.ReleaseChecks(
            supervisor_pid=1,
            mappings=[],
            guest_memory_kib=8 * ONE_GIB_KIB,
            memory_tolerance=0.9,
            proc=make_proc(tmp_path, {}),
            **overrides,
        )

    def test_enough_free_for_this_guest_again_is_enough(self, tmp_path):
        checks = self._checks(
            tmp_path, meminfo_path=make_meminfo(tmp_path, MemAvailable=16 * ONE_GIB_KIB)
        )

        assert checks.memory_returned().satisfied

    def test_memory_still_out_with_the_guest_is_not(self, tmp_path):
        """The fleet's standing bug: QEMU has exited and the host has not got its RAM back."""
        checks = self._checks(
            tmp_path, meminfo_path=make_meminfo(tmp_path, MemAvailable=1 * ONE_GIB_KIB)
        )

        condition = checks.memory_returned()

        assert not condition.satisfied
        assert "1.0 GiB of the 7.2 GiB this guest needs" in condition.detail

    def test_a_busy_host_that_gave_the_guest_s_memory_back_is_free(self, tmp_path):
        """The absolute rule alone would fail this node forever: something else on the host is
        using the memory, and a teardown is not responsible for that."""
        checks = self._checks(
            tmp_path,
            meminfo_path=make_meminfo(tmp_path, MemAvailable=9 * ONE_GIB_KIB),
            baseline_mem_available_kib=1 * ONE_GIB_KIB,
        )

        condition = checks.memory_returned()

        assert condition.satisfied
        assert "up 8.0 GiB since teardown began" in condition.detail

    def test_a_guest_that_never_touched_its_allocation_still_completes(self, tmp_path):
        """And the relative rule alone would fail this one: a guest gives back what it held,
        which under TDX is not something QEMU's RSS can be asked about."""
        checks = self._checks(
            tmp_path,
            meminfo_path=make_meminfo(tmp_path, MemAvailable=64 * ONE_GIB_KIB),
            baseline_mem_available_kib=64 * ONE_GIB_KIB,
        )

        assert checks.memory_returned().satisfied

    def test_hugepages_out_of_the_pool_hold_the_condition_open(self, tmp_path):
        checks = self._checks(
            tmp_path,
            hugepages=True,
            meminfo_path=make_meminfo(
                tmp_path, MemAvailable=64 * ONE_GIB_KIB, HugePages_Total=100, HugePages_Free=40
            ),
        )

        condition = checks.memory_returned()

        assert not condition.satisfied
        assert "60 of 100 hugepages are still out of the pool" in condition.detail

    def test_an_empty_hugepage_pool_is_not_something_to_wait_for(self, tmp_path):
        checks = self._checks(
            tmp_path,
            hugepages=True,
            meminfo_path=make_meminfo(
                tmp_path, MemAvailable=64 * ONE_GIB_KIB, HugePages_Total=0, HugePages_Free=0
            ),
        )

        assert checks.memory_returned().satisfied

    def test_a_host_with_no_configured_guest_size_has_nothing_to_reclaim(self, tmp_path):
        checks = release.ReleaseChecks(
            supervisor_pid=1,
            mappings=[],
            guest_memory_kib=None,
            proc=make_proc(tmp_path, {}),
            meminfo_path=make_meminfo(tmp_path, MemAvailable=1),
        )

        assert checks.memory_returned().satisfied


class TestThePortCondition:
    def test_a_port_something_else_holds_is_not_free(self, tmp_path):
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            checks = release.ReleaseChecks(
                supervisor_pid=1,
                mappings=[ports.parse(f"tcp:127.0.0.1:{port}:2200")],
                proc=make_proc(tmp_path, {}),
            )

            condition = checks.ports_free()

        assert not condition.satisfied
        assert str(port) in condition.detail

    def test_a_cvm_that_forwarded_nothing_has_nothing_to_release(self, tmp_path):
        checks = release.ReleaseChecks(supervisor_pid=1, mappings=[], proc=make_proc(tmp_path, {}))

        assert checks.ports_free().satisfied


class Fake:
    """A stand-in for ReleaseChecks that replays a scripted sequence of evaluations."""

    def __init__(self, *rounds):
        self._rounds = list(rounds)

    def evaluate(self):
        flags = self._rounds.pop(0) if len(self._rounds) > 1 else self._rounds[0]
        return [
            release.Condition(name, satisfied, f"{name} {satisfied}")
            for name, satisfied in zip(release.CONDITION_NAMES, flags, strict=True)
        ]


def virtual_clock():
    """A clock that only moves when the loop sleeps, so the timings are the loop's own arithmetic
    rather than the test machine's scheduling."""
    elapsed = [0.0]

    def now():
        return elapsed[0]

    def sleep(seconds):
        elapsed[0] += seconds

    return now, sleep


class TestTheWait:
    def test_a_node_that_is_already_free_answers_without_sleeping(self):
        now, sleep = virtual_clock()
        slept = []

        report = release.verify_released(
            Fake((True, True, True, True)),
            timeout=100,
            poll=5,
            now=now,
            sleep=lambda seconds: slept.append(seconds) or sleep(seconds),
        )

        assert report.complete
        assert slept == []
        assert all(t.satisfied_after_seconds == 0.0 for t in report.timings.values())

    def test_every_condition_is_timed_from_when_it_first_held(self):
        """One total says nothing about which hardware class is slow. Four numbers do."""
        now, sleep = virtual_clock()

        report = release.verify_released(
            Fake(
                (False, False, False, False),
                (True, True, False, True),
                (True, True, True, True),
            ),
            timeout=100,
            poll=5,
            now=now,
            sleep=sleep,
        )

        assert report.complete
        assert report.timings[release.PROCESS_REAPED].satisfied_after_seconds == 5.0
        assert report.timings[release.MEMORY_RETURNED].satisfied_after_seconds == 10.0

    def test_a_condition_that_stops_holding_loses_its_timing(self):
        """Completion means all four true in ONE evaluation. A port that was free a minute ago
        and is bound now is not evidence of anything."""
        now, sleep = virtual_clock()

        report = release.verify_released(
            Fake((False, True, True, True), (True, True, True, False)),
            timeout=5,
            poll=5,
            now=now,
            sleep=sleep,
        )

        assert not report.complete
        assert report.timings[release.PORTS_FREE].satisfied_after_seconds is None

    def test_a_timeout_names_every_condition_that_did_not_hold(self):
        now, sleep = virtual_clock()

        report = release.verify_released(
            Fake((True, False, False, True)), timeout=5, poll=5, now=now, sleep=sleep
        )

        assert not report.complete
        assert report.unsatisfied == [release.VFIO_RELEASED, release.MEMORY_RETURNED]
        assert "vfio_released" in report.why_incomplete()
        assert "memory_returned" in report.why_incomplete()

    def test_the_report_carries_all_four_conditions_whatever_the_outcome(self):
        now, sleep = virtual_clock()

        report = release.verify_released(
            Fake((False, False, False, False)), timeout=0, poll=5, now=now, sleep=sleep
        )

        assert not report.complete
        assert set(report.to_json()["conditions"]) == set(release.CONDITION_NAMES)
