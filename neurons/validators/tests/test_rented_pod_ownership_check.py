"""RentedPodOwnershipCheck: warn when a rented sysbox pod's / or /root is owned outside its user namespace."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from types import SimpleNamespace

import asyncssh
import pytest
from neurons.validators.src.services.task.checks import rented_pod_ownership
from neurons.validators.src.services.task.checks.rented_pod_ownership import (
    PATHS,
    RentedPodOwnershipCheck,
    judge,
    read_command,
)
from neurons.validators.src.services.task.messages import RentedPodOwnershipMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import FakeRedis, build_services, build_state


def _proc(tmp_path, pid: int, uid_map: str) -> None:
    d = tmp_path / str(pid)
    (d / "root" / "root").mkdir(parents=True)
    (d / "uid_map").write_text(uid_map)


def _ssh(proc: str):
    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout="/pod_1 4242\n/pod_2 4343\n", stderr="")
        # the real read, pointed at the fake /proc
        out = subprocess.run(
            ["sh", "-c", cmd.replace("/proc/", f"{proc}/")], capture_output=True, text=True
        )
        return SimpleNamespace(exit_status=out.returncode, stdout=out.stdout, stderr=out.stderr)

    return SimpleNamespace(run=run)


def _ctx(context_factory, ssh, **overrides):
    rented = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[
                    RentedPod(pod_id=str(i), container_name=f"pod_{i}", status="RUNNING")
                    for i in (1, 2)
                ],
            )
        }
    )
    # each context is its own run, as in the validator
    overrides.setdefault("pipeline_id", str(uuid.uuid4()))
    return context_factory(
        state=build_state(rented_data=rented, specs={"executor_pid_mode": "host"}), ssh=ssh, **overrides
    )


@pytest.mark.asyncio
async def test_files_outside_the_pods_uid_map_warn_but_pass(context_factory, tmp_path):
    me = os.getuid()
    # restarted with a new range its files are not in
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")  # files inside the range
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh(str(tmp_path))))
    assert result.passed is True
    assert result.event.reason_code == Msg.SHIFTED.reason
    pods = {p["container"]: p for p in result.event.what_we_saw["pods"]}
    assert pods["pod_1"]["outside_namespace"] == ["/", "/root"]
    assert pods["pod_1"]["host_owners"] == {"/": me, "/root": me}
    assert pods["pod_2"]["outside_namespace"] == []


@pytest.mark.asyncio
async def test_runc_pods_and_matching_owners_are_ok(context_factory, tmp_path):
    me = os.getuid()
    _proc(
        tmp_path, 4242, "         0          0 4294967295\n"
    )  # no user namespace: nothing can be outside it
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh(str(tmp_path))))
    assert result.event.reason_code == Msg.OK.reason
    pods = {p["container"]: p for p in result.event.what_we_saw["pods"]}
    assert pods["pod_1"]["uid_map_base"] == 0 and pods["pod_1"]["outside_namespace"] == []
    assert "unread" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_a_pod_not_read_back_is_unread_not_ok(context_factory, tmp_path):
    # pod_2 restarted between docker inspect and the /proc read: its pid is gone
    _proc(tmp_path, 4242, f"         0     {os.getuid()}      65536\n")
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh(str(tmp_path))))
    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason
    assert result.event.what_we_saw["unread"] == ["pod_2"]


@pytest.mark.asyncio
async def test_an_unread_pod_names_the_reads_that_came_back_empty(context_factory, tmp_path):
    # the pod's /proc root cannot be stat'ed from the session: the event says so instead of a bare unread
    _proc(tmp_path, 4242, f"         0     {os.getuid()}      65536\n")
    (tmp_path / "4343").mkdir()
    (tmp_path / "4343" / "uid_map").write_text("         0     100000      65536\n")
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh(str(tmp_path))))
    assert result.event.reason_code == Msg.UNREAD.reason
    assert result.event.what_we_saw["empty_reads"] == {"pod_2": ["/", "/root"]}


@pytest.mark.asyncio
async def test_malformed_ssh_output_is_unread_and_does_not_raise(context_factory):
    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout="/pod_1 4242\n/pod_2 ²\n", stderr="")
        return SimpleNamespace(
            exit_status=0, stdout="x|0 100000 65536|0|0\n4242|0 ² 65536|1|1\n", stderr=""
        )

    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, SimpleNamespace(run=run)))
    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason
    assert result.event.what_we_saw["unread"] == ["pod_1", "pod_2"]


@pytest.mark.asyncio
async def test_a_rental_missing_from_inspect_is_unread_not_ok(context_factory, tmp_path):
    # pod_2 disappeared: docker inspect prints only pod_1 and exits 1; pod_1 reads healthy
    _proc(tmp_path, 4242, f"         0     {os.getuid()}      65536\n")
    real = _ssh(str(tmp_path)).run

    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=1, stdout="/pod_1 4242\n", stderr="No such object")
        return await real(cmd, check)

    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, SimpleNamespace(run=run)))
    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason
    assert result.event.what_we_saw["unread"] == ["pod_2"]


@pytest.mark.asyncio
async def test_an_oversized_number_is_unread_and_does_not_raise(context_factory):
    huge = "9" * 5000  # past int()'s digit limit

    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout=f"/pod_1 4242\n/pod_2 {huge}\n", stderr="")
        return SimpleNamespace(
            exit_status=0, stdout=f"{huge}|0 100000 65536|0|0\n4242|0 {huge} 65536|1|1\n", stderr=""
        )

    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, SimpleNamespace(run=run)))
    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason
    assert result.event.what_we_saw["unread"] == ["pod_1", "pod_2"]


def test_read_command_reports_a_missing_pid_without_failing(tmp_path):
    out = subprocess.run(
        ["sh", "-c", read_command([999], proc=str(tmp_path))], capture_output=True, text=True
    )
    assert out.stdout == "999|" + "|" * len(PATHS) + "\n"


@pytest.mark.parametrize(
    "line, outside",
    [
        ("4242|0 100000 65536|100000|", []),  # /root unread (no /root in the image): / alone decides
        ("4242|0 100000 65536|7|", ["/"]),  # / shifted with /root unread
        ("4242|0 100000 65536||100000", None),  # / unread, /root fine: unknown, not healthy
        ("4242|0 100000 65536||7", ["/root"]),  # /root shifted proves it even with / unread
        ("4242|0 100000 65536||", None),
        ("4242|0 100000 65536|100000|100000", []),
    ],
)
def test_an_unread_root_home_is_judged_from_slash_and_an_unread_slash_is_unknown(line, outside):
    pod = judge(line)
    if outside is None:
        assert pod is None
    else:
        assert pod["outside_namespace"] == outside


def test_a_shifted_sudo_file_warns_although_slash_and_root_home_are_inside():
    # a power loss left /etc/sudoers.d/README owned by nobody while / and /root were fine: sudo fails
    pod = judge("4242|0 100000 65536|100000|100000|100000|100000|65534|100000")
    assert pod["outside_namespace"] == ["/etc/sudoers.d/README"]


@pytest.mark.asyncio
async def test_the_read_includes_sudo_files_and_an_image_without_sudo_is_ok(
    context_factory, tmp_path
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me}      65536\n")
    (tmp_path / "4242" / "root" / "etc" / "sudoers.d").mkdir(parents=True)
    (tmp_path / "4242" / "root" / "etc" / "sudoers.d" / "README").write_text("")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")  # no sudo in the image
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh(str(tmp_path))))
    assert result.event.reason_code == Msg.OK.reason
    pods = {p["container"]: p for p in result.event.what_we_saw["pods"]}
    assert pods["pod_1"]["host_owners"] == {
        "/": me,
        "/root": me,
        "/etc/sudoers.d": me,
        "/etc/sudoers.d/README": me,
    }
    assert pods["pod_2"]["host_owners"] == {"/": me, "/root": me}


def test_a_shifted_package_dir_warns_although_slash_root_and_sudo_are_inside():
    # a partial shift left /usr/share owned by nobody while /, /root and sudo were fine: dpkg fails
    pod = judge(
        "4242|0 100000 65536|100000|100000|100000|100000|100000|100000|100000|100000|0|100000"
    )
    assert pod["outside_namespace"] == ["/usr/share"]


def _ssh_with_uptime(proc: str, uptime_and_ages: str):
    inner = _ssh(proc)

    async def run(cmd, check=False):
        if cmd.startswith("cut -d' ' -f1 /proc/uptime"):
            return SimpleNamespace(exit_status=0, stdout=uptime_and_ages, stderr="")
        return await inner.run(cmd, check=check)

    return SimpleNamespace(run=run)


@pytest.mark.parametrize(
    ("uptime_and_ages", "template", "after_boot"),
    [
        # host up a day, pod container started 30 s ago: a `docker restart`, not a host reboot
        ("86400.12\n 4242    30\n", Msg.SHIFTED_AFTER_CONTAINER_RESTART, 86370),
        # host up 700 s, container started 50 s after boot: brought back by the reboot
        ("700.50\n4242 650\n", Msg.SHIFTED, 50),
        # uptime not read: the cause is unknown, the host message stays
        ("\n4242 30\n", Msg.SHIFTED, None),
    ],
)
@pytest.mark.asyncio
async def test_a_shift_after_a_container_only_restart_does_not_blame_the_host(
    context_factory, tmp_path, uptime_and_ages, template, after_boot
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_uptime(str(tmp_path), uptime_and_ages))
    )
    assert result.event.reason_code == Msg.SHIFTED.reason
    assert result.event.remediation == template.remediation
    pods = {p["container"]: p for p in result.event.what_we_saw["pods"]}
    assert pods["pod_1"].get("started_after_boot_s") == after_boot
    assert "started_after_boot_s" not in pods["pod_2"]  # only the shifted pod is read


@pytest.mark.parametrize(
    ("created", "template", "before_boot"),
    [
        # created 36 min before the host booted at 20:20:03Z, started by the validator's stale-mount
        # recovery 1358 s after boot: it was on the host when it stopped, so the host message stays
        ("2026-11-17T19:44:26.034526789Z", Msg.SHIFTED, True),
        # created 30 s after boot and restarted later on the running host: a container-only restart
        ("2026-11-17T20:20:33.5Z", Msg.SHIFTED_AFTER_CONTAINER_RESTART, False),
        # a creation time that is not one: the age alone decides, as before
        ("yesterday", Msg.SHIFTED_AFTER_CONTAINER_RESTART, None),
        # within seconds of the boot either side: rounding and clock slew, and no container is created that soon
        # after a boot, so it was there at the stop however late the recovery started it
        ("2026-11-17T20:20:03.3Z", Msg.SHIFTED, True),
        ("2026-11-17T20:19:59.9Z", Msg.SHIFTED, True),
        ("2026-11-17T20:20:09Z", Msg.SHIFTED_AFTER_CONTAINER_RESTART, False),
    ],
)
@pytest.mark.asyncio
async def test_a_pod_created_before_the_boot_is_blamed_on_the_host_however_late_it_started(
    context_factory, tmp_path, created, template, before_boot
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    boot = 1794946803  # 2026-11-17T20:20:03Z
    out = f"1400.25\n4242 42\n--\n{boot}\n/pod_1 {created}\n"
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_uptime(str(tmp_path), out))
    )
    assert result.event.remediation == template.remediation
    pods = {p["container"]: p for p in result.event.what_we_saw["pods"]}
    assert pods["pod_1"]["started_after_boot_s"] == 1358
    assert pods["pod_1"].get("created_before_boot") is before_boot


@pytest.mark.parametrize(
    ("uptime_and_ages", "template"),
    [
        # pod_1 restarted 30 s ago on a host up a day, pod_2 started 50 s after boot: one cause does not fit both
        ("86400\n4242 30\n4343 86350\n", Msg.SHIFTED_MIXED),
        # pod_2's age not read: the cause is unknown, the host message stays
        ("86400\n4242 30\n", Msg.SHIFTED),
    ],
)
@pytest.mark.asyncio
async def test_shifted_pods_with_different_histories_are_not_all_blamed_on_the_host(
    context_factory, tmp_path, uptime_and_ages, template
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me + 200000}      65536\n")
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_uptime(str(tmp_path), uptime_and_ages))
    )
    assert result.event.remediation == template.remediation


@pytest.mark.asyncio
async def test_each_age_stays_with_its_own_pid_when_one_pid_prints_nothing(context_factory, tmp_path):
    # stand case e: ps prints nothing for pid 4242 (gone since the read) and 30 s for 4343
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me + 200000}      65536\n")
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_uptime(str(tmp_path), "86400\n 4343    30\n"))
    )
    pods = {p["container"]: p for p in result.event.what_we_saw["pods"]}
    assert "started_after_boot_s" not in pods["pod_1"]
    assert pods["pod_2"]["started_after_boot_s"] == 86370
    assert result.event.remediation == Msg.SHIFTED.remediation  # pod_1's history is unknown


UPTIME_S = 100000  # the host booted 100000 s before the read


def _starts(ages: str, clock: str | None) -> str:
    """the check's read, from the fixtures' ages: each last field becomes the kernel start time in ticks since
    boot (100 a second), then the boot epoch and CLK_TCK (None: not readable)"""
    out = []
    for line in ages.splitlines():
        fields = line.split()
        if fields and fields[-1].isdigit():
            fields[-1] = str((UPTIME_S - int(fields[-1])) * 100)
        out.append(" ".join(fields))
    return "\n".join(out) + "\n--\n" + (clock if clock is not None else "") + "\n"


def _ssh_with_runtime_ages(proc: str, ages: str, clock: str | None = "1794500000\n100"):
    inner = _ssh(proc)

    async def run(cmd, check=False):
        if cmd.startswith("ps -eo pid=,ppid=,comm= | grep"):
            return SimpleNamespace(exit_status=0, stdout=_starts(ages, clock), stderr="")
        return await inner.run(cmd, check=check)

    return SimpleNamespace(run=run)


@pytest.mark.parametrize(
    ("ages", "restarted"),
    [
        # needrestart restarted sysbox-fs and the fabric manager 100 s ago; pod_1 runs for a day, pod_2 for 50 s
        (
            "  10     1 sysbox-mgr 90000\n  11     1 sysbox-fs     100\n"
            "  12 1 nv-fabricmanage 100\n--\n 4242 86400\n 4343 50\n",
            {"nv-fabricmanager": ["pod_1"], "sysbox-fs": ["pod_1"]},
        ),
        # every runtime process is older than both pods (a reboot brings them up first): nothing to flag
        ("10 1 sysbox-mgr 90000\n11 1 sysbox-fs 90000\n--\n4242 86400\n4343 50\n", None),
        # a pod age not read flags nothing for that pod
        ("11 1 sysbox-fs 100\n--\n4343 50\n", None),
        # the daemons run since boot; their own helpers (re-execs, seconds old): no restart
        (
            "10 1 sysbox-mgr 90000\n11 1 sysbox-fs 90000\n900 11 sysbox-fs 1\n901 10 sysbox-mgr 2\n"
            "--\n4242 86400\n4343 50\n",
            None,
        ),
        # a restarted daemon is flagged even beside an orphaned helper of the old one
        (
            "11 1 sysbox-fs 100\n77 1 sysbox-fs 90000\n--\n4242 86400\n4343 50\n",
            {"sysbox-fs": ["pod_1"]},
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_runtime_process_younger_than_a_pod_is_listed_as_its_own_outcome_without_failing(
    context_factory, tmp_path, ages, restarted
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_runtime_ages(str(tmp_path), ages))
    )
    assert result.passed is True
    # its own outcome so the restart is logged at INFO even when the last cycle was OK
    assert result.event.reason_code == (Msg.RUNTIME_RESTARTED if restarted else Msg.OK).reason
    assert result.event.what_we_saw.get("runtime_restarted_under_pods") == restarted


@pytest.mark.parametrize("template", [Msg.SHIFTED, Msg.SHIFTED_AFTER_CONTAINER_RESTART, Msg.SHIFTED_MIXED])
def test_shifted_remediation_sends_provider_to_support_not_a_host_ownership_change(template):
    assert "uid_map" not in template.remediation
    assert "Contact support" in template.remediation
    assert "free of charge" in template.remediation


@pytest.mark.parametrize(
    ("created_1", "age_1", "created_2", "age_2", "template"),
    [
        # both created before the boot, one started late by the stale-mount recovery: the host stop
        ("2026-11-17T19:00:00Z", 42, "2026-11-17T19:00:00Z", 1390, Msg.SHIFTED),
        # one created 30 s after the boot and started 50 s after it: not there at the stop, so not the host's alone
        ("2026-11-17T20:20:33Z", 1350, "2026-11-17T19:00:00Z", 1390, Msg.SHIFTED_MIXED),
        # both created after the boot, early and late starts: created or restarted on the running host
        ("2026-11-17T20:20:33Z", 1350, "2026-11-17T20:30:00Z", 42, Msg.SHIFTED_AFTER_CONTAINER_RESTART),
        # creation unknown: early start reads as the host stop, late start as a restart
        ("?", 1350, "?", 42, Msg.SHIFTED_MIXED),
        ("?", 1350, "2026-11-17T19:00:00Z", 1390, Msg.SHIFTED),
        ("?", 42, "2026-11-17T20:20:33Z", 1390, Msg.SHIFTED_AFTER_CONTAINER_RESTART),
    ],
)
@pytest.mark.asyncio
async def test_every_shifted_pod_gets_the_cause_its_creation_and_start_show(
    context_factory, tmp_path, created_1, age_1, created_2, age_2, template
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me + 200000}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me + 200000}      65536\n")
    boot = 1794946803  # 2026-11-17T20:20:03Z, uptime 1400 s
    out = f"1400.25\n4242 {age_1}\n4343 {age_2}\n--\n{boot}\n/pod_1 {created_1}\n/pod_2 {created_2}\n"
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh_with_uptime(str(tmp_path), out)))
    assert result.event.remediation == template.remediation


@pytest.mark.parametrize(
    ("clock", "started_at"),
    [
        # boot epoch and ticks a second both read
        ("1794500000\n100", "2026-11-13T18:00:00Z"),
        # boot time or tick rate not readable: the report says unknown instead of leaving the time out
        ("", "unknown"),
        ("1794500000", "unknown"),
    ],
)
@pytest.mark.asyncio
async def test_a_flagged_runtime_carries_when_it_started_from_the_kernel_start_time(
    context_factory, tmp_path, clock, started_at
):
    """the check first ran days after a restart: the report dates the restart itself, from the kernel's start
    time of the daemon (boot epoch + ticks since boot), not a clock and an age read apart (arhangel66 on
    lio#2379: a 3 s later clock read moved the date across midnight)"""
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    # boot 1794500000 + uptime 100000 = a read at 2026-11-13T20:00:00Z; sysbox-fs started 7200 s before it
    # (2026-11-13T18:00:00Z), pod_1 50000 s before, pod_2 50 s before (after the restart: not flagged)
    ages = "10 1 sysbox-mgr 99000\n11 1 sysbox-fs 7200\n--\n4242 50000\n4343 50\n"
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_runtime_ages(str(tmp_path), ages, clock))
    )
    assert result.event.reason_code == Msg.RUNTIME_RESTARTED.reason
    assert result.event.what_we_saw["runtime_restarted_under_pods"] == {"sysbox-fs": ["pod_1"]}
    assert result.event.what_we_saw["runtime_started_at"] == {"sysbox-fs": started_at}


@pytest.mark.asyncio
async def test_the_runtime_read_runs_in_a_real_shell_and_dates_from_the_kernel(monkeypatch):
    """the command itself, run by bash on this Linux host: a process started after the "pod" (this test
    process) is flagged and dated within seconds of now"""
    import calendar
    import time

    from neurons.validators.src.services.task.checks import rented_pod_ownership as mod

    monkeypatch.setattr(mod, "RUNTIME_PROCESSES", ("sleep",))
    daemon = subprocess.Popen(["sleep", "30"])
    try:

        async def run(cmd, check=False):
            out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
            return SimpleNamespace(exit_status=out.returncode, stdout=out.stdout, stderr=out.stderr)

        restarted, started_at = await RentedPodOwnershipCheck()._runtime_restarted_under(
            SimpleNamespace(ssh=SimpleNamespace(run=run)), [{"pid": os.getpid(), "container": "pod_1"}]
        )
    finally:
        daemon.kill()
    assert restarted == {"sleep": ["pod_1"]}
    started = calendar.timegm(time.strptime(started_at["sleep"], "%Y-%m-%dT%H:%M:%SZ"))
    assert abs(started - time.time()) < 60


# 22 Nov, last read before a host reboot: / /usr /usr/lib already back at host root, sudo files still in the range
_MID_REVERT = "4242|0 231072 65536|0||231072|231072|231072|231072|0|0|231072|231072"


def test_a_pod_whose_files_are_being_handed_back_to_the_host_is_not_shifted():
    pod = judge(_MID_REVERT)
    assert pod["outside_namespace"] == []
    assert pod["being_reverted"] == ["/", "/usr", "/usr/lib"]
    # every path at host root (never shifted into the range) and sudo files alone at host root stay shifted
    assert judge("4242|0 231072 65536|0||0|0|0|0|0|0|0|0")["outside_namespace"]
    assert judge("4242|0 231072 65536|231072|231072|0|0|0|0|231072|231072|231072|231072")["outside_namespace"] == [
        "/etc/sudoers", "/etc/sudoers.d", "/etc/sudoers.d/README", "/usr/bin/sudo",
    ]


@pytest.mark.asyncio
async def test_a_pod_read_mid_stop_is_unread_not_shifted(context_factory):
    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout="/pod_1 4242\n/pod_2 4343\n", stderr="")
        if cmd.startswith("for p in"):
            healthy = "4343|0 231072 65536|231072|231072"
            return SimpleNamespace(exit_status=0, stdout=f"{_MID_REVERT}\n{healthy}\n", stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, SimpleNamespace(run=run)))
    assert result.event.reason_code == Msg.UNREAD.reason
    assert result.event.what_we_saw["being_reverted"] == ["pod_1"]


@pytest.mark.parametrize(
    "line",
    [
        # / at host root beside a real shift is a shift, not a hand-back
        "1|0 231072 65536|0|165536|165536|165536|165536|165536|165536|165536|231072|165536",
        "1|0 231072 65536|0|231072|65534|65534|65534|65534|65534|65534|65534|65534",
        "1|0 231072 65536|0|0|0|0|0|0|0|0|0|231072",
        # / alone at host root: the hand-back reaches /usr and /usr/lib with it
        "1|0 231072 65536|0|231072|231072|231072|231072|231072|231072|231072|231072|231072",
    ],
)
def test_root_at_host_root_beside_anything_but_the_hand_back_shape_is_shifted(line):
    pod = judge(line)
    assert pod["outside_namespace"] and "being_reverted" not in pod


@pytest.mark.parametrize(
    "line",
    [
        # 21-22 Nov reads before a host reboot: the package dirs came back in either order
        "1|0 231072 65536|0||231072|231072|231072|231072|0|0|0|0",
        "1|0 231072 65536|0||231072|231072|231072|231072|0|0|231072|0",
    ],
)
def test_the_package_dirs_may_follow_the_top_dirs_in_the_hand_back(line):
    assert judge(line)["outside_namespace"] == []


def _mid_revert_ssh(pid: int = 4242):
    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout=f"/pod_1 {pid}\n/pod_2 4343\n", stderr="")
        if cmd.startswith("for p in"):
            line = _MID_REVERT.replace("4242|", f"{pid}|", 1)
            return SimpleNamespace(exit_status=0, stdout=f"{line}\n4343|0 231072 65536|231072|231072\n", stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    return SimpleNamespace(run=run)


@pytest.mark.asyncio
async def test_the_hand_back_shape_still_there_next_cycle_on_the_same_pod_is_shifted(context_factory):
    # a hand-back cut short (power lost), then the pod running on: the shape lasts, the pod is not stopping
    services = build_services()
    first = await RentedPodOwnershipCheck().run(_ctx(context_factory, _mid_revert_ssh(), services=services))
    second = await RentedPodOwnershipCheck().run(_ctx(context_factory, _mid_revert_ssh(), services=services))
    assert first.event.reason_code == Msg.UNREAD.reason
    assert second.event.reason_code == Msg.SHIFTED.reason
    pod = next(p for p in second.event.what_we_saw["pods"] if p["container"] == "pod_1")
    assert pod["outside_namespace"] == ["/", "/usr", "/usr/lib"]


def _clean_ssh(pid: int = 4242):
    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout=f"/pod_1 {pid}\n/pod_2 4343\n", stderr="")
        if cmd.startswith("for p in"):
            lines = f"{pid}|0 231072 65536|231072|231072\n4343|0 231072 65536|231072|231072\n"
            return SimpleNamespace(exit_status=0, stdout=lines, stderr="")
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    return SimpleNamespace(run=run)


async def _reasons(context_factory, services, ssh_list):
    return [
        (await RentedPodOwnershipCheck().run(_ctx(context_factory, ssh, services=services))).event.reason_code
        for ssh in ssh_list
    ]


@pytest.mark.asyncio
async def test_a_pod_back_on_a_new_init_pid_still_in_the_hand_back_shape_is_shifted(context_factory):
    # a pod started again into the shape has finished its stop, so the shape lasts (a restart does not
    # repair it); a provider restarting the pod every cycle, or a pod crash-looping on its broken sudo, gets no unread
    reasons = await _reasons(context_factory, build_services(), [_mid_revert_ssh(4242 + i) for i in range(4)])
    assert reasons == [Msg.UNREAD.reason] + [Msg.SHIFTED.reason] * 3


@pytest.mark.asyncio
async def test_a_clean_read_ends_the_stop_so_a_later_hand_back_is_a_new_stop(context_factory):
    # the mark goes with a clean read, so a second honest stop (the pid may even be reused) reads unread again
    services = build_services()
    reasons = await _reasons(
        context_factory, services, [_mid_revert_ssh(4242), _clean_ssh(4242), _mid_revert_ssh(4242)]
    )
    assert reasons == [Msg.UNREAD.reason, Msg.OK.reason, Msg.UNREAD.reason]


@pytest.mark.asyncio
async def test_a_dry_run_reads_the_mark_but_never_writes_it(context_factory, monkeypatch):
    # the dry-run validator shares Redis with the live one, so its read must not turn live's first sight of
    # an honest stop into a second one
    services = build_services()
    monkeypatch.setattr(rented_pod_ownership.settings, "DRY_RUN", True)
    dry = await _reasons(context_factory, services, [_mid_revert_ssh(), _mid_revert_ssh(), _clean_ssh()])
    assert dry == [Msg.UNREAD.reason, Msg.UNREAD.reason, Msg.OK.reason]
    assert services.redis.store == {}
    monkeypatch.setattr(rented_pod_ownership.settings, "DRY_RUN", False)
    assert await _reasons(context_factory, services, [_mid_revert_ssh()]) == [Msg.UNREAD.reason]
    # a live mark the dry run sees is read, and the dry run leaves it in place
    monkeypatch.setattr(rented_pod_ownership.settings, "DRY_RUN", True)
    assert await _reasons(context_factory, services, [_mid_revert_ssh(), _clean_ssh()]) == [
        Msg.SHIFTED.reason, Msg.OK.reason
    ]
    assert len(services.redis.store) == 1


@pytest.mark.asyncio
async def test_the_hand_back_shape_is_shifted_when_redis_cannot_say(context_factory):
    ctx = _ctx(context_factory, _mid_revert_ssh(), services=build_services(redis=FakeRedis(failing=True)))
    result = await RentedPodOwnershipCheck().run(ctx)
    assert result.event.reason_code == Msg.SHIFTED.reason


class _NullSink:
    async def emit(self, event) -> None:
        pass


def _drops_after_the_ownership_read():
    """The mid-revert reads come back, then the session drops under the runtime read, as a recreated executor does."""
    inner = _mid_revert_ssh()
    closed = []

    async def run(cmd, check=False):
        if cmd.startswith("ps -eo"):
            closed.append(True)
            raise asyncssh.ConnectionLost("SSH connection closed")
        return await inner.run(cmd, check)

    return SimpleNamespace(run=run, is_closed=lambda: bool(closed))


@pytest.mark.asyncio
async def test_a_first_sight_run_again_after_a_dropped_session_stays_unread(context_factory):
    # the first sight writes the mark, then the session drops: the pipeline reconnects and runs the check again in the
    # same run, which reads its own mark as the same sight (unread), not a later one
    from neurons.validators.src.services.task.pipeline import Pipeline

    services = build_services()
    reconnects = []

    async def reconnect():
        reconnects.append(1)
        return SimpleNamespace(run=_mid_revert_ssh().run, is_closed=lambda: False)

    ctx = _ctx(context_factory, _drops_after_the_ownership_read(), services=services)
    _, events, _ = await Pipeline([RentedPodOwnershipCheck()], _NullSink(), reconnect=reconnect).run(ctx)
    assert [e.reason_code for e in events] == [Msg.UNREAD.reason] and reconnects == [1]
    # the next run still sees the shape: a lasting state
    assert await _reasons(context_factory, services, [_mid_revert_ssh()]) == [Msg.SHIFTED.reason]


class _SlowRedis(FakeRedis):
    async def mget(self, keys):
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_redis_slower_than_the_budget_leaves_the_hand_back_shifted_in_bounded_time(context_factory, monkeypatch):
    # a pool wait of a minute per call must not eat the validation deadline
    monkeypatch.setattr(rented_pod_ownership, "MID_REVERT_MARKS_BUDGET_SECONDS", 0.05)
    ctx = _ctx(context_factory, _mid_revert_ssh(), services=build_services(redis=_SlowRedis()))
    result = await asyncio.wait_for(RentedPodOwnershipCheck().run(ctx), timeout=5)
    assert result.event.reason_code == Msg.SHIFTED.reason


@pytest.mark.asyncio
async def test_the_marks_of_every_pod_move_in_one_read_and_one_write(context_factory):
    services = build_services()
    await RentedPodOwnershipCheck().run(_ctx(context_factory, _mid_revert_ssh(), services=services))
    assert services.redis.calls == 2


@pytest.mark.asyncio
async def test_a_candidate_and_a_clean_pod_move_their_marks_in_the_same_batch(context_factory):
    services = build_services()
    gone = f"{rented_pod_ownership.MID_REVERT_KEY_PREFIX}:{_ctx(context_factory, None).executor.uuid}:pod_2"
    services.redis.store[gone] = "an-earlier-run"
    await RentedPodOwnershipCheck().run(_ctx(context_factory, _mid_revert_ssh(), services=services))
    # pod_1 in the hand-back shape is marked, clean pod_2 loses its old mark
    assert [k.rsplit(":", 1)[1] for k in services.redis.store] == ["pod_1"]


@pytest.mark.asyncio
async def test_names_the_node_was_not_asked_about_are_ignored(context_factory):
    inner = _mid_revert_ssh()

    async def run(cmd, check=False):
        if cmd.startswith("/usr/bin/docker inspect"):
            forged = "".join(f"/forged_{i} {5000 + i}\n" for i in range(1000))
            return SimpleNamespace(exit_status=0, stdout="/pod_1 4242\n/pod_2 4343\n" + forged, stderr="")
        if cmd.startswith("for p in"):
            assert "5000" not in cmd
        return await inner.run(cmd, check)

    services = build_services()
    await RentedPodOwnershipCheck().run(_ctx(context_factory, SimpleNamespace(run=run), services=services))
    assert len(services.redis.store) == 1
