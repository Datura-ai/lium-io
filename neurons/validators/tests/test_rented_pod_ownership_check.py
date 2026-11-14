"""RentedPodOwnershipCheck: warn when a rented sysbox pod's / or /root is owned outside its user namespace."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.checks.rented_pod_ownership import (
    RentedPodOwnershipCheck,
    judge,
    read_command,
)
from neurons.validators.src.services.task.messages import RentedPodOwnershipMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_state


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


def _ctx(context_factory, ssh):
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
    return context_factory(
        state=build_state(rented_data=rented, specs={"executor_pid_mode": "host"}), ssh=ssh
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
    assert out.stdout == "999|||||||\n"


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


def _ssh_with_runtime_ages(proc: str, ages: str):
    inner = _ssh(proc)

    async def run(cmd, check=False):
        if cmd.startswith("ps -eo comm=,etimes="):
            return SimpleNamespace(exit_status=0, stdout=ages, stderr="")
        return await inner.run(cmd, check=check)

    return SimpleNamespace(run=run)


@pytest.mark.parametrize(
    ("ages", "restarted"),
    [
        # needrestart restarted sysbox-fs and the fabric manager 100 s ago; pod_1 runs for a day, pod_2 for 50 s
        (
            "sysbox-mgr 90000\nsysbox-fs     100\nnv-fabricmanage 100\n--\n 4242 86400\n 4343 50\n",
            {"nv-fabricmanager": ["pod_1"], "sysbox-fs": ["pod_1"]},
        ),
        # every runtime process is older than both pods (a reboot brings them up first): nothing to flag
        ("sysbox-mgr 90000\nsysbox-fs 90000\n--\n4242 86400\n4343 50\n", None),
        # a pod age not read flags nothing for that pod
        ("sysbox-fs 100\n--\n4343 50\n", None),
    ],
)
@pytest.mark.asyncio
async def test_a_runtime_process_younger_than_a_pod_is_listed_without_changing_the_verdict(
    context_factory, tmp_path, ages, restarted
):
    me = os.getuid()
    _proc(tmp_path, 4242, f"         0     {me}      65536\n")
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    result = await RentedPodOwnershipCheck().run(
        _ctx(context_factory, _ssh_with_runtime_ages(str(tmp_path), ages))
    )
    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason
    assert result.event.what_we_saw.get("runtime_restarted_under_pods") == restarted
