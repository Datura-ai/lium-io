"""RentedPodOwnershipCheck: warn when a rented sysbox pod's / or /root is owned outside its user namespace."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.checks.rented_pod_ownership import (
    RentedPodOwnershipCheck,
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
    _proc(tmp_path, 4242, "         0          0 4294967295\n")  # no user namespace: never judged
    _proc(tmp_path, 4343, f"         0     {me}      65536\n")
    result = await RentedPodOwnershipCheck().run(_ctx(context_factory, _ssh(str(tmp_path))))
    assert result.event.reason_code == Msg.OK.reason
    assert [p["container"] for p in result.event.what_we_saw["pods"]] == ["pod_2"]


def test_read_command_reports_a_missing_pid_without_failing(tmp_path):
    out = subprocess.run(
        ["sh", "-c", read_command([999], proc=str(tmp_path))], capture_output=True, text=True
    )
    assert out.stdout == "999|||\n"
