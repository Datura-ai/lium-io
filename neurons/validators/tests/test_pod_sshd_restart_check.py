"""PodSshdRestartCheck: re-run the sshd bootstrap in a rented pod with SSH down, only if the renter never logged in."""

from __future__ import annotations

import re
import subprocess
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.checks import pod_sshd_restart
from neurons.validators.src.services.task.checks.pod_sshd_restart import PodSshdRestartCheck
from neurons.validators.src.services.task.checks.rented_pod_ssh import ConnectOutcome
from neurons.validators.src.services.task.messages import PodSshdRestartMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from protocol.vc_protocol.validator_requests import PodSshResult

from tests.helpers import build_state

PORTS = {"1": 40001, "2": 40002, "3": 40003}


def _pod_fs(root, *, watchdog: str | None, cmdline: bytes = b"", marker=False, wtmp=b""):
    """A pod's filesystem: the create's watchdog pidfile, that pid's title, the login marker and wtmp."""
    for d in ("run", "proc/77", "var/lib/lium", "var/log", "tmp"):
        (root / d).mkdir(parents=True, exist_ok=True)
    if watchdog is not None:
        (root / "run" / "sshd-watchdog.pid").write_text(watchdog)
    (root / "proc" / "77" / "cmdline").write_bytes(cmdline)
    if marker:
        (root / "var/lib/lium/ssh-login-seen").touch()
    (root / "var/log/wtmp").write_bytes(wtmp)


def _ssh(fs_by_container: dict, ran: list):
    async def run(cmd, check=False):
        name = next(n for n in fs_by_container if f" -i {n} sh -c " in cmd)
        root = str(fs_by_container[name])
        # the real gate, with the pod's paths under its fake root and docker exec dropped
        cmd = cmd.replace(f"/usr/bin/docker exec -u 0 -i {name} ", "")
        cmd = re.sub(r"(?<![\w/])(/run/|/proc/|/var/lib/lium/|/var/log/wtmp|/tmp/)", lambda m: root + m[1], cmd)
        out = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
        if "bootstrap ran" in out.stdout:
            ran.append(name)
        return SimpleNamespace(exit_status=out.returncode, stdout=out.stdout, stderr=out.stderr)

    return SimpleNamespace(run=run)


def _ctx(context_factory, ssh, manual=False):
    rented = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[
                    RentedPod(pod_id=i, container_name=f"pod_{i}", status="RUNNING", ssh_port=port)
                    for i, port in PORTS.items()
                ],
            )
        },
        manual_rental_executors={"executor-123": {"gpu_model": "H100", "gpu_count": 1}} if manual else {},
    )
    return context_factory(state=build_state(rented_data=rented), ssh=ssh)


@pytest.fixture
def probe(monkeypatch, tmp_path):
    script = tmp_path / "bootstrap.sh"
    script.write_text("echo bootstrap ran\n")
    monkeypatch.setattr(pod_sshd_restart, "_BOOTSTRAP_SCRIPT", script)
    results = {port: PodSshResult.REFUSED for port in PORTS.values()}

    async def fake(host, port, timeout):
        return ConnectOutcome(results[port])

    monkeypatch.setattr(pod_sshd_restart, "tcp_connect_fault", fake)
    return results


@pytest.mark.asyncio
async def test_restarts_sshd_only_where_the_renter_never_logged_in(context_factory, tmp_path, probe):
    fs = {f"pod_{i}": tmp_path / f"pod_{i}" for i in PORTS}
    _pod_fs(fs["pod_1"], watchdog="77", cmdline=b"bash\x00")  # watchdog gone (pid reused), never logged in
    _pod_fs(fs["pod_2"], watchdog="77", marker=True)  # the watchdog saw a login
    _pod_fs(fs["pod_3"], watchdog="77", wtmp=b"\x07" * 384)  # a wtmp login record
    ran: list = []
    result = await PodSshdRestartCheck().run(_ctx(context_factory, _ssh(fs, ran)))
    assert result.passed is True
    assert result.event.reason_code == Msg.RESTARTED.reason
    assert ran == ["pod_1"]
    assert result.event.what_we_saw["pods"] == {
        "1": "restarted",
        "2": "renter logged in",
        "3": "renter logged in",
    }


@pytest.mark.asyncio
async def test_leaves_pods_with_a_live_watchdog_or_no_lium_bootstrap(context_factory, tmp_path, probe):
    fs = {f"pod_{i}": tmp_path / f"pod_{i}" for i in PORTS}
    _pod_fs(fs["pod_1"], watchdog="77", cmdline=b"sh\x00/tmp/lium-ssh-bootstrap.sh\x00--watchdog-loop\x00")
    _pod_fs(fs["pod_2"], watchdog=None)  # the image starts its own sshd
    _pod_fs(fs["pod_3"], watchdog="77")
    probe[PORTS["3"]] = PodSshResult.BANNER  # answering: not touched
    ran: list = []
    result = await PodSshdRestartCheck().run(_ctx(context_factory, _ssh(fs, ran)))
    assert ran == []
    assert result.event.reason_code == Msg.OK.reason
    assert result.event.what_we_saw["pods"] == {
        "1": "watchdog running",
        "2": "not bootstrapped by lium",
    }


@pytest.mark.asyncio
async def test_timeouts_and_manual_rentals_are_left_alone(context_factory, tmp_path, probe):
    fs = {f"pod_{i}": tmp_path / f"pod_{i}" for i in PORTS}
    for root in fs.values():
        _pod_fs(root, watchdog="77")
    ran: list = []
    result = await PodSshdRestartCheck().run(_ctx(context_factory, _ssh(fs, ran), manual=True))
    assert ran == [] and result.event.what_we_saw["pods"] == {}
    for port in PORTS.values():
        probe[port] = PodSshResult.TIMEOUT
    result = await PodSshdRestartCheck().run(_ctx(context_factory, _ssh(fs, ran)))
    assert ran == [] and result.event.what_we_saw["pods"] == {}
