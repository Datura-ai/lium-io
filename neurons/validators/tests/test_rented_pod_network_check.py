"""DAH-5264 — RentedPodNetworkCheck: a per-pod summary of tracked connections, read on the host; never fails."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import asyncssh
import pytest
from neurons.validators.src.services.task.checks.rented_pod_network import (
    RentedPodNetworkCheck,
    parse_summary,
    read_command,
)
from neurons.validators.src.services.task.messages import RentedPodNetworkMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_state

SETTINGS = "neurons.validators.src.services.task.checks.rented_pod_network.settings"

TABLE = """\
ipv4     2 tcp      6 117 SYN_SENT src=172.17.0.2 dst=10.0.0.1 sport=40001 dport=2083 [UNREPLIED] src=10.0.0.1 dst=192.168.1.5 sport=2083 dport=40001 mark=0 zone=0 use=2
ipv4     2 tcp      6 117 SYN_SENT src=172.17.0.2 dst=10.0.0.2 sport=40002 dport=2083 [UNREPLIED] src=10.0.0.2 dst=192.168.1.5 sport=2083 dport=40002 mark=0 zone=0 use=2
ipv4     2 tcp      6 431999 ESTABLISHED src=172.17.0.2 dst=151.101.0.223 sport=40003 dport=443 src=151.101.0.223 dst=192.168.1.5 sport=443 dport=40003 [ASSURED] mark=0 zone=0 use=2
ipv4     2 udp      17 29 src=172.17.0.2 dst=10.0.0.3 sport=5000 dport=53 [UNREPLIED] src=10.0.0.3 dst=192.168.1.5 sport=53 dport=5000 mark=0 zone=0 use=2
ipv4     2 tcp      6 300 ESTABLISHED src=172.17.0.9 dst=1.1.1.1 sport=1 dport=443 src=1.1.1.1 dst=172.17.0.9 sport=443 dport=1 [ASSURED] mark=0 zone=0 use=2
"""
TABLE += "".join(
    f"ipv4 2 tcp 6 117 SYN_SENT src=172.17.0.2 dst=10.1.0.{i} sport=4{i:04d} dport=22 [UNREPLIED] "
    f"src=10.1.0.{i} dst=192.168.1.5 sport=22 dport=4{i:04d} mark=0 zone=0 use=2\n"
    for i in range(12)
)


def test_read_command_summarises_a_real_table(tmp_path):
    table = tmp_path / "nf_conntrack"
    table.write_text(TABLE)
    out = subprocess.run(
        ["sh", "-c", read_command(["172.17.0.2", "172.17.0.4"], table=str(table))],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pods = parse_summary(out, {"172.17.0.2": "pod_a", "172.17.0.4": "pod_b"})
    a, b = pods
    assert a == {
        "container": "pod_a",
        "flows": 16,
        "unreplied_flows": 15,
        "destinations": 16,
        "unreplied_destinations": 15,
        "top_unreplied_ports": [{"port": "22", "flows": 12}],
    }
    assert b["flows"] == 0 and b["container"] == "pod_b"


def test_read_command_bounds_its_state_and_marks_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.task.checks.rented_pod_network._MAX_KEYS", 5
    )
    table = tmp_path / "nf_conntrack"
    table.write_text(TABLE)
    out = subprocess.run(
        ["sh", "-c", read_command(["172.17.0.2"], table=str(table))],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.startswith("T\n")
    (pod,) = parse_summary(out, {"172.17.0.2": "pod_a"})
    assert pod["flows"] == 16
    assert pod["destinations"] + pod["unreplied_destinations"] <= 5


@pytest.mark.asyncio
async def test_truncated_summary_is_flagged(context_factory):
    result = await _run(_ctx(context_factory, _ssh("T\nS 172.17.0.2 9 3 5 3\n")))
    assert result.event.what_we_saw["truncated"] is True


def test_read_command_without_table_exits_3(tmp_path):
    res = subprocess.run(["sh", "-c", read_command(["172.17.0.2"], table=str(tmp_path / "x"))])
    assert res.returncode == 3


def _ssh(summary="", table_exit=0, inspect="/pod_1 172.17.0.2\n", raise_exc=None):
    async def run(cmd, *args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout=inspect, stderr="")
        return SimpleNamespace(exit_status=table_exit, stdout=summary, stderr="")

    return SimpleNamespace(run=run)


def _ctx(context_factory, ssh, pods=True, pid_mode="host"):
    rented = RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="1", container_name="pod_1", status="RUNNING")]
                if pods
                else [],
            )
        }
    )
    specs = {"executor_pid_mode": pid_mode} if pid_mode is not None else {}
    return context_factory(state=build_state(rented_data=rented, specs=specs), ssh=ssh)


async def _run(ctx, limit=256):
    with patch(SETTINGS) as s:
        s.RENTED_POD_OUTBOUND_FANOUT_DESTINATIONS = limit
        return await RentedPodNetworkCheck().run(ctx)


@pytest.mark.asyncio
async def test_summary_below_limit(context_factory):
    result = await _run(_ctx(context_factory, _ssh("S 172.17.0.2 9 3 5 3\n")))
    assert result.passed is True
    assert result.event.reason_code == Msg.SUMMARY.reason
    assert result.event.what_we_saw["pods"][0]["unreplied_destinations"] == 3


@pytest.mark.asyncio
async def test_every_summary_is_logged_at_info(context_factory, caplog):
    with caplog.at_level("INFO"):
        await _run(_ctx(context_factory, _ssh("S 172.17.0.2 9 3 5 3\n")))
        await _run(_ctx(context_factory, _ssh("S 172.17.0.2 4 1 2 1\n")))
    logged = [
        r.msg.extra["pods"][0]["flows"]
        for r in caplog.records
        if r.levelname == "INFO" and getattr(r.msg, "message", None) == Msg.SUMMARY.event
    ]
    assert logged == [9, 4]


@pytest.mark.asyncio
async def test_many_unreplied_destinations_warn_but_pass(context_factory):
    result = await _run(
        _ctx(context_factory, _ssh("S 172.17.0.2 900 800 850 800\nP 172.17.0.2 2083 700\n"))
    )
    assert result.passed is True
    assert result.event.reason_code == Msg.FANOUT.reason
    assert result.event.what_we_saw["pods"][0]["top_unreplied_ports"] == [
        {"port": "2083", "flows": 700}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh",
    [
        _ssh(table_exit=3),
        _ssh(inspect="/pod_1 <no value>\n"),
        _ssh(raise_exc=asyncssh.ConnectionLost("gone")),
    ],
)
async def test_unreadable_never_fails(context_factory, ssh):
    result = await _run(_ctx(context_factory, ssh))
    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("pid_mode", ["", "private", None])
async def test_executor_outside_host_pid_namespace_is_unread(context_factory, pid_mode):
    result = await _run(_ctx(context_factory, _ssh("S 172.17.0.2 9 3 5 3\n"), pid_mode=pid_mode))
    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason


@pytest.mark.asyncio
async def test_not_rented(context_factory):
    result = await _run(_ctx(context_factory, _ssh(), pods=False))
    assert result.event.reason_code == Msg.NOT_RENTED.reason
