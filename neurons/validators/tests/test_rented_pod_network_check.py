"""DAH-5264 — RentedPodNetworkCheck: a per-pod summary of tracked connections, read on the host; never fails."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import asyncssh
import pytest
from neurons.validators.src.services.task.checks.rented_pod_network import (
    _SOCKET_AWK,
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
    result = subprocess.run(
        ["sh", "-c", read_command(["172.17.0.2", "172.17.0.4"], table=str(table))],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr == "SOURCE conntrack\n"
    pods = parse_summary(result.stdout, {"172.17.0.2": "pod_a", "172.17.0.4": "pod_b"})
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


def test_read_command_keeps_the_busiest_port_past_500_summary_rows(tmp_path):
    rows = [(port, 10) for port in range(1000, 1600)] + [(2000, 1000)]
    table = tmp_path / "nf_conntrack"
    table.write_text(
        "".join(
            f"ipv4 2 tcp 6 117 SYN_SENT src=172.17.0.2 dst=10.2.{i // 250}.{i % 250} sport=1 dport={port} "
            f"[UNREPLIED] src=10.2.0.1 dst=192.168.1.5 sport={port} dport=1 mark=0 zone=0 use=2\n"
            for port, n in rows
            for i in range(n)
        )
    )
    out = subprocess.run(
        ["sh", "-c", read_command(["172.17.0.2"], table=str(table))],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len(out.splitlines()) > 500
    (pod,) = parse_summary(out, {"172.17.0.2": "pod_a"})
    assert pod["top_unreplied_ports"][0] == {"port": "2000", "flows": 1000}


def test_read_command_keeps_a_read_failure():
    # readable, but reading it fails with an I/O error
    res = subprocess.run(
        ["sh", "-c", read_command(["172.17.0.2"], table="/proc/self/mem")], capture_output=True
    )
    assert res.returncode not in (0, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_status", "reason"),
    [
        (0, Msg.SUMMARY.reason),
        (1, Msg.UNREAD.reason),
        (2, Msg.UNREAD.reason),
        (127, Msg.UNREAD.reason),
        (-1, Msg.UNREAD.reason),
    ],
)
async def test_only_a_successful_read_is_a_summary(context_factory, exit_status, reason):
    # an empty table read successfully is zero traffic; a failed or killed read is no evidence
    result = await _run(_ctx(context_factory, _ssh("", table_exit=exit_status)))
    assert result.passed is True
    assert result.event.reason_code == reason


@pytest.mark.asyncio
async def test_truncated_summary_is_flagged(context_factory):
    result = await _run(_ctx(context_factory, _ssh("T\nS 172.17.0.2 9 3 5 3\n")))
    assert result.event.what_we_saw["truncated"] is True


def _ssh(
    summary="",
    table_exit=0,
    inspect="/pod_1 123 172.17.0.2\n",
    raise_exc=None,
    stderr="SOURCE conntrack\n",
):
    async def run(cmd, *args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        if cmd.startswith("/usr/bin/docker inspect"):
            return SimpleNamespace(exit_status=0, stdout=inspect, stderr="")
        return SimpleNamespace(exit_status=table_exit, stdout=summary, stderr=stderr)

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
    assert result.event.what_we_saw["source"] == "conntrack"
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


@pytest.fixture
def socket_tables(tmp_path):
    root = tmp_path / "proc"
    for pid in (1, 123):
        (root / str(pid) / "ns").mkdir(parents=True)
        (root / str(pid) / "ns/net").symlink_to(f"net:[{pid}]")
    net = root / "123/net"
    net.mkdir()
    (root / "1/net").mkdir()
    rows = [("tcp", f"{i:02X}0200C0", "02", "0016") for i in range(1, 41)]
    rows += [
        ("tcp", "08080808", "01", "01BB"),
        ("tcp", "09090909", "0A", "0016"),
        ("tcp", "00000000", "02", "0016"),
        ("tcp6", "0000000000000000FFFF0000010200C0", "02", "0016"),
        ("tcp6", "B80D0120000000000000000001000000", "01", "01BB"),
        ("tcp6", "b80d0120000000000000000001000000", "01", "01BB"),
        ("tcp6", "0" * 32, "02", "0016"),
        ("udp", "016433C6", "07", "0035"),
        ("udp6", "B80D0120000000000000000002000000", "01", "0035"),
    ]
    for table in ("tcp", "tcp6", "udp", "udp6"):
        (root / "1/net" / table).touch()
        (net / table).write_text(
            "  sl local_address rem_address st\n"
            + "".join(
                f" {i}: 00000000:1234 {address}:{port} {state} 0:0 0:0 0 0\n"
                for i, (kind, address, state, port) in enumerate(rows)
                if kind == table
            )
        )
    return root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["summary", "ipv6_disabled", "host_network", "not_read", "no_pid", "truncated"]
)
async def test_socket_source(context_factory, socket_tables, monkeypatch, mode):
    root = socket_tables
    pid = 123
    if mode == "host_network":
        (root / "123/ns/net").unlink()
        (root / "123/ns/net").symlink_to("net:[1]")
    elif mode == "not_read":
        (root / "123/net/tcp6").unlink()
    elif mode == "ipv6_disabled":
        for table_pid in (1, 123):
            for table in ("tcp6", "udp6"):
                (root / str(table_pid) / "net" / table).unlink()
    elif mode == "no_pid":
        pid = 0
    elif mode == "truncated":
        monkeypatch.setattr(
            "neurons.validators.src.services.task.checks.rented_pod_network._MAX_KEYS", 5
        )
    command = read_command(
        [], table=str(root / "absent"), pids_by_name={"pod_1": pid}, proc_root=str(root)
    ).replace("awk -v m=", "mawk -v m=")
    remote = subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)
    calls = []
    ssh = _ssh(remote.stdout, inspect=f"/pod_1 {pid}\n", stderr=remote.stderr)

    async def run(cmd, **kwargs):
        calls.append(cmd)
        return await ssh.run(cmd, **kwargs)

    result = await _run(_ctx(context_factory, SimpleNamespace(run=run)), limit=40)
    assert result.passed is True
    assert len(calls) == 2 and "{{.State.Pid}}" in calls[0]
    assert f"pod_1:{pid}" in calls[1]
    event = result.event.what_we_saw
    assert event["source"] == "sockets"
    assert "TCP SYN_SENT only" in event["unreplied_scope"]
    if mode in ("host_network", "not_read", "no_pid"):
        assert result.event.reason_code == Msg.UNREAD.reason
        assert event["skipped_pods"] == [{"container": "pod_1", "reason": mode}]
        assert "pods" not in event and remote.stdout == ""
    elif mode == "truncated":
        assert event["truncated"] is True and remote.stdout.startswith("T\n")
        pod = event["pods"][0]
        assert pod["flows"] == 46
        assert pod["destinations"] + pod["unreplied_destinations"] <= 5
    else:
        assert result.event.reason_code == Msg.FANOUT.reason
        flows, unreplied, destinations = (42, 40, 42) if mode == "ipv6_disabled" else (46, 41, 44)
        assert set(remote.stdout.splitlines()) == {
            f"S pod_1 {flows} {unreplied} {destinations} 40",
            f"P pod_1 22 {unreplied}",
        }
        assert event["pods"][0]["top_unreplied_ports"] == [{"port": "22", "flows": unreplied}]


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        ("016433C6", "198.51.100.1"),
        ("B80D012078563412EFCDAB9001000000", "2001:0db8:1234:5678:90ab:cdef:0000:0001"),
        ("0000000000000000FFFF0000010200C0", "192.0.2.1"),
        ("0000000000000000FFFF000000000000", ""),
        ("00000000", ""),
        ("0" * 32, ""),
    ],
)
def test_socket_hex_decoding(encoded, decoded):
    result = subprocess.run(
        ["mawk", "-v", f"encoded={encoded}", _SOCKET_AWK + "BEGIN{print address(encoded)}"],
        input="",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == decoded
