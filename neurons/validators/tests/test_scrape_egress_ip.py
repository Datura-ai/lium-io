"""get_egress_ip() in machine_scrape.py: the host's own outbound address from Cloudflare's trace,
recorded as specs.egress_ip so the backend can list a node behind a reverse proxy at the host's
location, not the proxy's. The trace runs in a running renter pod's network namespace (where its
traffic leaves), else in the host's, never in the executor container's. Any failure reads "" and
never fails the scrape."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
TRACE = "fl=1f1\nh=www.cloudflare.com\nip=203.0.113.7\nts=1.2\nloc=DE\ncolo=FRA\n"
TRACE_ADDR = "192.0.2.10"


POD_PID = 4242


class _Fake:
    stdout = TRACE
    fail = False
    fail_pids: set = set()
    calls: list = []
    containers: list = [{"Id": "e1", "Names": ["/executor"]}]

    @classmethod
    def run(cls, cmd, **kwargs):
        cls.calls.append((cmd, kwargs))
        if cls.fail or int(cmd[2]) in cls.fail_pids:
            raise subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(stdout=cls.stdout)

    @classmethod
    def docker_api_get(cls, path):
        if path == "/containers/json":
            return cls.containers
        assert path == "/containers/p1/json"
        return {"State": {"Pid": POD_PID}}

    @staticmethod
    def getaddrinfo(host, port, family, kind):
        assert (host, port) == ("www.cloudflare.com", 443)
        return [(family, kind, 6, "", (TRACE_ADDR, port))]


@pytest.fixture
def get_egress_ip():
    fake_subprocess = SimpleNamespace(run=_Fake.run)
    fake_socket = SimpleNamespace(
        getaddrinfo=_Fake.getaddrinfo, AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM
    )
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"EGRESS_TRACE_HOST", "EGRESS_TRACE_PATH", "EGRESS_TRACE_TIMEOUT_SECONDS", "get_egress_ip"},
        {"subprocess": fake_subprocess, "socket": fake_socket, "os": os, "docker_api_get": _Fake.docker_api_get},
    )
    yield ns["get_egress_ip"]
    _Fake.stdout, _Fake.fail, _Fake.fail_pids, _Fake.calls = TRACE, False, set(), []
    _Fake.containers = [{"Id": "e1", "Names": ["/executor"]}]


@pytest.fixture
def with_pod():
    _Fake.containers = [{"Id": "e1", "Names": ["/executor"]}, {"Id": "p1", "Names": ["/pod_abc"]}]


def test_reads_the_ip_line(get_egress_ip) -> None:
    assert get_egress_ip() == "203.0.113.7"


def test_trace_runs_in_the_host_network_namespace(get_egress_ip) -> None:
    get_egress_ip()
    [(cmd, kwargs)] = _Fake.calls
    assert cmd[:5] == ["nsenter", "-t", "1", "-n", "curl"]
    # a proxy in the executor (env or ~/.curlrc) would report its own IP again
    assert cmd[5:8] == ["-q", "--noproxy", "*"]
    assert f"www.cloudflare.com:443:{TRACE_ADDR}" in cmd
    assert cmd[-1] == "https://www.cloudflare.com/cdn-cgi/trace"
    assert kwargs["check"] is True and kwargs["timeout"]
    assert "LD_LIBRARY_PATH" not in kwargs["env"]


def test_no_ip_line_is_empty(get_egress_ip) -> None:
    _Fake.stdout = "h=www.cloudflare.com\n"
    assert get_egress_ip() == ""


def test_host_namespace_failure_is_empty_not_the_container_route(get_egress_ip) -> None:
    _Fake.fail = True
    assert get_egress_ip() == ""
    assert len(_Fake.calls) == 1


def test_trace_runs_in_a_running_pod_network_namespace(get_egress_ip, with_pod) -> None:
    assert get_egress_ip() == "203.0.113.7"
    [(cmd, kwargs)] = _Fake.calls
    assert cmd[:8] == ["nsenter", "-t", str(POD_PID), "-n", "curl", "-q", "--noproxy", "*"]
    assert f"www.cloudflare.com:443:{TRACE_ADDR}" in cmd
    assert "LD_LIBRARY_PATH" not in kwargs["env"]


def test_pod_namespace_failure_falls_back_to_the_host(get_egress_ip, with_pod) -> None:
    _Fake.fail_pids = {POD_PID}
    assert get_egress_ip() == "203.0.113.7"
    assert [cmd[2] for cmd, _ in _Fake.calls] == [str(POD_PID), "1"]


def test_docker_failure_uses_the_host(get_egress_ip) -> None:
    _Fake.containers = None
    assert get_egress_ip() == "203.0.113.7"
    assert [cmd[2] for cmd, _ in _Fake.calls] == ["1"]


def test_scrape_keys_are_mapped_for_the_backend() -> None:
    service = (SRC / "services" / "file_encrypt_service.py").read_text()
    assert "'data_egress_ip': \"egress_ip\"" in service
    assert "'data_egress_ip': \"\"" in service
