"""get_egress_ip() in machine_scrape.py: the host's own outbound address from Cloudflare's trace,
recorded as specs.egress_ip so the backend can list a node behind a reverse proxy at the host's
location, not the proxy's. The trace runs in a short-lived container on a running pod's network
(else the network pods are created on), from the executor's own image; on any failure in the host
network namespace. Never inside a pod, never in the executor container's own route. Any failure
reads "" and never fails the scrape."""

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
EXECUTOR_IMAGE = "sha256:executor"
PINNED = ["-q", "--noproxy", "*"]


class _Fake:
    stdout = TRACE
    fail: set = set()
    calls: list = []
    containers: list | None = []
    pod_mode = "lium-rentals"
    pod_networks: dict = {}

    @classmethod
    def run(cls, cmd, **kwargs):
        cls.calls.append((cmd, kwargs))
        if cmd[0] in cls.fail:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return SimpleNamespace(stdout=cls.stdout)

    @classmethod
    def docker_api_get(cls, path):
        if path == "/containers/json":
            return cls.containers
        if path == "/containers/p1/json":
            return {
                "HostConfig": {"NetworkMode": cls.pod_mode},
                "NetworkSettings": {"Networks": cls.pod_networks},
                "State": {"Pid": 4242},
            }
        assert path == f"/containers/{socket.gethostname()}/json"
        return {"Image": EXECUTOR_IMAGE}

    @staticmethod
    def getaddrinfo(host, port, family, kind):
        assert (host, port) == ("www.cloudflare.com", 443)
        return [(family, kind, 6, "", (TRACE_ADDR, port))]


@pytest.fixture
def get_egress_ip():
    fake_subprocess = SimpleNamespace(run=_Fake.run)
    fake_socket = SimpleNamespace(
        getaddrinfo=_Fake.getaddrinfo, AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM,
        gethostname=socket.gethostname,
    )
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {
            "EGRESS_TRACE_HOST", "EGRESS_TRACE_PATH", "EGRESS_TRACE_TIMEOUT_SECONDS",
            "EGRESS_PROBE_START_SECONDS", "EGRESS_PROBE_NAME_PREFIX", "EGRESS_RENTAL_NETWORK",
            "get_egress_probe_network", "get_egress_ip",
        },
        {"subprocess": fake_subprocess, "socket": fake_socket, "os": os, "docker_api_get": _Fake.docker_api_get},
    )
    yield ns["get_egress_ip"]
    _Fake.stdout, _Fake.fail, _Fake.calls = TRACE, set(), []
    _Fake.containers, _Fake.pod_mode, _Fake.pod_networks = [], "lium-rentals", {}


@pytest.fixture
def with_pod():
    _Fake.containers = [{"Id": "e1", "Names": ["/executor"]}, {"Id": "p1", "Names": ["/pod_abc"]}]


def _probe_network(cmd):
    return cmd[cmd.index("--network") + 1]


def test_reads_the_ip_line(get_egress_ip) -> None:
    assert get_egress_ip() == "203.0.113.7"


def test_trace_runs_in_a_probe_on_the_running_pod_network(get_egress_ip, with_pod) -> None:
    _Fake.pod_mode = "some-net"
    assert get_egress_ip() == "203.0.113.7"
    [(cmd, kwargs)] = _Fake.calls
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--name") + 1].startswith("lium_egress_probe_")
    assert _probe_network(cmd) == "some-net"
    # the executor's own image (by id, so nothing is pulled), never the pod's namespace
    start = cmd.index("--entrypoint")
    assert cmd[start:start + 6] == ["--entrypoint", "curl", EXECUTOR_IMAGE, *PINNED]
    assert "4242" not in cmd and "nsenter" not in cmd
    assert f"www.cloudflare.com:443:{TRACE_ADDR}" in cmd
    assert cmd[-1] == "https://www.cloudflare.com/cdn-cgi/trace"
    assert kwargs["check"] is True and kwargs["timeout"]
    assert "LD_LIBRARY_PATH" not in kwargs["env"]


def test_default_bridge_pod_probes_on_bridge(get_egress_ip, with_pod) -> None:
    _Fake.pod_mode = "default"
    get_egress_ip()
    assert _probe_network(_Fake.calls[0][0]) == "bridge"


@pytest.mark.parametrize("mode", ["default", "bridge"])
def test_default_mode_pod_moved_to_another_network_probes_there(get_egress_ip, with_pod, mode) -> None:
    _Fake.pod_mode, _Fake.pod_networks = mode, {"lium-rentals": {}}
    get_egress_ip()
    assert _probe_network(_Fake.calls[0][0]) == "lium-rentals"


def test_no_pod_probes_on_the_rental_network(get_egress_ip) -> None:
    get_egress_ip()
    assert _probe_network(_Fake.calls[0][0]) == "lium-rentals"


@pytest.mark.parametrize("mode", ["container:p2", "none"])
def test_pod_in_a_shared_namespace_uses_the_host(get_egress_ip, with_pod, mode) -> None:
    _Fake.pod_mode = mode
    assert get_egress_ip() == "203.0.113.7"
    assert [cmd[:5] for cmd, _ in _Fake.calls] == [["nsenter", "-t", "1", "-n", "curl"]]


def test_probe_failure_removes_it_and_falls_back_to_the_host(get_egress_ip) -> None:
    _Fake.fail = {"docker"}
    assert get_egress_ip() == "203.0.113.7"
    probe, remove, host = [cmd for cmd, _ in _Fake.calls]
    assert remove == ["docker", "rm", "-f", probe[probe.index("--name") + 1]]
    assert host[:8] == ["nsenter", "-t", "1", "-n", "curl", *PINNED]


def test_docker_api_failure_uses_the_host(get_egress_ip) -> None:
    _Fake.containers = None
    assert get_egress_ip() == "203.0.113.7"
    assert [cmd[0] for cmd, _ in _Fake.calls] == ["nsenter"]


def test_no_ip_line_is_empty(get_egress_ip) -> None:
    _Fake.stdout = "h=www.cloudflare.com\n"
    assert get_egress_ip() == ""


def test_every_failure_is_empty_not_the_container_route(get_egress_ip) -> None:
    _Fake.fail = {"docker", "nsenter"}
    assert get_egress_ip() == ""
    assert [cmd[0] for cmd, _ in _Fake.calls] == ["docker", "docker", "nsenter"]


def test_scrape_keys_are_mapped_for_the_backend() -> None:
    service = (SRC / "services" / "file_encrypt_service.py").read_text()
    assert "'data_egress_ip': \"egress_ip\"" in service
    assert "'data_egress_ip': \"\"" in service
