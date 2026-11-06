"""get_egress_ip() in machine_scrape.py: the host's own outbound address from Cloudflare's trace,
recorded as specs.egress_ip so the backend can list a node behind a reverse proxy at the host's
location, not the proxy's. Any failure reads "" and never fails the scrape."""

from __future__ import annotations

from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
TRACE = "fl=1f1\nh=www.cloudflare.com\nip=203.0.113.7\nts=1.2\nloc=DE\ncolo=FRA\n"


class _Conn:
    body = TRACE.encode()
    fail = False

    def __init__(self, host, timeout):
        assert (host, timeout) == ("www.cloudflare.com", 5)

    def request(self, method, path):
        if self.fail:
            raise OSError("unreachable")
        assert (method, path) == ("GET", "/cdn-cgi/trace")

    def getresponse(self):
        return self

    def read(self, limit):
        return self.body[:limit]

    def close(self):
        pass


@pytest.fixture
def get_egress_ip():
    fake_http = type("H", (), {"client": type("C", (), {"HTTPSConnection": _Conn})})
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"EGRESS_TRACE_HOST", "EGRESS_TRACE_PATH", "EGRESS_TRACE_TIMEOUT_SECONDS", "get_egress_ip"},
        {"http": fake_http},
    )
    yield ns["get_egress_ip"]
    _Conn.body, _Conn.fail = TRACE.encode(), False


def test_reads_the_ip_line(get_egress_ip) -> None:
    assert get_egress_ip() == "203.0.113.7"


def test_no_ip_line_is_empty(get_egress_ip) -> None:
    _Conn.body = b"h=www.cloudflare.com\n"
    assert get_egress_ip() == ""


def test_fetch_failure_is_empty(get_egress_ip) -> None:
    _Conn.fail = True
    assert get_egress_ip() == ""


def test_scrape_keys_are_mapped_for_the_backend() -> None:
    service = (SRC / "services" / "file_encrypt_service.py").read_text()
    assert "'data_egress_ip': \"egress_ip\"" in service
    assert "'data_egress_ip': \"\"" in service
