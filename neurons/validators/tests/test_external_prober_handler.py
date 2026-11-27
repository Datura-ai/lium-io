"""The outside prober answers no verdict (503) when its own egress is down, never an all-blocked 200 (DAH-4537)."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "external_prober_handler", Path(__file__).parents[1] / "external_prober" / "handler.py"
)
handler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handler)

EVENT = {
    "headers": {"Authorization": "Bearer t0k"},
    "body": json.dumps({"targets": [{"host": "1.2.3.4", "port": 8080}]}),
}


@pytest.fixture(autouse=True)
def probe_token_env(monkeypatch):
    monkeypatch.setenv("PROBE_TOKEN", "t0k")


def test_egress_down_answers_no_verdict(monkeypatch):
    monkeypatch.setattr(handler, "_get", lambda url, timeout, read=None: None)
    assert handler.handler(EVENT, None)["statusCode"] == 503


def test_canary_ok_reports_each_target(monkeypatch):
    monkeypatch.setattr(handler, "_get", lambda url, timeout, read=None: "ok" if url == handler.CANARY_URL else None)
    res = handler.handler(EVENT, None)
    assert res["statusCode"] == 200
    assert json.loads(res["body"])["results"] == [{"host": "1.2.3.4", "port": 8080, "body": None}]


def test_redirects_are_not_followed():
    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/private")
                self.end_headers()
                self.wfile.write(b"moved")
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"private")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert handler._fetch("127.0.0.1", server.server_port, 3) == "moved"
    finally:
        server.shutdown()


def test_a_canary_with_a_long_answer_still_proves_egress(monkeypatch):
    """The canary only has to answer; the 256-byte rule is for node answers, so a long canary page is still egress."""

    class Canary(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * 1000)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Canary)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(handler, "CANARY_URL", f"http://127.0.0.1:{server.server_port}/")
    try:
        assert handler._canary_up(3)
        assert handler._fetch("127.0.0.1", server.server_port, 3) is None
    finally:
        server.shutdown()


@pytest.mark.parametrize("status", [200, 302])
def test_a_node_that_stalls_mid_answer_is_no_answer(status):
    """A node that sends part of its body and then stalls gives no answer for itself, whatever its status."""
    release = threading.Event()

    class Stall(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"x" * 256)
            self.wfile.flush()
            release.wait(5)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Stall)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert handler._fetch("127.0.0.1", server.server_port, 0.5) is None
    finally:
        release.set()
        server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("size, passes", [(256, True), (257, False)])
async def test_both_probe_paths_judge_an_answer_of_the_same_size_alike(size, passes):
    """The outside prober and the validator's own probe accept and reject exactly the same answers."""
    from neurons.validators.src.services.executor_connectivity.models import PortPair
    from neurons.validators.src.services.executor_connectivity.port_tester import (
        PROBE_BODY_LIMIT,
        PortTester,
    )

    assert handler.MAX_BODY == PROBE_BODY_LIMIT
    answer = b"token:7".ljust(size)

    class Node(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Node)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # the validator's verdict on the outside prober's answer, as in ExternalPortProbe.reached
        outside = (handler._fetch("127.0.0.1", server.server_port, 3) or "").strip() == "token:7"
        inside = await PortTester().test_one("127.0.0.1", PortPair(internal=7, external=server.server_port), "token")
    finally:
        server.shutdown()
    assert outside is passes
    assert inside is passes
