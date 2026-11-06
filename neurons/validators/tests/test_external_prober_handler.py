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
    monkeypatch.setattr(handler, "_get", lambda url, timeout: None)
    assert handler.handler(EVENT, None)["statusCode"] == 503


def test_canary_ok_reports_each_target(monkeypatch):
    monkeypatch.setattr(handler, "_get", lambda url, timeout: "ok" if url == handler.CANARY_URL else None)
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
