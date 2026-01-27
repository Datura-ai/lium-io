import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MINER_HOTKEY_SS58_ADDRESS", "5TestMinerHotkeyAddress")
os.environ.setdefault("DB_URI", "sqlite:///tmp/test.db")

from core.config import settings  # noqa: E402
from neurons.executor.src.services.tdx_service import TDXQuoteService  # noqa: E402


async def _run_quote_flow(monkeypatch):
    quote_payload = {"quote": "00", "event_log": "[]"}
    captured: dict[str, object] = {"count": 0}

    async def handle_get_quote(request: web.Request):
        captured["count"] = int(captured["count"]) + 1
        captured["body"] = await request.json()
        return web.json_response(quote_payload)

    app = web.Application()
    app.router.add_post("/GetQuote", handle_get_quote)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    original_endpoint = os.environ.get("DSTACK_SIMULATOR_ENDPOINT")
    monkeypatch.setenv("DSTACK_SIMULATOR_ENDPOINT", f"http://127.0.0.1:{port}")

    original_enabled = settings.ENABLE_TDX_ATTESTATION
    original_timeout = settings.TDX_QUOTE_TIMEOUT
    settings.ENABLE_TDX_ATTESTATION = True
    settings.TDX_QUOTE_TIMEOUT = 2

    try:
        service = TDXQuoteService()
        host_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestHostKey executor"
        expected_digest = hashlib.sha256(
            b"SSH_HOST_KEY:" + host_key.encode("utf-8")
        ).hexdigest()

        quote = await service.get_quote(host_key)
        assert quote == json.dumps(quote_payload, separators=(",", ":"))

        body = captured["body"]
        assert isinstance(body, dict)
        assert body["report_data"] == expected_digest

        cached = await service.get_quote(host_key)
        assert cached == quote
        assert captured["count"] == 1

        # Missing host key should skip the request
        assert await service.get_quote(None) is None
    finally:
        settings.ENABLE_TDX_ATTESTATION = original_enabled
        settings.TDX_QUOTE_TIMEOUT = original_timeout
        if original_endpoint is None:
            monkeypatch.delenv("DSTACK_SIMULATOR_ENDPOINT", raising=False)
        else:
            monkeypatch.setenv("DSTACK_SIMULATOR_ENDPOINT", original_endpoint)
        await runner.cleanup()


def test_tdx_quote_service_fetches_and_caches_quotes(monkeypatch):
    asyncio.run(_run_quote_flow(monkeypatch))


def test_tdx_quote_service_disabled(monkeypatch):
    original_enabled = settings.ENABLE_TDX_ATTESTATION
    settings.ENABLE_TDX_ATTESTATION = False
    try:
        service = TDXQuoteService()
        result = asyncio.run(service.get_quote("ssh-ed25519 AAAAC3NzaTest executor"))
        assert result is None
    finally:
        settings.ENABLE_TDX_ATTESTATION = original_enabled
