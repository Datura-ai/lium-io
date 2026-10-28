"""GET /miner-identity: the executor names the miner hotkey it is configured for and echoes the
caller's nonce, without a signature (the hotkey is public on chain)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router

from core.config import settings

NONCE = "a1" * 16


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    return TestClient(app)


def test_returns_configured_hotkey_and_echoes_nonce():
    response = _client().get("/miner-identity", params={"nonce": NONCE})

    assert response.status_code == 200
    assert response.json() == {"miner_hotkey": settings.MINER_HOTKEY_SS58_ADDRESS, "nonce": NONCE}  # no claim set


def test_follows_the_configured_hotkey(monkeypatch):
    monkeypatch.setattr(
        settings, "MINER_HOTKEY_SS58_ADDRESS", "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    )

    response = _client().get("/miner-identity", params={"nonce": NONCE})

    assert response.json()["miner_hotkey"] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def test_carries_the_account_claim_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "MINER_ACCOUNT_CLAIM", "c1" * 16)

    response = _client().get("/miner-identity", params={"nonce": NONCE})

    assert response.json() == {
        "miner_hotkey": settings.MINER_HOTKEY_SS58_ADDRESS,
        "nonce": NONCE,
        "account_claim": "c1" * 16,
    }


def test_nonce_is_required():
    assert _client().get("/miner-identity").status_code == 422


def test_short_or_non_hex_nonce_is_refused():
    client = _client()
    assert client.get("/miner-identity", params={"nonce": "ab" * 8}).status_code == 422
    assert client.get("/miner-identity", params={"nonce": "zz" * 16}).status_code == 422
    assert client.get("/miner-identity", params={"nonce": "ab" * 65}).status_code == 422
