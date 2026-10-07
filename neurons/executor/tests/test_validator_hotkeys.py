"""DAH-3394: the executor accepts a signature from either of two configured validator hotkeys.

The validator hotkey is rotated in two releases: this one trusts `current` and `next`, the one after
drops `current`. Every test names the regression it guards; signatures are real (ephemeral keypairs),
shaped exactly as the miner relays them, and go through the routes, not the helper.
"""

from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router
from services.miner_service import MinerService

import dependencies.auth as auth
from core.config import settings

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef validator@lium"


@pytest.fixture(scope="module")
def current_validator():
    return bittensor.Keypair.create_from_uri("//LiumRotationCurrent")


@pytest.fixture(scope="module")
def next_validator():
    return bittensor.Keypair.create_from_uri("//LiumRotationNext")


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumRotationMiner")


@pytest.fixture()
def client(miner_keypair, monkeypatch):
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner_keypair.ss58_address)
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", miner_keypair.ss58_address)
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    service = MagicMock()
    service.upload_ssh_key = AsyncMock(return_value={"ssh_username": "root", "ssh_port": 2200})
    service.remove_ssh_key = AsyncMock(return_value=None)
    app.dependency_overrides[MinerService] = lambda: service
    return TestClient(app)


def _trust(monkeypatch, **hotkeys: bittensor.Keypair) -> None:
    monkeypatch.setattr(
        auth, "VALIDATOR_HOTKEYS_SS58", {key_id: kp.ss58_address for key_id, kp in hotkeys.items()}
    )


def _upload_payload(miner_kp, validator_kp) -> dict:
    # miners/services/executor_service.py:send_pubkey_to_executor, no nonce
    return {
        "public_key": _SSH_KEY,
        "data_to_sign": _SSH_KEY,
        "signature": "0x" + miner_kp.sign(_SSH_KEY).hex(),
        "validator_signature": "0x" + validator_kp.sign(_SSH_KEY).hex(),
    }


# --- /upload_ssh_key ------------------------------------------------------------------------------


def test_upload_signed_by_a_third_hotkey_is_rejected_with_two_configured(
    client, miner_keypair, current_validator, next_validator, monkeypatch
):
    # regression: a loop that returns the last comparison's result, or one that treats "no key
    # raised" as verified, accepts any signer once more than one hotkey is configured
    _trust(monkeypatch, current=current_validator, next=next_validator)
    stranger = bittensor.Keypair.create_from_uri("//LiumRotationStranger")

    response = client.post("/upload_ssh_key", json=_upload_payload(miner_keypair, stranger))

    assert response.status_code == 401


def test_upload_signed_by_the_next_hotkey_is_rejected_while_only_current_is_configured(
    client, miner_keypair, current_validator, next_validator, monkeypatch
):
    # regression: `next` accepted before the release that names it — exactly today's behaviour
    # must hold for a single configured key
    _trust(monkeypatch, current=current_validator)

    response = client.post("/upload_ssh_key", json=_upload_payload(miner_keypair, next_validator))

    assert response.status_code == 401


# --- the dependencies/auth.py verifiers (ping, hardware, containers) ------------------------------


# --- core/config.py: where the two hotkeys come from ----------------------------------------------


