"""DAH-3394: the executor accepts a signature from either of two configured validator hotkeys.

The validator hotkey is rotated in two releases: this one trusts `current` and `next`, the one after
drops `current`. Every test names the regression it guards; signatures are real (ephemeral keypairs),
shaped exactly as the miner relays them, and go through the routes, not the helper.
"""

import importlib.util
import os
import sys
import types
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


def _ping_payload(validator_kp) -> dict:
    return {"signature": validator_kp.sign("ping_request").hex()}


# --- /upload_ssh_key ------------------------------------------------------------------------------


def test_upload_signed_by_the_current_hotkey_is_accepted_when_next_is_configured(
    client, miner_keypair, current_validator, next_validator, monkeypatch
):
    # regression: configuring `next` replaces `current` instead of adding to it — the live
    # validator is refused by every executor that restarts onto the release
    _trust(monkeypatch, current=current_validator, next=next_validator)

    response = client.post("/upload_ssh_key", json=_upload_payload(miner_keypair, current_validator))

    assert response.status_code == 200


def test_upload_signed_by_the_next_hotkey_is_accepted_when_configured(
    client, miner_keypair, current_validator, next_validator, monkeypatch
):
    # regression: the verifier reads only the first configured hotkey, so the chain swap
    # (phase 2) is fleet-wide downtime
    _trust(monkeypatch, current=current_validator, next=next_validator)

    response = client.post("/upload_ssh_key", json=_upload_payload(miner_keypair, next_validator))

    assert response.status_code == 200


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


def test_ping_signed_by_the_next_hotkey_is_accepted_when_configured(
    client, current_validator, next_validator, monkeypatch
):
    # regression: the rotation reaches _validate_validator_signature (ssh key uploads) but not
    # dependencies.auth.verify_signature, so /ping, /hardware_utilization and /containers/*
    # go dark after the chain swap while rentals still work
    _trust(monkeypatch, current=current_validator, next=next_validator)

    response = client.post("/ping", json=_ping_payload(next_validator))

    assert response.status_code == 200


def test_ping_signed_by_a_third_hotkey_is_rejected_with_two_configured(
    client, current_validator, next_validator, monkeypatch
):
    # regression: verify_signature's error path (a raised or False verify on the first key) is
    # read as "verified" once a second key is in the loop
    _trust(monkeypatch, current=current_validator, next=next_validator)
    stranger = bittensor.Keypair.create_from_uri("//LiumRotationStranger")

    response = client.post("/ping", json=_ping_payload(stranger))

    assert response.status_code == 401


# --- core/config.py: where the two hotkeys come from ----------------------------------------------


def _load_config_with_override(monkeypatch, next_hotkey: str | None):
    # a fresh module object from core/config.py with `core.config_override` standing in as
    # docker_build.sh would have written it (next only; `current` keeps the code default). Reloading
    # sys.modules["core.config"] instead would hand every other test a second `settings` object.
    if next_hotkey is None:
        monkeypatch.delitem(sys.modules, "core.config_override", raising=False)
    else:
        monkeypatch.setitem(
            sys.modules, "core.config_override", types.SimpleNamespace(_VALIDATOR_NEXT_HOTKEY_SS58=next_hotkey)
        )
    # the executor must never take a hotkey from its environment; set it to prove it is ignored
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY_SS58", bittensor.Keypair.create_from_uri("//FromEnv").ss58_address)
    path = os.path.join(os.path.dirname(__file__), "..", "src", "core", "config.py")
    spec = importlib.util.spec_from_file_location("config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_an_override_only_the_current_hotkey_is_configured(monkeypatch):
    # regression: an empty `next` becomes a second entry ("" or a duplicate of current), or the
    # environment variable of the same name is honoured — the fleet as deployed today must see
    # exactly one hotkey, and a provider's .env must not add a signer
    config = _load_config_with_override(monkeypatch, next_hotkey=None)

    assert config.VALIDATOR_HOTKEYS_SS58 == {"current": config.VALIDATOR_HOTKEY_SS58}


def test_the_build_time_override_adds_the_next_hotkey_after_current(monkeypatch, next_validator):
    # regression: the override is read but never reaches the dict (or lands first, so the log
    # says `current` for the new key), or surrounding whitespace is kept
    config = _load_config_with_override(monkeypatch, next_hotkey=f" {next_validator.ss58_address} ")

    assert list(config.VALIDATOR_HOTKEYS_SS58.items()) == [
        ("current", config.VALIDATOR_HOTKEY_SS58),
        ("next", next_validator.ss58_address),
    ]


def test_a_next_hotkey_equal_to_current_is_not_listed_twice(monkeypatch):
    # regression: the same key verified twice, and a log line claiming `next` is in use
    current = _load_config_with_override(monkeypatch, next_hotkey=None).VALIDATOR_HOTKEY_SS58
    config = _load_config_with_override(monkeypatch, next_hotkey=current)

    assert config.VALIDATOR_HOTKEYS_SS58 == {"current": current}


def test_a_next_hotkey_that_is_not_an_ss58_address_stops_the_executor_at_import(monkeypatch):
    # regression: a mistyped `next` is discovered by the first 401 after the chain swap, on the
    # whole fleet, instead of by CI on this PR
    with pytest.raises(ValueError, match="next validator hotkey"):
        _load_config_with_override(monkeypatch, next_hotkey="5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13q")
