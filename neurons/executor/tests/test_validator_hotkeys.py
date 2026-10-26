"""DAH-3394: the executor accepts a signature from either of two configured validator hotkeys.

The validator hotkey is rotated in two releases: this one trusts `current` and `next`, the one after
drops `current`. Every test names the regression it guards; signatures are real (ephemeral keypairs),
shaped exactly as the miner relays them, and go through the routes, not the helper.
"""

import importlib.machinery
import importlib.util
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Keypair
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
    return Keypair.create_from_uri("//LiumRotationCurrent")


@pytest.fixture(scope="module")
def next_validator():
    return Keypair.create_from_uri("//LiumRotationNext")


@pytest.fixture(scope="module")
def miner_keypair():
    return Keypair.create_from_uri("//LiumRotationMiner")


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


def _trust(monkeypatch, **hotkeys: Keypair) -> None:
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
    stranger = Keypair.create_from_uri("//LiumRotationStranger")

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


def _load_config_with_override(monkeypatch, next_hotkey: str | None):
    # a fresh module object from core/config.py with `core.config_override` standing in as
    # docker_build.sh writes it for a rotation build (both hotkeys; see below). Reloading
    # sys.modules["core.config"] instead would hand every other test a second `settings` object.
    if next_hotkey is None:
        monkeypatch.delitem(sys.modules, "core.config_override", raising=False)
    else:
        # what docker_build.sh writes for a rotation build: both hotkeys (config.py since
        # DAH-3114 reads `_VALIDATOR_HOTKEY_SS58` from any override it finds). A real module with
        # a __spec__, because `importlib.util.find_spec` raises on a bare namespace in sys.modules.
        override = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("core.config_override", None))
        override._VALIDATOR_HOTKEY_SS58 = _load_config_with_override(monkeypatch, next_hotkey=None).VALIDATOR_HOTKEY_SS58
        override._VALIDATOR_NEXT_HOTKEY_SS58 = next_hotkey
        monkeypatch.setitem(sys.modules, "core.config_override", override)
    # the executor must never take a hotkey from its environment; set it to prove it is ignored
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY_SS58", Keypair.create_from_uri("//FromEnv").ss58_address)
    path = os.path.join(os.path.dirname(__file__), "..", "src", "core", "config.py")
    spec = importlib.util.spec_from_file_location("config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_an_override_the_built_in_pair_is_configured_current_first(monkeypatch):
    # regression: the default build trusts only `current` (the fleet is refused by the validator
    # after the chain swap), lists `next` first (the log names the wrong key as `current`), or
    # honours the environment variable of the same name (a provider's .env adds a signer)
    config = _load_config_with_override(monkeypatch, next_hotkey=None)

    assert list(config.VALIDATOR_HOTKEYS_SS58.items()) == [
        ("current", "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"),
        ("next", "5F1aH9124FPvDNMWKgPCCJ8WeBBV3B3vpijDCyxZVyZgrEV9"),
    ]


def test_the_active_anchor_stays_the_current_hotkey_until_the_swap_release(monkeypatch):
    # regression: the new address is written into `current` (the still-live validator is refused
    # by every executor that restarts onto this release) instead of `next`
    config = _load_config_with_override(monkeypatch, next_hotkey=None)

    assert config.VALIDATOR_HOTKEY_SS58 == config._BUILTIN_VALIDATOR_HOTKEY_SS58
    assert config.VALIDATOR_HOTKEY_SS58 != config._BUILTIN_VALIDATOR_NEXT_HOTKEY_SS58


def test_an_override_that_names_only_current_trusts_one_signer(monkeypatch):
    # regression: the built-in `next` (the prod validator's new hotkey) leaks into a staging or e2e
    # build whose override names only its own `current`
    override = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("core.config_override", None))
    override._VALIDATOR_HOTKEY_SS58 = Keypair.create_from_uri("//LiumStagingOnly").ss58_address
    monkeypatch.setitem(sys.modules, "core.config_override", override)
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY_SS58", Keypair.create_from_uri("//FromEnv").ss58_address)
    path = os.path.join(os.path.dirname(__file__), "..", "src", "core", "config.py")
    spec = importlib.util.spec_from_file_location("config_under_test_staging", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    assert config.VALIDATOR_HOTKEYS_SS58 == {"current": override._VALIDATOR_HOTKEY_SS58}


