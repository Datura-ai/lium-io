"""DAH-6430: /hardware_utilization and /ping accept a signature bound to this executor's miner hotkey
and a fresh timestamp; the legacy fixed-string form stays accepted while ACCEPT_UNBOUND_SIGNATURES."""

import asyncio
import time

import pytest
from bittensor_wallet import Keypair
from fastapi import HTTPException
from payloads.backend import HardwareUtilizationPayload, PingPayload

import dependencies.auth as auth
from core.config import settings

OWN = Keypair.create_from_uri("//LiumTestMinerOwn").ss58_address
OTHER = Keypair.create_from_uri("//LiumTestMinerOther").ss58_address


@pytest.fixture(scope="module")
def validator():
    return Keypair.create_from_uri("//LiumTestValidatorBound")


@pytest.fixture(autouse=True)
def setup(validator, monkeypatch):
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEYS_SS58", {"current": validator.ss58_address})
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", OWN)
    monkeypatch.setattr(settings, "LIUM_POOL_HOTKEY", "")
    monkeypatch.setattr(settings, "CONTAINER_SIGNATURE_MAX_AGE_SECONDS", 5)


def _bound(validator, purpose, miner_hotkey, timestamp, payload_cls):
    sig = validator.sign(auth.bound_signing_message(purpose, miner_hotkey, timestamp).encode()).hex()
    legacy = validator.sign(purpose.encode()).hex()
    return payload_cls(signature=legacy, miner_hotkey=miner_hotkey, timestamp=timestamp, bound_signature=sig)


@pytest.mark.parametrize(
    "purpose,payload_cls,verifier",
    [
        ("hardware_utilization_request", HardwareUtilizationPayload, auth.verify_allowed_hotkey_signature),
        ("ping_request", PingPayload, auth.verify_ping_signature),
    ],
)
def test_bound_form(validator, purpose, payload_cls, verifier, monkeypatch):
    now = int(time.time())
    asyncio.run(verifier(_bound(validator, purpose, OWN, now, payload_cls)))  # accepted

    monkeypatch.setattr(auth, "ACCEPT_UNBOUND_SIGNATURES", False)
    asyncio.run(verifier(_bound(validator, purpose, OWN, now, payload_cls)))  # accepted without the legacy path
    for refused in (
        _bound(validator, purpose, OWN, now - 60, payload_cls),  # stale
        _bound(validator, purpose, OTHER, now, payload_cls),  # another miner's nodes
        _bound(validator, "ping_request" if purpose != "ping_request" else "hardware_utilization_request",
               OWN, now, payload_cls),  # signed for the other endpoint
    ):
        with pytest.raises(HTTPException) as err:
            asyncio.run(verifier(refused))
        assert err.value.status_code == 401


def test_a_refused_bound_form_falls_back_to_the_legacy_one_while_accepted(validator):
    # a stale bound signature (host clock off) with a valid legacy one: accepted during the rollout
    asyncio.run(auth.verify_ping_signature(_bound(validator, "ping_request", OWN, int(time.time()) - 60, PingPayload)))


def test_legacy_form_only_while_accepted(validator, monkeypatch):
    legacy = PingPayload(signature=validator.sign(b"ping_request").hex())
    asyncio.run(auth.verify_ping_signature(legacy))

    monkeypatch.setattr(auth, "ACCEPT_UNBOUND_SIGNATURES", False)
    with pytest.raises(HTTPException) as err:
        asyncio.run(auth.verify_ping_signature(legacy))
    assert err.value.status_code == 401
