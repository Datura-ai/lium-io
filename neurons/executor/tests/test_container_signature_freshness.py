"""DAH-3200: a signed /containers request is accepted only while its timestamp is fresh.

The backend signs `{"container_name"|"gpu_uuids", "timestamp": int(time.time())}` with the validator
hotkey and the executor verified the signature but never read the clock, so a captured request stayed
valid for the life of the container. Both verifiers now refuse a timestamp more than
CONTAINER_SIGNATURE_MAX_AGE_SECONDS away from the executor clock, in either direction, before the
signature is even checked. Signatures below are real (ephemeral keypair), as the backend produces them.
"""

import json
import time

import bittensor
import pytest
from fastapi import HTTPException

import dependencies.auth as auth
from core.config import settings

WINDOW = 5  # seconds; a stale request only gets staler on a slow runner, so the refused cases hold


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidatorFreshness")


@pytest.fixture(autouse=True)
def trust_the_test_validator(validator_keypair, monkeypatch):
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEYS_SS58", {"current": validator_keypair.ss58_address})
    monkeypatch.setattr(settings, "CONTAINER_SIGNATURE_MAX_AGE_SECONDS", WINDOW)


def _sign(keypair, signing_data: dict) -> str:
    # lium-platform pod.py: json.dumps(payload, sort_keys=True) then wallet.hotkey.sign(...).hex()
    return keypair.sign(json.dumps(signing_data, sort_keys=True).encode()).hex()


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# --- the verifiers -----------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-(WINDOW + 2), WINDOW + 60], ids=["stale", "future"])
def test_a_container_logs_signature_outside_the_window_is_refused_even_though_it_verifies(
    validator_keypair, offset
):
    timestamp = int(time.time()) + offset
    signature = _sign(validator_keypair, {"container_name": "container_x", "timestamp": timestamp})

    with pytest.raises(HTTPException) as refused:
        _run(auth.verify_container_logs_signature("container_x", timestamp, signature))

    assert refused.value.status_code == 401
    assert f"the accepted window is {WINDOW}s" in refused.value.detail
    assert ("behind" if offset < 0 else "ahead of") in refused.value.detail


def test_a_bad_signature_with_a_fresh_timestamp_is_still_refused(validator_keypair):
    # freshness is added in front of the signature check, not instead of it
    stranger = bittensor.Keypair.create_from_uri("//LiumTestStrangerFreshness")
    timestamp = int(time.time())
    signature = _sign(stranger, {"container_name": "container_x", "timestamp": timestamp})

    with pytest.raises(HTTPException) as refused:
        _run(auth.verify_container_logs_signature("container_x", timestamp, signature))

    assert refused.value.status_code == 401
    assert "Invalid signature" in refused.value.detail


# --- the route: refused before any docker call ---------------------------------------------------


