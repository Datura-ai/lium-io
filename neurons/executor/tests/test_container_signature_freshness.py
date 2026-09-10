"""DAH-3200: a signed /containers request is accepted only while its timestamp is fresh.

The backend signs `{"container_name"|"gpu_uuids", "timestamp": int(time.time())}` with the validator
hotkey and the executor verified the signature but never read the clock, so a captured request stayed
valid for the life of the container. Both verifiers now refuse a timestamp more than
CONTAINER_SIGNATURE_MAX_AGE_SECONDS away from the executor clock, in either direction, before the
signature is even checked. Signatures below are real (ephemeral keypair), as the backend produces them.
"""

import json
import time
from unittest.mock import MagicMock

import bittensor
import pytest
import routes.apis as apis
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from payloads.backend import ContainerUtilizationPayload
from routes.apis import apis_router

import dependencies.auth as auth
from core.config import settings

WINDOW = 5  # seconds; a stale request only gets staler on a slow runner, so the refused cases hold


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidatorFreshness")


@pytest.fixture(autouse=True)
def trust_the_test_validator(validator_keypair, monkeypatch):
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEY_SS58", validator_keypair.ss58_address)
    monkeypatch.setattr(settings, "CONTAINER_SIGNATURE_MAX_AGE_SECONDS", WINDOW)


def _sign(keypair, signing_data: dict) -> str:
    # lium-platform pod.py: json.dumps(payload, sort_keys=True) then wallet.hotkey.sign(...).hex()
    return keypair.sign(json.dumps(signing_data, sort_keys=True).encode()).hex()


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# --- the verifiers -----------------------------------------------------------------------------


def test_a_fresh_container_logs_signature_is_accepted(validator_keypair):
    timestamp = int(time.time())
    signature = _sign(validator_keypair, {"container_name": "container_x", "timestamp": timestamp})

    _run(auth.verify_container_logs_signature("container_x", timestamp, signature))


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


def test_a_fresh_container_utilization_signature_is_accepted(validator_keypair):
    timestamp = int(time.time())
    payload = ContainerUtilizationPayload(
        gpu_uuids=["GPU-1"],
        timestamp=timestamp,
        signature=_sign(validator_keypair, {"gpu_uuids": ["GPU-1"], "timestamp": timestamp}),
    )

    _run(auth.verify_container_signature(payload))


@pytest.mark.parametrize("offset", [-(WINDOW + 2), WINDOW + 60], ids=["stale", "future"])
def test_a_container_utilization_signature_outside_the_window_is_refused(validator_keypair, offset):
    timestamp = int(time.time()) + offset
    payload = ContainerUtilizationPayload(
        gpu_uuids=["GPU-1"],
        timestamp=timestamp,
        signature=_sign(validator_keypair, {"gpu_uuids": ["GPU-1"], "timestamp": timestamp}),
    )

    with pytest.raises(HTTPException) as refused:
        _run(auth.verify_container_signature(payload))

    assert refused.value.status_code == 401


def test_the_window_comes_from_settings(validator_keypair, monkeypatch):
    monkeypatch.setattr(settings, "CONTAINER_SIGNATURE_MAX_AGE_SECONDS", 3 * WINDOW)
    timestamp = int(time.time()) - (WINDOW + 2)  # refused under WINDOW, accepted under 3 * WINDOW
    signature = _sign(validator_keypair, {"container_name": "container_x", "timestamp": timestamp})

    _run(auth.verify_container_logs_signature("container_x", timestamp, signature))


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


def test_stale_logs_request_is_refused_before_the_container_is_looked_up(
    validator_keypair, monkeypatch
):
    def no_docker():
        raise AssertionError(
            "docker.from_env() was called for a request that should have been refused"
        )

    monkeypatch.setattr(apis.docker, "from_env", no_docker)
    monkeypatch.setattr(apis, "_active_follow_log_streams", 0)
    app = FastAPI()
    app.include_router(apis_router)
    client = TestClient(app)

    timestamp = int(time.time()) - (WINDOW + 2)
    signature = _sign(validator_keypair, {"container_name": "container_x", "timestamp": timestamp})

    response = client.get(
        "/containers/container_x/logs",
        headers={"X-Signature": signature, "X-Timestamp": str(timestamp)},
    )

    assert response.status_code == 401
    assert f"the accepted window is {WINDOW}s" in response.json()["detail"]


def test_an_absurdly_large_timestamp_is_refused_not_a_500(monkeypatch):
    # the header parser admits any int; the skew must stay integer arithmetic so this answers 401
    def no_docker():
        raise AssertionError(
            "docker.from_env() was called for a request that should have been refused"
        )

    monkeypatch.setattr(apis.docker, "from_env", no_docker)
    app = FastAPI()
    app.include_router(apis_router)
    client = TestClient(app)
    huge = "1" + "0" * 400

    response = client.get(
        "/containers/container_x/logs",
        headers={"X-Signature": "00" * 64, "X-Timestamp": huge},
    )

    assert response.status_code == 401
    assert "ahead of the executor clock" in response.json()["detail"]


def test_fresh_logs_request_reaches_the_container_lookup(validator_keypair, monkeypatch):
    fake_container = MagicMock()
    fake_container.logs.return_value = iter([b"log line\n"])
    fake_client = MagicMock()
    fake_client.containers.get.return_value = fake_container
    monkeypatch.setattr(apis.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(apis, "_active_follow_log_streams", 0)
    app = FastAPI()
    app.include_router(apis_router)
    client = TestClient(app)

    timestamp = int(time.time())
    signature = _sign(validator_keypair, {"container_name": "container_x", "timestamp": timestamp})

    response = client.get(
        "/containers/container_x/logs",
        headers={"X-Signature": signature, "X-Timestamp": str(timestamp)},
    )

    assert response.status_code == 200
    assert response.content == b"log line\n"
    fake_client.containers.get.assert_called_once_with("container_x")
