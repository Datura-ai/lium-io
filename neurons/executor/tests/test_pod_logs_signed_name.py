"""
POST /pod_logs answers only a request whose miner signature was made over the container name it reads.

MinerMiddleware verifies the signature over `data_to_sign`; the route reads `container_name`. Both must be the
same string (the miner signs exactly the container name, ExecutorService.get_pod_logs), the way
/upload_ssh_key and /remove_ssh_key already require `public_key == data_to_sign` (issue #744).

Ephemeral keypairs, the real middleware and the real route; only the log store is replaced.
"""

from unittest.mock import AsyncMock, MagicMock

import bittensor
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router
from services.pod_log_service import PodLogService

# conftest.py already inserted src/ into sys.path and set required env vars.
from core.config import settings

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"
_CONTAINER = "pod_0b2a6d1e-9e2e-4e2e-8e2e-000000000e2e"
_OTHER_CONTAINER = "pod_0b2a6d1e-9e2e-4e2e-8e2e-00000000beef"


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestMinerPodLogs")


@pytest.fixture(scope="module")
def portal_keypair():
    """The fixed portal hotkey every executor also trusts (settings.DEFAULT_MINER_HOTKEY)."""
    return bittensor.Keypair.create_from_uri("//LiumTestPortalPodLogs")


@pytest.fixture()
def store():
    service = MagicMock(spec=PodLogService)
    service.find_by_continer_name = AsyncMock(
        return_value=[{"event": "start", "container_name": _CONTAINER}]
    )
    return service


@pytest.fixture()
def client(miner_keypair, portal_keypair, store, monkeypatch):
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner_keypair.ss58_address)
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", portal_keypair.ss58_address)

    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    app.dependency_overrides[PodLogService] = lambda: store
    return TestClient(app)


def _signed(keypair, data_to_sign: str, container_name: str) -> dict:
    return {
        "container_name": container_name,
        "data_to_sign": data_to_sign,
        "signature": "0x" + keypair.sign(data_to_sign).hex(),
    }


def test_the_miners_own_request_is_answered(client, miner_keypair, store):
    """What ExecutorService.get_pod_logs sends: data_to_sign is the container name."""
    response = client.post("/pod_logs", json=_signed(miner_keypair, _CONTAINER, _CONTAINER))

    assert response.status_code == 200, response.text
    assert response.json() == [{"event": "start", "container_name": _CONTAINER}]
    store.find_by_continer_name.assert_awaited_once_with(_CONTAINER)


def test_a_signature_over_another_string_does_not_read_a_container(client, miner_keypair, store):
    """A blob the miner signed for /upload_ssh_key is a valid miner signature, but not over this container name."""
    response = client.post("/pod_logs", json=_signed(miner_keypair, _SSH_KEY, _CONTAINER))

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "Container name mismatch"}
    store.find_by_continer_name.assert_not_awaited()


def test_a_signature_over_one_container_does_not_read_another(client, miner_keypair, store):
    response = client.post("/pod_logs", json=_signed(miner_keypair, _OTHER_CONTAINER, _CONTAINER))

    assert response.status_code == 400, response.text
    store.find_by_continer_name.assert_not_awaited()


def test_the_portal_hotkey_is_held_to_the_same_binding(client, portal_keypair, store):
    """The portal hotkey is trusted by every executor; its signatures must bind the name too."""
    accepted = client.post("/pod_logs", json=_signed(portal_keypair, _CONTAINER, _CONTAINER))
    assert accepted.status_code == 200, accepted.text

    refused = client.post("/pod_logs", json=_signed(portal_keypair, _SSH_KEY, _CONTAINER))
    assert refused.status_code == 400, refused.text
    store.find_by_continer_name.assert_awaited_once_with(_CONTAINER)


def test_surrounding_whitespace_is_not_a_mismatch(client, miner_keypair, store):
    """Same normalisation as the SSH-key routes: the signed string is compared stripped."""
    body = _signed(miner_keypair, f" {_CONTAINER}\n", _CONTAINER)

    response = client.post("/pod_logs", json=body)

    assert response.status_code == 200, response.text
    store.find_by_continer_name.assert_awaited_once_with(_CONTAINER)


def test_a_stranger_is_still_refused_before_the_binding_is_checked(client, store):
    """The middleware's 401 comes first: a matching name with a foreign signature reads nothing."""
    stranger = bittensor.Keypair.create_from_uri("//LiumTestStrangerPodLogs")

    response = client.post("/pod_logs", json=_signed(stranger, _CONTAINER, _CONTAINER))

    assert response.status_code == 401, response.text
    store.find_by_continer_name.assert_not_awaited()
