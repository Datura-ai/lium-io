"""Shared fixtures and the request-signing helper.

The signing helper uses the same bittensor library as the verifier on purpose: a second
signature implementation in the test suite would be a second thing to keep in step with the
protocol, and it would pass while the real clients failed. The one place a genuinely independent
signer matters is the cross-version golden vector, which is generated under bittensor 9.x by
tests/fixtures/generate_golden_vector.py and only verified here.
"""

import json
import secrets
import time
from pathlib import Path

import pytest
from bittensor.sp_core import Keypair
from cvmd.app import create_app
from cvmd.auth.blob import signing_blob
from cvmd.auth.middleware import (
    HOTKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
)
from cvmd.config import Config
from fastapi.testclient import TestClient

VALIDATOR_URI = "//Alice"
PLATFORM_URI = "//Bob"
STRANGER_URI = "//Charlie"


@pytest.fixture
def validator_key() -> Keypair:
    return Keypair.create_from_uri(VALIDATOR_URI)


@pytest.fixture
def platform_key() -> Keypair:
    return Keypair.create_from_uri(PLATFORM_URI)


@pytest.fixture
def stranger_key() -> Keypair:
    """A syntactically valid hotkey that is not in the authorized-clients file."""
    return Keypair.create_from_uri(STRANGER_URI)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    path = tmp_path / "state"
    path.mkdir()
    return path


@pytest.fixture
def clients_file(tmp_path: Path, validator_key: Keypair, platform_key: Keypair) -> Path:
    path = tmp_path / "authorized_clients.json"
    path.write_text(
        json.dumps(
            [
                {"hotkey": validator_key.ss58_address, "scope": "validation"},
                {"hotkey": platform_key.ss58_address, "scope": "renter"},
            ]
        )
    )
    return path


@pytest.fixture
def config(clients_file: Path, state_dir: Path) -> Config:
    return Config(authorized_clients=clients_file, state_dir=state_dir)


@pytest.fixture
def app(config: Config):
    return create_app(config)


@pytest.fixture
def client(app) -> TestClient:
    # raise_server_exceptions=False so a handler that blows up surfaces as a 500 response rather
    # than propagating into the test — test_replay.py needs that to simulate a mid-request crash.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def sign_headers(
    keypair: Keypair,
    *,
    method: str,
    request_target: str,
    body: bytes = b"",
    timestamp_ns: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Build the four auth headers for one request. Mirrors what a real client does."""
    timestamp = str(timestamp_ns if timestamp_ns is not None else time.time_ns())
    nonce = nonce if nonce is not None else secrets.token_hex(16)

    blob = signing_blob(
        method=method,
        request_target=request_target,
        body=body,
        timestamp=timestamp,
        nonce=nonce,
    )
    return {
        HOTKEY_HEADER: keypair.ss58_address,
        TIMESTAMP_HEADER: timestamp,
        NONCE_HEADER: nonce,
        SIGNATURE_HEADER: keypair.sign(blob).hex(),
    }


def signed_request(
    client: TestClient,
    keypair: Keypair,
    method: str,
    path: str,
    *,
    body: bytes = b"",
    timestamp_ns: int | None = None,
    nonce: str | None = None,
):
    headers = sign_headers(
        keypair,
        method=method,
        request_target=path,
        body=body,
        timestamp_ns=timestamp_ns,
        nonce=nonce,
    )
    return client.request(method, path, content=body, headers=headers)
