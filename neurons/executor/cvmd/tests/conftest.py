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
from datetime import UTC, datetime, timedelta
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
CATALOG_SIGNER_URI = "//Dave"
OTHER_SIGNER_URI = "//Eve"


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


# --------------------------------------------------------------------------- DAH-2578
#
# The manifest helpers sign with the same library cvmd verifies with, for the same reason the
# request signer does: a second implementation in the tests is a second thing to keep in step
# with the protocol, and it would pass while the real backend failed. What these helpers do NOT
# share with the daemon is the *envelope* — it is written out by hand below, so a change to
# `manifest.py` that quietly altered the wire shape breaks these tests rather than moving with
# them.


@pytest.fixture
def catalog_signer() -> Keypair:
    """The platform key the host is configured to trust."""
    return Keypair.create_from_uri(CATALOG_SIGNER_URI)


@pytest.fixture
def other_signer() -> Keypair:
    """A valid key that is not the one this host trusts."""
    return Keypair.create_from_uri(OTHER_SIGNER_URI)


def manifest_payload(
    *entries: dict,
    serial: int = 1,
    floors: dict[str, int] | None = None,
    issued_at: datetime | None = None,
    ttl_seconds: int = 86400,
) -> str:
    issued = issued_at or datetime.now(UTC)
    return json.dumps(
        {
            "version": 1,
            "serial": serial,
            "issued_at": issued.isoformat(),
            "expires_at": (issued + timedelta(seconds=ttl_seconds)).isoformat(),
            "floors": floors or {"os_image": 1, "qemu": 1, "compose": 1},
            "artifacts": list(entries),
        }
    )


def sign_manifest(payload: str, keypair: Keypair, *, signer: str | None = None) -> bytes:
    """Wrap `payload` in the signed envelope, exactly as the backend serves it."""
    blob = b"lium-cvm-catalog-v1\x00" + payload.encode()
    return json.dumps(
        {
            "schema": "lium-cvm-catalog/1",
            "payload": payload,
            "signer": signer or keypair.ss58_address,
            "signature": "0x" + keypair.sign(blob).hex(),
        }
    ).encode()


def manifest_entry(**overrides) -> dict:
    base = {
        "id": "validation-v3",
        "kind": "validation",
        "qemu": "10.1.0",
        "os_image_hash": "a" * 64,
        "os_image_name": "dstack-nvidia-0.5.11",
        "compose_hash": "b" * 64,
        "compose": "services:\n  executor:\n    image: example\n",
        "versions": {"os_image": 1, "qemu": 1, "compose": 1},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- DAH-2576

# The real launcher, in this repo. The launch tests import it rather than a stub: the whole
# claim of DAH-2576 is that cvmd calls *this file*, and a stub would pass while the claim was
# false. It needs nothing but the standard library, so importing it here costs nothing.
DSTACK_SCRIPTS = Path(__file__).resolve().parents[2] / "dstacktee" / "scripts"


@pytest.fixture
def dstack_scripts() -> Path:
    if not (DSTACK_SCRIPTS / "dstack.py").is_file():
        pytest.skip(f"no dstack.py at {DSTACK_SCRIPTS}")
    return DSTACK_SCRIPTS


@pytest.fixture
def dstack(dstack_scripts: Path):
    from cvmd.dstack.loader import load_dstack

    return load_dstack(dstack_scripts)


@pytest.fixture
def compose_file(tmp_path: Path) -> Path:
    """A compose file with no unresolved placeholders.

    `lium-cvm.sh` substitutes `${EXECUTOR_RUNNER_IMAGE_DIGEST}` *before* dstack measures the
    file. Under cvmd the catalog points at an already-resolved compose, and the compose-hash
    gate is what catches it if that ever stops being true.
    """
    path = tmp_path / "docker-compose.yml"
    path.write_text(
        "services:\n"
        "  executor-runner:\n"
        "    image: daturaai/compute-subnet-executor-runner@sha256:" + ("c" * 64) + "\n"
    )
    return path


@pytest.fixture
def guest_scripts(tmp_path: Path) -> tuple[Path, Path]:
    init = tmp_path / "init_script.sh"
    init.write_text("#!/bin/sh\necho init\n")
    pre_launch = tmp_path / "pre_launch_script.sh"
    pre_launch.write_text("#!/bin/sh\necho pre-launch\n")
    return init, pre_launch


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "cvm.env"
    path.write_text("EXTERNAL_PORT=8001\nSSH_PORT=2200\n")
    return path


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """An OS image directory with the one file cvmd measures.

    `setup_instance` only records the image path; `digest.txt` is read at measurement time and
    by `run_instance`, so this is enough for everything short of an actual boot.
    """
    path = tmp_path / "images" / "dstack-nvidia-0.5.11"
    path.mkdir(parents=True)
    (path / "digest.txt").write_text("d" * 64 + "\n")
    return path


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
