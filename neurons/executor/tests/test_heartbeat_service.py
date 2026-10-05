"""services/heartbeat_service.py: the signed node heartbeat."""

import asyncio
import base64
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_ssh_public_key,
)

from services import heartbeat_service as hb

# The backend verifies the same vector; a change to the signed bytes must change both sides.
VECTOR_SEED = bytes(range(32))
VECTOR_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAOhB7/zzhC+HXDdGOdLwJln5NYwm6UNXx3chmQSVTG4"
)
VECTOR_TIMESTAMP = 1790000000
VECTOR_NONCE = "ab" * 16
VECTOR_SIGNATURE = (
    "4nM5i/qa1NG9OOZDbW/O76bIWGWRCawJ4wVda+0tLz3hG9Ek7kRXp+oXIH9g2RxMQtQ2+DwHhauHllkcrwQfBw=="
)


def test_signed_bytes_match_the_shared_vector():
    key = Ed25519PrivateKey.from_private_bytes(VECTOR_SEED)
    assert hb.public_key_text(key) == VECTOR_PUBLIC_KEY
    signature = key.sign(hb.heartbeat_message(VECTOR_PUBLIC_KEY, VECTOR_TIMESTAMP, VECTOR_NONCE))
    assert base64.b64encode(signature).decode() == VECTOR_SIGNATURE


def test_build_heartbeat_verifies_with_the_reported_public_key():
    key = Ed25519PrivateKey.generate()
    body = hb.build_heartbeat(key, now=1790000123.9, version="1.2.3")

    assert body["timestamp"] == 1790000123
    assert len(body["nonce"]) == 32
    assert body["executor_version"] == "1.2.3"
    public = load_ssh_public_key(body["public_key"].encode())
    assert isinstance(public, Ed25519PublicKey)
    public.verify(
        base64.b64decode(body["signature"]),
        hb.heartbeat_message(body["public_key"], body["timestamp"], body["nonce"]),
    )


def test_every_heartbeat_has_a_fresh_nonce():
    key = Ed25519PrivateKey.generate()
    assert hb.build_heartbeat(key, now=1)["nonce"] != hb.build_heartbeat(key, now=1)["nonce"]


def test_host_key_is_read_from_next_to_the_public_key(tmp_path, monkeypatch):
    key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "ssh_host_ed25519_key"
    private_path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
    monkeypatch.setattr(hb.settings, "SSH_HOST_KEY_PATH", str(private_path) + ".pub")

    assert hb.host_private_key_path() == private_path
    loaded = hb.load_host_key(hb.host_private_key_path())
    assert hb.public_key_text(loaded) == hb.public_key_text(key)


def test_a_non_ed25519_host_key_is_refused(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ec

    path = tmp_path / "ssh_host_ecdsa_key"
    path.write_bytes(
        ec.generate_private_key(ec.SECP256R1()).private_bytes(
            Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()
        )
    )
    with pytest.raises(ValueError):
        hb.load_host_key(path)


def test_run_returns_at_once_when_disabled(monkeypatch):
    monkeypatch.setattr(hb.settings, "NODE_HEARTBEAT_ENABLED", False)
    with patch.object(hb, "load_host_key") as load:
        asyncio.run(asyncio.wait_for(hb.run_node_heartbeat(), timeout=1))
    load.assert_not_called()


def test_run_returns_when_the_host_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hb.settings, "NODE_HEARTBEAT_ENABLED", True)
    monkeypatch.setattr(hb.settings, "SSH_HOST_KEY_PATH", str(tmp_path / "missing.pub"))
    asyncio.run(asyncio.wait_for(hb.run_node_heartbeat(), timeout=1))
