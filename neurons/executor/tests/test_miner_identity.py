"""GET /miner-identity: the executor names the miner hotkey it is configured for and echoes the
caller's nonce, without a signature (the hotkey is public on chain)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router

from core.config import settings

NONCE = "a1" * 16


def _client() -> TestClient:
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    return TestClient(app)


def test_returns_configured_hotkey_and_echoes_nonce():
    response = _client().get("/miner-identity", params={"nonce": NONCE})

    assert response.status_code == 200
    assert response.json() == {"miner_hotkey": settings.MINER_HOTKEY_SS58_ADDRESS, "nonce": NONCE}  # no claim set


def test_follows_the_configured_hotkey(monkeypatch):
    monkeypatch.setattr(
        settings, "MINER_HOTKEY_SS58_ADDRESS", "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    )

    response = _client().get("/miner-identity", params={"nonce": NONCE})

    assert response.json()["miner_hotkey"] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def test_carries_the_account_claim_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "MINER_ACCOUNT_CLAIM", "c1" * 16)

    response = _client().get("/miner-identity", params={"nonce": NONCE})

    assert response.json() == {
        "miner_hotkey": settings.MINER_HOTKEY_SS58_ADDRESS,
        "nonce": NONCE,
        "account_claim": "c1" * 16,
    }


def test_nonce_is_required():
    assert _client().get("/miner-identity").status_code == 422


def test_short_or_non_hex_nonce_is_refused():
    client = _client()
    assert client.get("/miner-identity", params={"nonce": "ab" * 8}).status_code == 422
    assert client.get("/miner-identity", params={"nonce": "zz" * 16}).status_code == 422
    assert client.get("/miner-identity", params={"nonce": "ab" * 65}).status_code == 422


def _keys_dir(monkeypatch, tmp_path, *, mounted: bool):
    (tmp_path / "ssh_host_ed25519_key.pub").write_text("ssh-ed25519 AAAA")
    monkeypatch.setattr(settings, "SSH_HOST_KEYS_DIR", str(tmp_path))
    monkeypatch.setattr("routes.apis.os.path.ismount", lambda path: mounted and Path(path) == tmp_path)


def test_reports_the_ssh_port_when_host_keys_persist(monkeypatch, tmp_path):
    _keys_dir(monkeypatch, tmp_path, mounted=True)
    monkeypatch.setattr(settings, "SSH_PUBLIC_PORT", None)
    assert _client().get("/miner-identity", params={"nonce": NONCE}).json()["ssh_host_key_port"] == settings.SSH_PORT

    monkeypatch.setattr(settings, "SSH_PUBLIC_PORT", 40022)
    assert _client().get("/miner-identity", params={"nonce": NONCE}).json()["ssh_host_key_port"] == 40022


def test_no_ssh_port_when_host_keys_do_not_persist(monkeypatch, tmp_path):
    # CVM: ssh_host_keys.sh fills the dir, but it is not a volume
    _keys_dir(monkeypatch, tmp_path, mounted=False)
    assert "ssh_host_key_port" not in _client().get("/miner-identity", params={"nonce": NONCE}).json()


def test_no_ssh_port_when_the_volume_holds_no_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SSH_HOST_KEYS_DIR", str(tmp_path))
    monkeypatch.setattr("routes.apis.os.path.ismount", lambda path: True)
    assert "ssh_host_key_port" not in _client().get("/miner-identity", params={"nonce": NONCE}).json()



def _signing_keys_dir(monkeypatch, tmp_path, *, mounted: bool):
    """sshd's three host key types, as ssh-keygen -A writes them; returns each public key by its OpenSSH line."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

    keys = {
        "ed25519": ed25519.Ed25519PrivateKey.generate(),
        "ecdsa": ec.generate_private_key(ec.SECP256R1()),
        "rsa": rsa.generate_private_key(public_exponent=65537, key_size=2048),
    }
    publics = {}
    for name, key in keys.items():
        (tmp_path / f"ssh_host_{name}_key").write_bytes(
            key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption())
        )
        public = key.public_key()
        publics[public.public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()] = public
    (tmp_path / "ssh_host_ed25519_key.pub").write_text("not read")
    monkeypatch.setattr(settings, "SSH_HOST_KEYS_DIR", str(tmp_path))
    monkeypatch.setattr("routes.apis.os.path.ismount", lambda path: mounted and Path(path) == tmp_path)
    return publics


def _verifies(public, signature: bytes, message: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding

    try:
        if isinstance(public, ed25519.Ed25519PublicKey):
            public.verify(signature, message)
        elif isinstance(public, ec.EllipticCurvePublicKey):
            public.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        else:
            public.verify(signature, message, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature:
        return False
    return True


def test_every_host_key_signs_nonce_hotkey_and_claim(monkeypatch, tmp_path):
    import base64

    from routes.apis import identity_proof_message

    publics = _signing_keys_dir(monkeypatch, tmp_path, mounted=True)
    monkeypatch.setattr(settings, "MINER_ACCOUNT_CLAIM", "c1" * 16)

    proofs = _client().get("/miner-identity", params={"nonce": NONCE}).json()["host_key_proofs"]

    assert set(proofs) == set(publics)  # one per key type, keyed by the line ssh-keyscan shows
    message = identity_proof_message(NONCE, settings.MINER_HOTKEY_SS58_ADDRESS, "c1" * 16)
    other_claim = identity_proof_message(NONCE, settings.MINER_HOTKEY_SS58_ADDRESS, "d2" * 16)
    for line, public in publics.items():
        signature = base64.b64decode(proofs[line])
        assert _verifies(public, signature, message), line
        assert not _verifies(public, signature, other_claim), line


def test_no_host_key_proofs_without_persisted_keys(monkeypatch, tmp_path):
    _signing_keys_dir(monkeypatch, tmp_path, mounted=False)
    assert "host_key_proofs" not in _client().get("/miner-identity", params={"nonce": NONCE}).json()


def test_no_host_key_proofs_when_no_private_key_can_be_read(monkeypatch, tmp_path):
    _keys_dir(monkeypatch, tmp_path, mounted=True)  # only the .pub
    body = _client().get("/miner-identity", params={"nonce": NONCE}).json()
    assert body["ssh_host_key_port"] and "host_key_proofs" not in body


def test_host_keys_are_parsed_once_per_file_version(monkeypatch, tmp_path):
    import os

    from routes import apis

    publics = _signing_keys_dir(monkeypatch, tmp_path, mounted=True)
    apis._load_host_key.cache_clear()
    for _ in range(3):
        assert set(_client().get("/miner-identity", params={"nonce": NONCE}).json()["host_key_proofs"]) == set(publics)
    assert apis._load_host_key.cache_info().misses == 3  # one parse per key, not per request

    rsa_key = tmp_path / "ssh_host_rsa_key"
    os.utime(rsa_key, ns=(rsa_key.stat().st_atime_ns, rsa_key.stat().st_mtime_ns + 1))
    _client().get("/miner-identity", params={"nonce": NONCE})
    assert apis._load_host_key.cache_info().misses == 4  # a rewritten key file is read again
