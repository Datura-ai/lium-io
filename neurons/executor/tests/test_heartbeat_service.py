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
    # the backend refuses a replayed (key, nonce); a fixed nonce would make every second heartbeat a replay
    key = Ed25519PrivateKey.generate()
    nonces = {hb.build_heartbeat(key, now=1)["nonce"] for _ in range(20)}
    assert len(nonces) == 20
    assert all(len(n) == 32 and int(n, 16) >= 0 for n in nonces)


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


def test_an_interval_below_the_floor_is_clamped_not_rejected():
    from core.config import Settings

    assert Settings(NODE_HEARTBEAT_INTERVAL_SECONDS=5).NODE_HEARTBEAT_INTERVAL_SECONDS == 5
    with patch.object(hb.settings, "NODE_HEARTBEAT_INTERVAL_SECONDS", 5), patch.object(hb, "logger") as logger:
        assert hb.heartbeat_interval() == hb.MIN_INTERVAL_SECONDS
    logger.warning.assert_called_once()


def test_refused_answers_back_off_up_to_the_cap():
    assert hb.next_delay(60, 404, 1) == 120
    assert hb.next_delay(60, 401, 3) == 480
    assert hb.next_delay(60, 404, 20) == hb.MAX_BACKOFF_SECONDS
    assert 60 <= hb.next_delay(60, 204, 0) <= 70


@pytest.mark.asyncio
async def test_send_loop_logs_one_refusal_resets_on_success_and_survives_errors():
    key = Ed25519PrivateKey.generate()
    answers = iter([404, 404, RuntimeError("down"), 204, 500])
    delays = []

    async def fake_send(*_args):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def fake_sleep(seconds):
        delays.append(seconds)
        if len(delays) == 5:
            raise asyncio.CancelledError

    with patch.object(hb, "send_heartbeat", fake_send), patch.object(hb.asyncio, "sleep", fake_sleep), patch.object(
        hb, "logger"
    ) as logger:
        with pytest.raises(asyncio.CancelledError):
            await hb._send_loop(None, "http://x/v1/node-heartbeat", key, None, 60)

    assert delays[0] == 120 and delays[1] == 240
    assert all(60 <= d <= 70 for d in delays[2:])
    refusal_logs = [c for c in logger.info.call_args_list if "backing off" in c.args[0]]
    assert len(refusal_logs) == 1
