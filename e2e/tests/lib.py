"""Shared helpers for the lium-io e2e suites (run inside the tester container, PYTHONPATH=/app/src)."""

import json
import os
import time

import requests
from bittensor_wallet import Keypair
from datura.requests.validator_requests import AuthenticationPayload, fresh_upload_nonce, ssh_pubkey_signing_blob

ENV = os.environ
MINER_URL = f"http://{ENV['E2E_MINER_IP']}:{ENV['E2E_MINER_PORT']}"
EXECUTOR_URL = f"http://{ENV['E2E_EXECUTOR_IP']}:{ENV['E2E_EXECUTOR_PORT']}"
EXECUTOR_IP = ENV["E2E_EXECUTOR_IP"]
EXECUTOR_PORT = int(ENV["E2E_EXECUTOR_PORT"])
EXECUTOR_SSH_PORT = int(ENV["E2E_EXECUTOR_SSH_PORT"])
EXECUTOR_UUID = ENV["E2E_EXECUTOR_UUID"]
DEAD_EXECUTOR_UUID = ENV["E2E_DEAD_EXECUTOR_UUID"]
DEAD_EXECUTOR_IP = ENV["E2E_DEAD_EXECUTOR_IP"]
MINER_IP = ENV["E2E_MINER_IP"]
MINER_PORT = int(ENV["E2E_MINER_PORT"])
VALIDATOR_HOTKEY = ENV["E2E_VALIDATOR_HOTKEY"]
MINER_HOTKEY = ENV["E2E_MINER_HOTKEY"]
POOL_HOTKEY = ENV["E2E_POOL_HOTKEY"]
GPU = ENV.get("E2E_GPU", "") not in ("", "0", "false")
ARTIFACTS = "/e2e/artifacts"


def run(coro):
    """Run a coroutine on the tester's single event loop (tests/conftest.py)."""
    import asyncio

    return asyncio.get_event_loop().run_until_complete(coro)


def keypair(mnemonic: str) -> Keypair:
    return Keypair.create_from_mnemonic(mnemonic)


def miner_keypair() -> Keypair:
    return keypair(ENV["E2E_MINER_MNEMONIC"])


def stranger_keypair() -> Keypair:
    return keypair(ENV["E2E_STRANGER_MNEMONIC"])


def validator_keypair() -> Keypair:
    # the tester's wallet is the validator's (entrypoint.sh regenerated it from the same mnemonic)
    from core.config import settings

    return settings.get_bittensor_wallet().get_hotkey()


def sign(kp: Keypair, message: str) -> str:
    return f"0x{kp.sign(message).hex()}"


def validator_rest_headers(kp: Keypair, miner_hotkey: str = MINER_HOTKEY, timestamp: int | None = None) -> dict:
    """The four headers the validator sends the miner (MinerService._generate_auth_headers)."""
    payload = AuthenticationPayload(
        validator_hotkey=kp.ss58_address, miner_hotkey=miner_hotkey, timestamp=timestamp or int(time.time())
    )
    return {
        "X-Validator-Hotkey": kp.ss58_address,
        "X-Miner-Hotkey": miner_hotkey,
        "X-Timestamp": str(payload.timestamp),
        "X-Signature": sign(kp, payload.blob_for_signing()),
        "Content-Type": "application/json",
    }


def upload_ssh_key_payload(
    miner_kp: Keypair, validator_kp: Keypair, public_key: str, nonce: str | None = "", bound_to: str | None = MINER_HOTKEY
) -> dict:
    """What the miner posts to the executor's /upload_ssh_key (ExecutorService.send_pubkey_to_executor).

    `bound_to` is the miner hotkey the validator issued the key for (`validator_miner_signature`), as every validator
    upload carries; the stack's executor refuses a key without it (REQUIRE_MINER_BOUND_SSH_KEY, executor/e2e.env).
    `nonce` defaults to a fresh dated nonce, which executors require on upload; pass None for the legacy format."""
    if nonce == "":
        nonce = fresh_upload_nonce()
    body = {
        "public_key": public_key,
        "validator_signature": sign(validator_kp, ssh_pubkey_signing_blob(public_key, nonce)),
        "data_to_sign": public_key,
        "signature": sign(miner_kp, public_key),
    }
    if nonce is not None:
        body["nonce"] = nonce
    if bound_to is not None:
        body["miner_hotkey"] = bound_to
        body["validator_miner_signature"] = sign(validator_kp, ssh_pubkey_signing_blob(public_key, nonce, bound_to))
    return body


def validator_signed_upload_fields(validator_kp: Keypair, public_key: str) -> dict:
    """The validator-signed fields of an SSH-key upload (MinerService): validator_signature over the key and a fresh
    dated nonce, and validator_miner_signature binding the key to this miner's hotkey."""
    nonce = fresh_upload_nonce()
    return {
        "public_key": public_key,
        "validator_signature": sign(validator_kp, ssh_pubkey_signing_blob(public_key, nonce)),
        "nonce": nonce,
        "miner_hotkey": MINER_HOTKEY,
        "validator_miner_signature": sign(validator_kp, ssh_pubkey_signing_blob(public_key, nonce, MINER_HOTKEY)),
    }


def executor_sh(command: str, timeout: float = 60) -> tuple[int, str]:
    """Run one shell command on the executor over SSH (its docker CLI talks to the host's dockerd, where pods land);
    the key is installed for this command and removed after it."""
    priv, pub = ssh_keypair()
    body = upload_ssh_key_payload(miner_keypair(), validator_keypair(), pub)
    r = http("POST", f"{EXECUTOR_URL}/upload_ssh_key", json=body)
    assert r.status_code == 200, r.text
    try:
        return run(ssh_run(EXECUTOR_IP, r.json()["ssh_port"], r.json()["ssh_username"], priv, command, timeout=timeout))
    finally:
        http("POST", f"{EXECUTOR_URL}/remove_ssh_key", json=body)


def ssh_keypair() -> tuple[str, str]:
    """A fresh OpenSSH ed25519 key pair (private PEM text, public line) — what the validator mints per cycle."""
    import asyncssh

    key = asyncssh.generate_private_key("ssh-ed25519")
    return key.export_private_key().decode(), key.export_public_key().decode().strip()


async def ssh_run(host: str, port: int, username: str, private_key: str, command: str, timeout: float = 30) -> tuple[int, str]:
    """Run one command over SSH with a private key; returns (exit_status, stdout). Raises on auth failure."""
    import asyncio

    import asyncssh

    async with asyncio.timeout(timeout):
        async with asyncssh.connect(
            host, port=port, username=username, client_keys=[asyncssh.import_private_key(private_key)], known_hosts=None
        ) as conn:
            result = await conn.run(command, check=False)
            return result.exit_status, result.stdout


def wait_for(fn, timeout: float, interval: float = 2.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = fn()
            if last:
                return last
        except Exception as exc:  # noqa: BLE001 — the caller wants the condition, not the transient
            last = exc
        time.sleep(interval)
    raise AssertionError(f"{what} not met after {timeout}s (last: {last!r})")


def write_artifact(name: str, data) -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(ARTIFACTS, name), "w") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2, default=str)


def http(method: str, url: str, **kw) -> requests.Response:
    kw.setdefault("timeout", 30)
    return requests.request(method, url, **kw)
