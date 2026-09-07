"""Shared helpers for the lium-io e2e suites (run inside the tester container, PYTHONPATH=/app/src)."""

import json
import os
import time

import requests
from bittensor_wallet import Keypair
from datura.requests.validator_requests import AuthenticationPayload, ssh_pubkey_signing_blob

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
GPU = ENV.get("E2E_GPU", "") not in ("", "0", "false")
ARTIFACTS = "/e2e/artifacts"


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


def upload_ssh_key_payload(miner_kp: Keypair, validator_kp: Keypair, public_key: str, nonce: str | None = None) -> dict:
    """What the miner posts to the executor's /upload_ssh_key (ExecutorService.send_pubkey_to_executor)."""
    body = {
        "public_key": public_key,
        "validator_signature": sign(validator_kp, ssh_pubkey_signing_blob(public_key, nonce)),
        "data_to_sign": public_key,
        "signature": sign(miner_kp, public_key),
    }
    if nonce is not None:
        body["nonce"] = nonce
    return body


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
