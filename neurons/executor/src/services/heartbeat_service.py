"""Signed node heartbeat: tell the backend this node is up every NODE_HEARTBEAT_INTERVAL_SECONDS.

The heartbeat is signed with the private half of this host's SSH ed25519 host key. The validator already pins the
public half every time it connects, and reports it to the backend, so the backend trusts a heartbeat only for a key
a validator has seen on this node. Validator checks continue as before; the heartbeat adds uptime resolution
between them.

The signed bytes are `HEARTBEAT_DOMAIN \\n public_key \\n timestamp \\n nonce`; the backend builds the same string.
Off unless NODE_HEARTBEAT_ENABLED is set. A failed send is logged and retried at the next interval, never raised.
"""

import asyncio
import base64
import random
import secrets
import time
from pathlib import Path

import aiohttp
from core.logger import get_logger
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_ssh_private_key,
)

from core.config import settings

logger = get_logger(__name__)

HEARTBEAT_DOMAIN = "lium-node-heartbeat/v1"
HEARTBEAT_PATH = "/v1/node-heartbeat"
REQUEST_TIMEOUT_SECONDS = 10


def heartbeat_message(public_key: str, timestamp: int, nonce: str) -> bytes:
    return f"{HEARTBEAT_DOMAIN}\n{public_key}\n{timestamp}\n{nonce}".encode()


def host_private_key_path() -> Path:
    """The private host key next to SSH_HOST_KEY_PATH (`…/ssh_host_ed25519_key.pub` → `…/ssh_host_ed25519_key`)."""
    public_path = settings.SSH_HOST_KEY_PATH
    return Path(public_path[: -len(".pub")] if public_path.endswith(".pub") else public_path)


def load_host_key(path: Path) -> Ed25519PrivateKey:
    key = load_ssh_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an ed25519 key")
    return key


def public_key_text(key: Ed25519PrivateKey) -> str:
    """`ssh-ed25519 <base64>`, the form the validator reports (without the comment field)."""
    return key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()


def build_heartbeat(
    key: Ed25519PrivateKey, *, now: float | None = None, version: str | None = None
) -> dict:
    public_key = public_key_text(key)
    timestamp = int(now if now is not None else time.time())
    nonce = secrets.token_hex(16)
    signature = base64.b64encode(key.sign(heartbeat_message(public_key, timestamp, nonce))).decode()
    return {
        "public_key": public_key,
        "timestamp": timestamp,
        "nonce": nonce,
        "signature": signature,
        "executor_version": version,
    }


def _executor_version() -> str | None:
    try:
        return (Path(__file__).resolve().parents[2] / "version.txt").read_text().strip() or None
    except Exception:
        return None


async def send_heartbeat(
    session: aiohttp.ClientSession, url: str, key: Ed25519PrivateKey, version: str | None
) -> int:
    async with session.post(
        url,
        json=build_heartbeat(key, version=version),
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
    ) as response:
        return response.status


async def run_node_heartbeat() -> None:
    """Background task started from the app lifespan. Returns at once when the feature is off or has no key."""
    if not settings.NODE_HEARTBEAT_ENABLED or not settings.COMPUTE_REST_API_URL:
        return
    path = host_private_key_path()
    try:
        key = load_host_key(path)
    except Exception as e:
        logger.warning(f"Node heartbeat disabled: cannot read the SSH host key at {path}: {e}")
        return

    url = settings.COMPUTE_REST_API_URL.rstrip("/") + HEARTBEAT_PATH
    interval = max(10, settings.NODE_HEARTBEAT_INTERVAL_SECONDS)
    version = _executor_version()
    # spread a fleet that restarts together over one interval
    await asyncio.sleep(random.uniform(0, interval))
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                status = await send_heartbeat(session, url, key, version)
                if status != 204:
                    logger.info(f"Node heartbeat answered {status}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.info(f"Node heartbeat not sent: {type(e).__name__}")
            await asyncio.sleep(interval + random.uniform(0, 10))
