"""The validator this executor trusts, as named by a message the SN51 subnet owner signed (DAH-6590).

The message is `lium-validator:<validator hotkey>:<revision>` (datura.validator_trust). It comes from
`<COMPUTE_REST_API_URL>/watchtower/validator-message` every few minutes, and from the image
(`src/validator_trust_message.json`) when one was released with it. A message is taken when the current subnet
owner's coldkey signed it and its revision is higher than the held one's (one counter, whoever signed the held one);
on any error the held message stays.

The taken message, the key that signed it and the current owner are kept in one small file on the reserve_data
volume, which outlives a container recreate, so a restarted node keeps trusting the same validator with the backend
and the chain both unreachable. The file is written atomically, off the event loop, when a message or an owner is
taken; a failed write is logged and tried again at the next refresh. A missing, unreadable or badly signed file is
ignored: the node uses the image copy, else the baked hotkeys, and keeps polling.

The owner coldkey starts as the address known at release and is re-read from the public chain nodes at start, then
about once an hour with random jitter. A new owner is taken only when every public node answers with it; a failed or
disagreeing read keeps the known owner, whose messages are still taken (a chain outage never blocks a validator
change). An owner change only changes the key that verifies new messages: the held validator stays until the new
owner signs a higher revision.

While no message is held, the executor trusts the hotkeys baked into the image (core.config), as before. A non-prod
build (core/config_override.py) names its own validator; when it also names a staging owner key
(datura.validator_trust.override_trust) it follows messages that key signed, starting from the build's own signed
message, and never reads the chain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
from http import HTTPStatus
from pathlib import Path

import aiohttp
from bittensor_wallet import Keypair
from datura.validator_trust import (
    PUBLIC_CHAIN_NODES,
    RELEASE_SUBNET_OWNER_COLDKEY,
    SignedTrustMessage,
    TrustedValidator,
    override_trust,
    subnet_owner_from_reply,
    subnet_owner_request,
    verify_trust_message,
)

from core.config import settings

logger = logging.getLogger(__name__)

STATE_DIR = Path("/var/lium-reserve")
STATE_FILE = "validator_trust.json"
IMAGE_MESSAGE_PATH = Path(__file__).resolve().parents[1] / "validator_trust_message.json"
MESSAGE_REFRESH_SECONDS = 300
OWNER_REFRESH_SECONDS = 3600
# each request as a whole (connect, headers and body), so a server that answers slowly cannot hold the loop
FETCH_TIMEOUT_SECONDS = 5.0
FETCH_LIMIT_BYTES = 64 * 1024


async def fetch_json(url: str, body: dict | None = None) -> dict | None:
    """The JSON a GET (or, with `body`, a POST) returns; None on a 404. Ends within FETCH_TIMEOUT_SECONDS and reads at
    most FETCH_LIMIT_BYTES; raises on anything else."""
    # no decompression: the cap counts the bytes as they arrive, so a small compressed reply cannot expand past it
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
        headers = {"Accept-Encoding": "identity"}
        async with session.request("GET" if body is None else "POST", url, json=body, headers=headers) as response:
            if response.status == HTTPStatus.NOT_FOUND:
                return None
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError(f"encoded reply: {response.headers['Content-Encoding']}")
            data = bytearray()
            async for chunk in response.content.iter_any():
                data += chunk
                if len(data) > FETCH_LIMIT_BYTES:
                    raise ValueError(f"reply over {FETCH_LIMIT_BYTES} bytes")
            return json.loads(data)


async def read_subnet_owner_coldkey(url: str) -> str:
    """The subnet owner coldkey as one chain node reports it. Raises on any failure."""
    return subnet_owner_from_reply(await fetch_json(url, subnet_owner_request()))


def _signed(data: dict) -> SignedTrustMessage:
    return SignedTrustMessage(message=str(data["message"]), signature=str(data["signature"]))


def _write_json(path: Path, data: dict) -> None:
    """Atomic and durable: a crash mid-write leaves the previous file whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ValidatorTrust:
    def __init__(
        self,
        state_dir: Path = STATE_DIR,
        image_message: Path = IMAGE_MESSAGE_PATH,
        enabled: bool = True,
        owner: str | None = None,
        read_chain: bool = True,
        message: SignedTrustMessage | None = None,
    ):
        self.state_dir = state_dir
        self.image_message = image_message
        self.enabled = enabled
        # the key the start message is checked against; with read_chain off it is the only owner ever taken
        self.release_owner = owner or RELEASE_SUBNET_OWNER_COLDKEY
        self.read_chain = read_chain
        # the build's own signed message (a staging build), used instead of the image copy
        self.message = message
        self.owner = self.release_owner
        self.held: TrustedValidator | None = None
        self.held_signed: SignedTrustMessage | None = None
        # the key that signed the held message: an owner change keeps the message until the new owner signs a higher one
        self.held_signer: str | None = None
        # the state changed and is not on the volume yet: written again at the next refresh
        self._unsaved = False

    def load(self) -> None:
        """At start: the kept state, then the image copy (signed by the release owner) when its revision is higher.
        A missing, unreadable or badly signed state is ignored."""
        if not self.enabled:
            return
        try:
            state = json.loads((self.state_dir / STATE_FILE).read_text())
            Keypair(ss58_address=state["owner"])
            # a fixed owner (no chain reads) takes only its own key's messages from the volume
            signer = state["signer"] if self.read_chain else self.owner
            if state["message"] is not None:
                self._hold(_signed(state["message"]), signer)
            if self.read_chain:
                self.owner = state["owner"]
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Validator trust: kept state ignored: %s: %s", type(exc).__name__, exc)
            self.held = self.held_signed = self.held_signer = None
        # the kept state is on the volume already; an image message taken over it is not, so the next refresh writes it
        self._unsaved = False
        try:
            start = self.message or _signed(json.loads(self.image_message.read_text()))
            self._hold(start, self.release_owner)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Validator trust: image message refused: %s: %s", type(exc).__name__, exc)
        logger.info("Validator trust at start: %s", self.report())

    def _hold(self, signed: SignedTrustMessage, signer: str) -> bool:
        """Hold `signed` when `signer` signed it and its revision is higher than the held one's. Raises on a bad
        signature."""
        trusted = verify_trust_message(signed, signer)
        if self.held is not None and trusted.revision <= self.held.revision:
            return False
        self.held, self.held_signed, self.held_signer = trusted, signed, signer
        self._unsaved = True
        logger.warning("Validator trust: now trusting %s (revision %d)", trusted.validator, trusted.revision)
        return True

    def offer(self, signed: SignedTrustMessage) -> bool:
        """Take a served message: signed by the current owner, and a higher revision than the held one."""
        if not self.enabled:
            return False
        try:
            return self._hold(signed, self.owner)
        except Exception as exc:
            logger.warning("Validator trust: message refused: %s", exc)
            return False

    def _state(self) -> dict:
        return {
            "owner": self.owner,
            "signer": self.held_signer,
            "message": self.held_signed._asdict() if self.held_signed else None,
        }

    async def save(self) -> None:
        """Write the state if it changed, off the event loop; a failure is logged and tried again next refresh."""
        if not self._unsaved:
            return
        try:
            await asyncio.to_thread(_write_json, self.state_dir / STATE_FILE, self._state())
            self._unsaved = False
        except OSError as exc:
            logger.warning("Validator trust: state not kept on the volume, retrying next refresh: %s", exc)

    def hotkeys(self, baked: dict[str, str]) -> dict[str, str]:
        """Key id -> ss58 the executor accepts signatures from: the signed validator alone, else `baked`."""
        return {"signed": self.held.validator} if self.held is not None else baked

    def report(self) -> dict:
        return {
            "validator": self.held.validator if self.held else None,
            "revision": self.held.revision if self.held else None,
            "subnet_owner": self.owner if self.enabled else None,
        }

    async def refresh_message(self) -> None:
        """A 404 or any failure keeps the message held."""
        if not self.enabled or not settings.COMPUTE_REST_API_URL:
            return
        url = settings.COMPUTE_REST_API_URL.rstrip("/") + "/watchtower/validator-message"
        try:
            data = await fetch_json(url)
            if data is not None:
                self.offer(_signed(data))
        except Exception as exc:
            logger.warning("Validator trust: message not refreshed from %s: %s: %s", url, type(exc).__name__, exc)
        await self.save()

    async def refresh_owner(self) -> bool:
        """True when every public chain node answered with one owner (at least two nodes, so one cannot name an
        owner by itself); a failure or a disagreement keeps the known owner. A fixed owner is never read."""
        if not self.enabled or not self.read_chain:
            return False
        answers = set()
        for url in PUBLIC_CHAIN_NODES:
            try:
                answers.add(await read_subnet_owner_coldkey(url))
            except Exception as exc:
                logger.warning("Validator trust: subnet owner not read from %s: %s: %s", url, type(exc).__name__, exc)
                return False
        if len(answers) != 1:
            logger.warning("Validator trust: public chain nodes disagree on the subnet owner: %s", sorted(answers))
            return False
        (owner,) = answers
        if owner != self.owner:
            logger.warning("Validator trust: subnet owner changed from %s to %s", self.owner, owner)
            self.owner, self._unsaved = owner, True
            await self.save()
        return True


validator_trust = ValidatorTrust(**override_trust("core.config_override"))


async def run_validator_trust_refresh() -> None:
    """Background loop: the message every MESSAGE_REFRESH_SECONDS; the owner every round until a read succeeds,
    then about hourly with random jitter."""
    if not validator_trust.enabled:
        return
    owner_due = asyncio.get_running_loop().time()
    while True:
        if asyncio.get_running_loop().time() >= owner_due and await validator_trust.refresh_owner():
            owner_due = asyncio.get_running_loop().time() + random.uniform(0.5, 1.5) * OWNER_REFRESH_SECONDS
        await validator_trust.refresh_message()
        await asyncio.sleep(MESSAGE_REFRESH_SECONDS)
