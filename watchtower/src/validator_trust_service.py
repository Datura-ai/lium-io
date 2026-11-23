"""The validator whose digest signature this watchtower trusts, as named by a message the SN51 subnet owner signed.

The message is `lium-validator:<validator hotkey>:<revision>` (datura.validator_trust), the same one the executor and
the miner take. It comes from the digest endpoint's sibling route `<…>/watchtower/validator-message` every cycle,
before the digest is read, and from the image (`src/validator_trust_message.json`). A message is taken when the
current subnet owner's coldkey signed it and its revision is higher than the held one's; on any error the held
message stays. There is no expiry.

The watchtower container has no volume of its own, so the taken message lives in memory: a restart starts again
from the image copy. The digest comes from the same backend as the message, so a restart while it is unreachable has
no digest to check either; the next message read restores the current validator before the next digest is checked.

The owner coldkey starts as the address known at release and is re-read from the public chain nodes every cycle
until a read succeeds, then about once an hour with random jitter. A new owner is taken only when every public node
answers with it; a failed or disagreeing read keeps the known owner.

While no message is held, the watchtower trusts the hotkey baked into the image (config.py). A non-prod build
(config_override.py) names its own validator; when it also names a staging owner key (datura.validator_trust.override_trust) it follows messages that key signed,
starting from the build's own signed message, and never reads the chain.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from http import HTTPStatus
from pathlib import Path

import aiohttp
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

from config import WATCHTOWER_ENDPOINT_URL, WATCHTOWER_USER_AGENT
from logger import get_logger

logger = get_logger(__name__)

IMAGE_MESSAGE_PATH = Path(__file__).resolve().parent / "validator_trust_message.json"
# the digest endpoint is `<api>/watchtower/digest`; the message is served next to it
VALIDATOR_MESSAGE_URL = WATCHTOWER_ENDPOINT_URL.rsplit("/", 1)[0] + "/validator-message"
OWNER_REFRESH_SECONDS = 3600
FETCH_TIMEOUT_SECONDS = 5.0
FETCH_LIMIT_BYTES = 64 * 1024


def fetch_json(url: str, body: dict | None = None) -> dict | None:
    """The JSON a GET (or, with `body`, a POST) returns; None on a 404. The whole request (connect, headers and body)
    ends within FETCH_TIMEOUT_SECONDS and reads at most FETCH_LIMIT_BYTES, so a slow server cannot hold the update
    loop; raises on anything else. Synchronous for the watchtower loop; the bound is aiohttp's total timeout, as in the
    executor (a socket read timeout alone restarts on every short read)."""
    return asyncio.run(_fetch_json(url, body))


async def _fetch_json(url: str, body: dict | None) -> dict | None:
    # no decompression: the cap counts the bytes as they arrive, so a small compressed reply cannot expand past it
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
    headers = {"Accept-Encoding": "identity", "User-Agent": WATCHTOWER_USER_AGENT}
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
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


def read_subnet_owner_coldkey(url: str) -> str:
    """The subnet owner coldkey as one chain node reports it. Raises on any failure."""
    return subnet_owner_from_reply(fetch_json(url, subnet_owner_request()))


def _signed(data: dict) -> SignedTrustMessage:
    return SignedTrustMessage(message=str(data["message"]), signature=str(data["signature"]))


class ValidatorTrust:
    def __init__(
        self,
        image_message: Path = IMAGE_MESSAGE_PATH,
        enabled: bool = True,
        owner: str | None = None,
        read_chain: bool = True,
        message: SignedTrustMessage | None = None,
        api_url: str | None = None,
        network: str = "",
    ):
        self.image_message = image_message
        self.enabled = enabled
        # the key the start message is checked against; with read_chain off it is the only owner ever taken
        self.release_owner = owner or RELEASE_SUBNET_OWNER_COLDKEY
        self.read_chain = read_chain
        # the build's own signed message (a staging build), used instead of the image copy
        self.message = message
        # a staging build reads its owner's messages from the backend that serves them, not this host's backend
        self.api_url = api_url
        # the network this build's lines name (datura.validator_trust.trust_line): fixed at build time
        self.network = network
        self.owner = self.release_owner
        self.held: TrustedValidator | None = None
        self._owner_due = 0.0

    def load(self) -> None:
        """At start: the image copy, signed by the release owner. A missing or badly signed copy is ignored."""
        if not self.enabled:
            return
        try:
            start = self.message or _signed(json.loads(self.image_message.read_text()))
            self._hold(start, self.release_owner)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"Validator trust: image message refused: {type(exc).__name__}: {exc}")
        logger.info(f"Validator trust at start: {self.report()}")

    def _hold(self, signed: SignedTrustMessage, signer: str) -> bool:
        """Hold `signed` when `signer` signed it and its revision is higher than the held one's. Raises on a bad
        signature."""
        trusted = verify_trust_message(signed, signer, self.network)
        if self.held is not None and trusted.revision <= self.held.revision:
            return False
        self.held = trusted
        logger.warning(f"Validator trust: now trusting {trusted.validator} (revision {trusted.revision})")
        return True

    def offer(self, signed: SignedTrustMessage) -> bool:
        """Take a served message: signed by the current owner, and a higher revision than the held one."""
        if not self.enabled:
            return False
        try:
            return self._hold(signed, self.owner)
        except Exception as exc:
            logger.warning(f"Validator trust: message refused: {exc}")
            return False

    def hotkeys(self, baked: str) -> tuple[str, ...]:
        """The hotkeys a digest signature is checked against: the signed validator alone, else `baked`."""
        return (self.held.validator,) if self.held is not None else (baked,)

    def report(self) -> dict:
        return {
            "validator": self.held.validator if self.held else None,
            "revision": self.held.revision if self.held else None,
            "subnet_owner": self.owner if self.enabled else None,
        }

    def message_url(self) -> str:
        return self.api_url.rstrip("/") + "/watchtower/validator-message" if self.api_url else VALIDATOR_MESSAGE_URL

    def refresh_message(self) -> None:
        """A 404 or any failure keeps the message held."""
        if not self.enabled:
            return
        try:
            data = fetch_json(self.message_url())
            if data is not None:
                self.offer(_signed(data))
        except Exception as exc:
            url = self.message_url()
            logger.warning(f"Validator trust: message not refreshed from {url}: {type(exc).__name__}: {exc}")

    def refresh_owner(self) -> bool:
        """True when every public chain node answered with one owner (at least two nodes, so one cannot name an
        owner by itself); a failure or a disagreement keeps the known owner. A fixed owner is never read."""
        if not self.enabled or not self.read_chain:
            return False
        answers = set()
        for url in PUBLIC_CHAIN_NODES:
            try:
                answers.add(read_subnet_owner_coldkey(url))
            except Exception as exc:
                logger.warning(f"Validator trust: subnet owner not read from {url}: {type(exc).__name__}: {exc}")
                return False
        if len(answers) != 1:
            logger.warning(f"Validator trust: public chain nodes disagree on the subnet owner: {sorted(answers)}")
            return False
        (owner,) = answers
        if owner != self.owner:
            logger.warning(f"Validator trust: subnet owner changed from {self.owner} to {owner}")
            self.owner = owner
        return True

    def refresh(self) -> None:
        """Once per update cycle, before the digest is read: the owner when due, then the message."""
        if not self.enabled:
            return
        if time.monotonic() >= self._owner_due and self.refresh_owner():
            self._owner_due = time.monotonic() + random.uniform(0.5, 1.5) * OWNER_REFRESH_SECONDS
        self.refresh_message()


validator_trust = ValidatorTrust(**override_trust("config_override"))
