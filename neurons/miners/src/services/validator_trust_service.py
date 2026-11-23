"""The validator this miner trusts, as named by a message the SN51 subnet owner signed.

The message is `lium-validator:<validator hotkey>:<revision>` (datura.validator_trust), the same one the executor
takes. It comes from `<COMPUTE_REST_API_URL>/watchtower/validator-message` every few minutes, and from the image
(`src/validator_trust_message.json`). A message is taken when the current subnet owner's coldkey signed it and its
revision is higher than the held one's; on any error the held message stays. There is no expiry.

Every message taken is kept in the miner database (table validator_trust_message), so a restarted miner keeps
trusting the same validator with the backend and the chain both unreachable; at start the highest revision of the
stored ones and the image copy wins. A failed write is logged and tried again at the next refresh.

The executor rows keyed to a validator this miner trusted before (or to a baked hotkey) are served to the trusted
one, so a validator change needs no row migration. While no message is held, the miner trusts the baked hotkeys
(core.config). A non-prod build (core/config_override.py) names its own validator; when it also names a staging owner key (datura.validator_trust.override_trust) it follows messages that key signed,
starting from the build's own signed message, and never reads the chain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
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
from sqlalchemy import Engine
from sqlmodel import Session, select

from core.config import BAKED_VALIDATOR_HOTKEYS, settings
from core.db import engine as db_engine
from models.validator_trust import ValidatorTrustMessage

logger = logging.getLogger(__name__)

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


class ValidatorTrust:
    def __init__(
        self,
        engine: Engine | None = None,
        image_message: Path = IMAGE_MESSAGE_PATH,
        enabled: bool = True,
        owner: str | None = None,
        read_chain: bool = True,
        message: SignedTrustMessage | None = None,
    ):
        self.engine = engine or db_engine
        self.image_message = image_message
        self.enabled = enabled
        # shortcut: the owner is not stored, so after an owner change a restart trusts the release owner until a chain
        # read succeeds; store it next to the messages if that window matters.
        # the key the start message is checked against; with read_chain off it is the only owner ever taken
        self.release_owner = owner or RELEASE_SUBNET_OWNER_COLDKEY
        self.read_chain = read_chain
        # the build's own signed message (a staging build), used instead of the image copy
        self.message = message
        self.owner = self.release_owner
        self.held: TrustedValidator | None = None
        # every validator a message named that this miner took, the held one included
        self.trusted_before: set[str] = set()
        # messages taken and not in the database yet: written at the next refresh
        self._unsaved: list[ValidatorTrustMessage] = []

    def load(self) -> None:
        """At start: the stored messages, then the image copy (signed by the release owner) when its revision is
        higher. An unreadable table or a badly signed row is ignored."""
        if not self.enabled:
            return
        try:
            with Session(self.engine) as session:
                rows = session.exec(select(ValidatorTrustMessage).order_by(ValidatorTrustMessage.revision)).all()
        except Exception as exc:
            logger.warning("Validator trust: stored messages not read: %s: %s", type(exc).__name__, exc)
            rows = []
        for row in rows:
            try:
                # a fixed owner (no chain reads) takes only its own key's stored messages
                signer = row.signer if self.read_chain else self.owner
                self._hold(SignedTrustMessage(row.message, row.signature), signer, stored=True)
            except Exception as exc:
                logger.warning("Validator trust: stored message %d ignored: %s", row.revision, exc)
        try:
            start = self.message or _signed(json.loads(self.image_message.read_text()))
            self._hold(start, self.release_owner)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Validator trust: image message refused: %s: %s", type(exc).__name__, exc)
        logger.info("Validator trust at start: %s", self.report())

    def _hold(self, signed: SignedTrustMessage, signer: str, stored: bool = False) -> bool:
        """Hold `signed` when `signer` signed it and its revision is higher than the held one's. Raises on a bad
        signature."""
        trusted = verify_trust_message(signed, signer)
        if self.held is not None and trusted.revision <= self.held.revision:
            return False
        self.held = trusted
        self.trusted_before.add(trusted.validator)
        if not stored:
            self._unsaved.append(
                ValidatorTrustMessage(
                    revision=trusted.revision, message=signed.message, signature=signed.signature, signer=signer
                )
            )
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

    def _write(self, rows: list[ValidatorTrustMessage]) -> None:
        with Session(self.engine) as session:
            for row in rows:
                session.merge(row)
            session.commit()

    async def save(self) -> None:
        """Store the messages taken since the last save, off the event loop; a failure is retried next refresh."""
        pending = list(self._unsaved)
        if not pending:
            return
        try:
            await asyncio.to_thread(self._write, pending)
            del self._unsaved[: len(pending)]
        except Exception as exc:
            logger.warning("Validator trust: message not stored, retrying next refresh: %s: %s", type(exc).__name__, exc)

    def hotkeys(self) -> tuple[str, ...]:
        """The validator hotkeys whose sign-in this miner accepts: the signed validator alone, else the baked ones."""
        return (self.held.validator,) if self.held is not None else BAKED_VALIDATOR_HOTKEYS

    def validator(self) -> str:
        """The validator new executor rows are keyed to."""
        return self.hotkeys()[0]

    def served_keys(self, validator_hotkey: str) -> set[str]:
        """The executor row keys served to `validator_hotkey`: for the trusted validator also the rows keyed to a
        validator trusted before or to a baked hotkey; for any other validator only its own rows."""
        if validator_hotkey not in self.hotkeys():
            return {validator_hotkey}
        return {*self.hotkeys(), *BAKED_VALIDATOR_HOTKEYS, *self.trusted_before}

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
            self.owner = owner
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
