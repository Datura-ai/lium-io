"""The validator this executor trusts, as named by a message the SN51 subnet owner signed (DAH-6590).

The message is `lium-validator:<validator hotkey>:<revision>` (datura.validator_trust). It comes from
`<COMPUTE_REST_API_URL>/watchtower/validator-message` every few minutes, and from the image
(`src/validator_trust_message.json`) when one was released with it. A message is taken when the subnet
owner's hotkey signed it and its revision is higher than that owner's highest one known (revisions count per
owner: after an owner change on chain, the new owner's first message is taken); on any error the last good one
stays. The node keeps, for every owner, the highest message it has verified from that owner, so an owner that
returns never falls back to an older revision of its own.

The owner hotkey starts as the address known at release and is re-read from public chain nodes at start, then
about once an hour with random jitter; a change is taken only when all public nodes report it. Until one read has
succeeded since start, no message is taken from the backend, so a restarted node never takes a former owner's
message. The owner, the held message's owner and the messages are kept together in one file on the reserve_data
volume (written atomically), which outlives a container recreate, so a restarted node keeps trusting the same
validator with the backend and the chain both unreachable.

While no message is held, the executor trusts the hotkeys baked into the image (core.config), as
before. A non-prod build (core/config_override.py) names its own validator and never reads a message.
"""

from __future__ import annotations

import asyncio
import importlib.util
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
    RELEASE_SUBNET_OWNER_HOTKEY,
    SignedTrustMessage,
    TrustedValidator,
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


async def read_subnet_owner_hotkey(url: str) -> str:
    """The subnet owner hotkey as one chain node reports it. Raises on any failure."""
    return subnet_owner_from_reply(await fetch_json(url, subnet_owner_request()))


def _write_json(path: Path, data: dict) -> None:
    """Atomic: a crash mid-write leaves the previous file whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(data, file)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ValidatorTrust:
    def __init__(self, state_dir: Path = STATE_DIR, image_message: Path = IMAGE_MESSAGE_PATH, enabled: bool = True):
        self.state_dir = state_dir
        self.image_message = image_message
        self.enabled = enabled
        self.owner = RELEASE_SUBNET_OWNER_HOTKEY
        # set by the first chain read that names one owner; the backend's messages wait for it
        self.owner_confirmed = False
        self.held: TrustedValidator | None = None
        # the owner whose signature the held message carries: an owner change keeps it until the new owner signs
        self.held_owner: str | None = None
        # every owner's highest verified message: the revision floor per owner, kept across owner changes
        self.messages: dict[str, tuple[TrustedValidator, SignedTrustMessage]] = {}
        # the volume refused the state (full, read-only): written again on every refresh until it lands
        self._unsaved = False

    def load(self) -> None:
        """At start: the kept state (owner, held owner, messages) and the image copy, signed by the owner at release."""
        if not self.enabled:
            return
        held_owner = None
        try:
            state = json.loads((self.state_dir / STATE_FILE).read_text())
            Keypair(ss58_address=state["owner"])
            self.owner, held_owner = state["owner"], state.get("held")
            kept = [(SignedTrustMessage(**signed), signer) for signer, signed in state.get("messages", {}).items()]
        except FileNotFoundError:
            kept = []
        except Exception as exc:
            logger.warning("Validator trust: kept state unreadable, using the release owner: %s: %s", type(exc).__name__, exc)
            kept = []
        try:
            image = json.loads(self.image_message.read_text())
            kept.append((SignedTrustMessage(message=str(image["message"]), signature=str(image["signature"])), RELEASE_SUBNET_OWNER_HOTKEY))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Validator trust: image message unreadable: %s: %s", type(exc).__name__, exc)
        for signed, signer in kept:
            try:
                self._remember(verify_trust_message(signed, signer), signed, signer)
            except Exception as exc:
                logger.warning("Validator trust: kept message refused: %s", exc)
        self.held_owner = held_owner
        self._select()
        logger.info("Validator trust at start: %s", self.report())

    def _remember(self, trusted: TrustedValidator, signed: SignedTrustMessage, signer: str) -> bool:
        """Keep `signed` as `signer`'s message when its revision is above the highest one known from that signer."""
        known = self.messages.get(signer)
        if known is not None and trusted.revision <= known[0].revision:
            return False
        self.messages[signer] = (trusted, signed)
        return True

    def _select(self) -> None:
        """Hold the current owner's message; without one, the held owner's (what this node last trusted), else the
        image copy's. Run at start and after every owner change."""
        for signer in (self.owner, self.held_owner, RELEASE_SUBNET_OWNER_HOTKEY):
            if signer in self.messages:
                break
        else:
            return
        self._hold(signer)

    def _hold(self, signer: str) -> None:
        trusted = self.messages[signer][0]
        if (trusted, signer) != (self.held, self.held_owner):
            self.held, self.held_owner = trusted, signer
            logger.warning("Validator trust: now trusting %s (revision %d)", trusted.validator, trusted.revision)
        self._save()

    def offer(self, signed: SignedTrustMessage) -> bool:
        """Take a served message: signed by the current owner, and a higher revision than that owner's highest."""
        if not self.enabled:
            return False
        try:
            trusted = verify_trust_message(signed, self.owner)
        except Exception as exc:
            logger.warning("Validator trust: message refused: %s", exc)
            return False
        if not self._remember(trusted, signed, self.owner):
            return False
        self._hold(self.owner)
        return True

    def _save(self) -> None:
        """One atomic file for the owner, the held owner and the messages: a restart never reads them apart."""
        state = {
            "owner": self.owner,
            "held": self.held_owner,
            "messages": {signer: signed._asdict() for signer, (_, signed) in self.messages.items()},
        }
        try:
            _write_json(self.state_dir / STATE_FILE, state)
            self._unsaved = False
        except OSError as exc:
            # trusted for this run; written again on the next refresh, so a recreate does not lose it
            self._unsaved = True
            logger.warning("Validator trust: state not kept on disk yet: %s", exc)

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
        """A 404 or any failure keeps the message held. Volume writes run off the event loop."""
        if not self.enabled:
            return
        if self._unsaved:
            await asyncio.to_thread(self._save)
        if not self.owner_confirmed or not settings.COMPUTE_REST_API_URL:
            return
        url = settings.COMPUTE_REST_API_URL.rstrip("/") + "/watchtower/validator-message"
        try:
            data = await fetch_json(url)
            if data is None:
                return
            signed = SignedTrustMessage(message=str(data["message"]), signature=str(data["signature"]))
            await asyncio.to_thread(self.offer, signed)
        except Exception as exc:
            logger.warning("Validator trust: message not refreshed from %s: %s: %s", url, type(exc).__name__, exc)

    async def refresh_owner(self) -> None:
        """A new owner is taken only when every public chain node answers with it (at least two), so one
        node cannot name an owner by itself; any failure or disagreement keeps the last one and takes no served
        message until a later read agrees."""
        if not self.enabled:
            return
        self.owner_confirmed = False
        answers = set()
        for url in PUBLIC_CHAIN_NODES:
            try:
                answers.add(await read_subnet_owner_hotkey(url))
            except Exception as exc:
                logger.warning("Validator trust: subnet owner not read from %s: %s: %s", url, type(exc).__name__, exc)
                return
        if len(answers) != 1:
            logger.warning("Validator trust: public chain nodes disagree on the subnet owner: %s", sorted(answers))
            return
        (owner,) = answers
        self.owner_confirmed = True
        if owner != self.owner:
            logger.warning("Validator trust: subnet owner changed from %s to %s", self.owner, owner)
            self.owner = owner
            self._select()
            await asyncio.to_thread(self._save)


validator_trust = ValidatorTrust(enabled=importlib.util.find_spec("core.config_override") is None)


async def run_validator_trust_refresh() -> None:
    """Background loop: the message every MESSAGE_REFRESH_SECONDS; the owner every round until a read succeeds,
    then about hourly with random jitter."""
    if not validator_trust.enabled:
        return
    owner_due = asyncio.get_running_loop().time()
    while True:
        if asyncio.get_running_loop().time() >= owner_due:
            await validator_trust.refresh_owner()
            if validator_trust.owner_confirmed:
                owner_due = asyncio.get_running_loop().time() + random.uniform(0.5, 1.5) * OWNER_REFRESH_SECONDS
        await validator_trust.refresh_message()
        await asyncio.sleep(MESSAGE_REFRESH_SECONDS)
