"""The validator this executor trusts, as named by a message the SN51 subnet owner signed (DAH-6590).

The message is `lium-validator:<validator hotkey>:<revision>` (datura.validator_trust). It comes from
`<COMPUTE_REST_API_URL>/watchtower/validator-message` every few minutes, and from the image
(`src/validator_trust_message.json`) when one was released with it. A message is taken when the subnet
owner's hotkey signed it and its revision is higher than the one held (revisions count per owner: after an
owner change on chain, the new owner's first message is taken); on any error the last good one stays. The accepted message is kept on the reserve_data volume, which outlives a container recreate,
so a restarted node keeps trusting it with the backend and the chain both unreachable.

The owner hotkey starts as the address known at release and is re-read from public chain nodes once
an hour at a random offset; a change is taken only when all public nodes report it. It is kept on the same
volume, and each kept message records the owner that signed it.

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

import requests
from bittensor_wallet import Keypair

from datura.validator_trust import (
    PUBLIC_CHAIN_NODES,
    RELEASE_SUBNET_OWNER_HOTKEY,
    SignedTrustMessage,
    TrustedValidator,
    read_subnet_owner_hotkey,
    verify_trust_message,
)

from core.config import settings

logger = logging.getLogger(__name__)

STATE_DIR = Path("/var/lium-reserve")
MESSAGE_FILE = "validator_trust.json"
OWNER_FILE = "subnet_owner.json"
IMAGE_MESSAGE_PATH = Path(__file__).resolve().parents[1] / "validator_trust_message.json"
MESSAGE_REFRESH_SECONDS = 300
OWNER_REFRESH_SECONDS = 3600
FETCH_TIMEOUT_SECONDS = 5.0


def _read_signed(path: Path) -> SignedTrustMessage | None:
    try:
        data = json.loads(path.read_text())
        return SignedTrustMessage(message=str(data["message"]), signature=str(data["signature"]))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Validator trust: ignoring unreadable %s: %s: %s", path, type(exc).__name__, exc)
        return None


def _read_signer(path: Path) -> str | None:
    try:
        return str(json.loads(path.read_text())["owner"])
    except Exception:
        return None  # a file from before the owner was recorded: checked against the current owner


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
        self.held: TrustedValidator | None = None
        # the owner whose signature the held message carries: kept with it, so a later owner change on chain
        # does not void it, and the revision floor applies per owner
        self.held_owner: str | None = None
        # state the volume refused (full, read-only): written again on every refresh until it lands
        self._unsaved: dict[str, dict] = {}

    def load(self) -> None:
        """At start: the owner kept on disk, then the higher revision of the image copy and the kept message.

        Each is checked against the owner that signed it: the image copy against the owner at release, the kept
        message against the owner recorded with it when it was accepted."""
        if not self.enabled:
            return
        try:
            owner = json.loads((self.state_dir / OWNER_FILE).read_text())["owner"]
            Keypair(ss58_address=owner)
            self.owner = owner
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Validator trust: kept owner unreadable, using the release one: %s", exc)
        image = _read_signed(self.image_message)
        if image is not None:
            self._take(image, RELEASE_SUBNET_OWNER_HOTKEY)
        kept = _read_signed(self.state_dir / MESSAGE_FILE)
        if kept is not None:
            self._take(kept, _read_signer(self.state_dir / MESSAGE_FILE) or self.owner)
        logger.info("Validator trust at start: %s", self.report())

    def offer(self, signed: SignedTrustMessage) -> bool:
        """Take a served message: signed by the current owner, and a higher revision than the held one unless the
        held one carries another owner's signature (a new owner starts its own revisions)."""
        if not self.enabled:
            return False
        if not self._take(signed, self.owner):
            return False
        self._save(MESSAGE_FILE, {**signed._asdict(), "owner": self.owner})
        return True

    def _take(self, signed: SignedTrustMessage, signer: str) -> bool:
        try:
            trusted = verify_trust_message(signed, signer)
        except Exception as exc:
            logger.warning("Validator trust: message refused: %s", exc)
            return False
        if self.held is not None and self.held_owner == signer and trusted.revision <= self.held.revision:
            return False
        self.held, self.held_owner = trusted, signer
        logger.warning("Validator trust: now trusting %s (revision %d)", trusted.validator, trusted.revision)
        return True

    def _save(self, name: str, data: dict) -> None:
        self._unsaved[name] = data
        self._save_unsaved()

    def _save_unsaved(self) -> None:
        for name, data in list(self._unsaved.items()):
            try:
                _write_json(self.state_dir / name, data)
            except OSError as exc:
                # trusted for this run; written again on the next refresh, so a recreate does not lose it
                logger.warning("Validator trust: %s not kept on disk yet: %s", name, exc)
                continue
            del self._unsaved[name]

    def hotkeys(self, baked: dict[str, str]) -> dict[str, str]:
        """Key id -> ss58 the executor accepts signatures from: the signed validator alone, else `baked`."""
        return {"signed": self.held.validator} if self.held is not None else baked

    def report(self) -> dict:
        return {
            "validator": self.held.validator if self.held else None,
            "revision": self.held.revision if self.held else None,
            "subnet_owner": self.owner if self.enabled else None,
        }

    def refresh_message(self) -> None:
        """Blocking: run it off the event loop. A 404 or any failure keeps the message held."""
        if not self.enabled:
            return
        self._save_unsaved()
        if not settings.COMPUTE_REST_API_URL:
            return
        url = settings.COMPUTE_REST_API_URL.rstrip("/") + "/watchtower/validator-message"
        try:
            response = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
            if response.status_code == HTTPStatus.NOT_FOUND:
                return
            response.raise_for_status()
            data = response.json()
            self.offer(SignedTrustMessage(message=str(data["message"]), signature=str(data["signature"])))
        except Exception as exc:
            logger.warning("Validator trust: message not refreshed from %s: %s: %s", url, type(exc).__name__, exc)

    def refresh_owner(self) -> None:
        """Blocking: a new owner is taken only when every public chain node answers with it (at least two), so one
        node cannot name an owner by itself; any failure or disagreement keeps the last one."""
        if not self.enabled:
            return
        answers = set()
        for url in PUBLIC_CHAIN_NODES:
            try:
                answers.add(read_subnet_owner_hotkey(url, timeout=FETCH_TIMEOUT_SECONDS))
            except Exception as exc:
                logger.warning("Validator trust: subnet owner not read from %s: %s: %s", url, type(exc).__name__, exc)
                return
        if len(answers) != 1:
            logger.warning("Validator trust: public chain nodes disagree on the subnet owner: %s", sorted(answers))
            return
        (owner,) = answers
        if owner != self.owner:
            logger.warning("Validator trust: subnet owner changed from %s to %s", self.owner, owner)
            self.owner = owner
            self._save(OWNER_FILE, {"owner": owner})


validator_trust = ValidatorTrust(enabled=importlib.util.find_spec("core.config_override") is None)


async def run_validator_trust_refresh() -> None:
    """Background loop: the message every MESSAGE_REFRESH_SECONDS, the owner hourly at a random offset."""
    if not validator_trust.enabled:
        return
    owner_due = asyncio.get_running_loop().time() + random.uniform(0, OWNER_REFRESH_SECONDS)
    while True:
        if asyncio.get_running_loop().time() >= owner_due:
            await asyncio.to_thread(validator_trust.refresh_owner)
            owner_due += OWNER_REFRESH_SECONDS
        await asyncio.to_thread(validator_trust.refresh_message)
        await asyncio.sleep(MESSAGE_REFRESH_SECONDS)
