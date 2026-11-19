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
volume, which outlives a container recreate, so a restarted node keeps trusting the same validator with the backend
and the chain both unreachable. Each change is written durably (temp file, fsync, rename, fsync of the directory)
as soon as it is made. Until the newest state is on the volume, and after a start that could not read the kept state
(or found a message in it whose signature fails) until a served message confirms one, the executor trusts no
validator: it never falls back to one that may have been replaced. That unconfirmed mark is part of the written
state, so a restart before the confirmation stays closed too. The state a run writes is marked running, and the
shutdown writes the newest state marked ended: a start that finds the running mark (the last run ended without
writing its newest state: a failed or cut-short shutdown write, a crash) cannot tell whether a newer message was
taken, and stays closed the same way until a served message confirms a validator. Only a message the owner signed
confirms: a 404 or a failure never does, so the backend must keep serving a signed message (one naming the current
validator before any replacement) for a closed node to reopen. The operator's recovery for a node that cannot reach it
is to remove the kept state file, which returns the node to the image's message or baked validators. A kept state with any field
missing or malformed is unreadable. Only a volume that refused every write of a run, the running mark included,
cannot be read for it: a restart then reads the last state the volume holds.

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
# the shutdown waits this long for the last state write; well inside the pod's termination grace
FLUSH_SECONDS = 5.0
FLUSH_RETRY_PAUSE_SECONDS = 0.2


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
    """Atomic and durable: a crash mid-write leaves the previous file whole, and once this returns the new file
    survives a crash or power loss (file and directory entry both synced)."""
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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


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
        # the newest state is not on the volume yet: no validator is trusted until it is (written again on every
        # refresh while the volume refuses it)
        self._unsaved = False
        # the kept state existed but could not be read (or a message in it failed its signature) at a start, and no
        # served message from the confirmed owner has named a validator since: no validator is trusted. Kept in the
        # state file, so a restart before that confirmation stays closed even after the unreadable file was replaced
        self._unconfirmed = False
        # set by start_persisting: changes are written by its task, off the event loop, never by the caller
        self._wake: asyncio.Event | None = None
        # set at shutdown: the writer writes the newest state once more, then ends
        self._stopping = False
        # set at shutdown: the newest state is written marked ended, not running
        self._ended = False
        # the state the volume holds as far as this run knows: the writer writes whenever the state differs from it
        self._written: dict | None = None

    def load(self) -> None:
        """At start: the kept state (owner, held owner, messages) and the image copy, signed by the owner at release.
        A kept state that cannot be read, has a field missing or malformed, or holds a message whose signature fails,
        is dropped whole, and the node stays closed until a served message confirms a validator; so does one the last
        run left marked running."""
        if not self.enabled:
            return
        held_owner, state, kept = None, None, []
        try:
            state = json.loads((self.state_dir / STATE_FILE).read_text())
            Keypair(ss58_address=state["owner"])
            messages, held_owner, running, unconfirmed = (
                state["messages"], state["held"], state["running"], state["unconfirmed"]
            )
            if not (isinstance(messages, dict) and isinstance(running, bool) and isinstance(unconfirmed, bool)):
                raise ValueError("a field of the kept state is malformed")
            if held_owner is not None and held_owner not in messages:
                raise ValueError("the held owner has no kept message")
            for signer, signed in messages.items():
                signed = SignedTrustMessage(**signed)
                kept.append((verify_trust_message(signed, signer), signed, signer))
            self.owner = state["owner"]
            self._unconfirmed = unconfirmed
            if running:
                logger.warning(
                    "Validator trust: the last run ended before its newest state was written, trusting no validator "
                    "until a served message confirms one"
                )
                self._unconfirmed = True
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(
                "Validator trust: kept state unreadable, trusting no validator until a served message confirms one: "
                "%s: %s", type(exc).__name__, exc,
            )
            kept, state, held_owner, self._unconfirmed = [], None, None, True
        try:
            image = json.loads(self.image_message.read_text())
            signed = SignedTrustMessage(message=str(image["message"]), signature=str(image["signature"]))
            kept.append((verify_trust_message(signed, RELEASE_SUBNET_OWNER_HOTKEY), signed, RELEASE_SUBNET_OWNER_HOTKEY))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Validator trust: image message refused: %s: %s", type(exc).__name__, exc)
        for trusted, signed, signer in kept:
            self._remember(trusted, signed, signer)
        self._written = state
        self.held_owner = held_owner
        # chosen in memory before the first request; a choice the volume does not hold yet is written by the writer
        # once the refresh loop starts it, so a stalled volume never holds up the start
        self._select(save=False)
        # the unconfirmed mark is written too, so a restart before the confirmation still finds it
        # (the running mark alone does not close it: it is written once the writer starts)
        self._unsaved = (self.held is not None or self._unconfirmed) and (
            state is None or {**self._state(), "running": state["running"]} != state
        )
        logger.info("Validator trust at start: %s", self.report())

    def _remember(self, trusted: TrustedValidator, signed: SignedTrustMessage, signer: str) -> bool:
        """Keep `signed` as `signer`'s message when its revision is above the highest one known from that signer."""
        known = self.messages.get(signer)
        if known is not None and trusted.revision <= known[0].revision:
            return False
        self.messages[signer] = (trusted, signed)
        return True

    def _select(self, save: bool = True) -> None:
        """Hold the current owner's message; without one, the held owner's (what this node last trusted), else the
        image copy's. Run at start and after every owner change."""
        for signer in (self.owner, self.held_owner, RELEASE_SUBNET_OWNER_HOTKEY):
            if signer in self.messages:
                break
        else:
            return
        self._hold(signer, save)

    def _hold(self, signer: str, save: bool = True) -> None:
        trusted = self.messages[signer][0]
        if (trusted, signer) != (self.held, self.held_owner):
            self.held, self.held_owner = trusted, signer
            logger.warning("Validator trust: now trusting %s (revision %d)", trusted.validator, trusted.revision)
        if save:
            self._save()
        else:
            self._unsaved = True

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
            if self._unconfirmed and self.messages[self.owner][0] == trusted:
                # the confirmed owner's highest message, served again: it confirms what this node holds
                self._unconfirmed = False
                self._hold(self.owner)
            return False
        self._unconfirmed = False
        self._hold(self.owner)
        return True

    def _state(self) -> dict:
        """One document for the owner, the held owner, the messages and the unconfirmed mark: a restart never reads
        them apart."""
        return {
            "owner": self.owner,
            "held": self.held_owner,
            "messages": {signer: signed._asdict() for signer, (_, signed) in self.messages.items()},
            "unconfirmed": self._unconfirmed,
            "running": not self._ended,
        }

    def _save(self) -> None:
        """Keep the state on the volume: through the writer once start_persisting ran (the caller never waits on the
        volume), else written now (at start)."""
        self._unsaved = True
        if self._wake is not None:
            self._wake.set()
            return
        try:
            state = self._state()
            _write_json(self.state_dir / STATE_FILE, state)
            self._written, self._unsaved = state, False
        except OSError as exc:
            # no validator is trusted until it lands; written again on the next refresh
            logger.warning("Validator trust: state not on the volume, trusting no validator: %s", exc)

    def start_persisting(self) -> asyncio.Task:
        """From now on the state file has one writer, a task beside the refreshes. Each write takes the newest state
        and runs in a worker thread, one at a time: a change made during a write is written after it, so an older
        state never replaces a newer one, and a slow or hung volume holds up only the file, never the refreshes. Until
        the newest state is written, no validator is trusted (hotkeys); a failed write is written again at the next
        refresh."""
        self._wake, self._stopping = asyncio.Event(), False
        # the first write marks this run running, so a restart after a shutdown that could not write its newest
        # state knows it
        if self._state() != self._written:
            self._wake.set()
        return asyncio.create_task(self._persist())

    async def _persist(self) -> None:
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                # read before the write: a stop asked during it gets one more turn, which writes the newer state
                stopping = self._stopping
                state = self._state()
                if state != self._written:
                    try:
                        await asyncio.to_thread(_write_json, self.state_dir / STATE_FILE, state)
                    except OSError as exc:
                        logger.warning("Validator trust: state not on the volume, trusting no validator: %s", exc)
                    else:
                        self._written = state
                        # trusted only once on the volume: open when what was written is still the state, whatever
                        # was saved meanwhile; a change made during the write waits for its own
                        self._unsaved = self._state() != state
                if stopping:
                    if self._state() == self._written:
                        return
                    # no refresh comes after a shutdown: a failed last write is written again until it lands or
                    # stop_persisting's FLUSH_SECONDS deadline ends the writer
                    await asyncio.sleep(FLUSH_RETRY_PAUSE_SECONDS)
                    self._wake.set()
        finally:
            self._wake = None

    async def stop_persisting(self, writer: asyncio.Task) -> None:
        """At shutdown: one last turn of the writer, within FLUSH_SECONDS so a hung volume cannot hold up the shutdown.
        Nothing trusted depends on it: a validator is trusted only once its state is on the volume."""
        if self._wake is not None:
            self._stopping = self._ended = True
            self._wake.set()
        try:
            await asyncio.wait_for(writer, FLUSH_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Validator trust: state not written within %ss of shutdown", FLUSH_SECONDS)

    def closed(self) -> bool:
        """No validator is trusted: the newest state is not on the volume, or a kept state was unreadable at a start
        (or held a message failing its signature) and no served message has confirmed one since."""
        return self.enabled and (self._unsaved or self._unconfirmed)

    def hotkeys(self, baked: dict[str, str]) -> dict[str, str]:
        """Key id -> ss58 the executor accepts signatures from: none while closed, the signed validator alone, else
        `baked`."""
        if self.closed():
            return {}
        return {"signed": self.held.validator} if self.held is not None else baked

    def report(self) -> dict:
        return {
            "validator": self.held.validator if self.held else None,
            "revision": self.held.revision if self.held else None,
            "subnet_owner": self.owner if self.enabled else None,
            "closed": self.closed(),
        }

    async def refresh_message(self) -> None:
        """A 404 or any failure keeps the message held, and confirms nothing."""
        if not self.enabled:
            return
        if self._unsaved:
            self._save()
        elif self._wake is not None and self._state() != self._written:
            # only the running mark is unwritten: written again, trusting what the volume holds meanwhile
            self._wake.set()
        if not self.owner_confirmed or not settings.COMPUTE_REST_API_URL:
            return
        url = settings.COMPUTE_REST_API_URL.rstrip("/") + "/watchtower/validator-message"
        try:
            data = await fetch_json(url)
            if data is None:
                # a 404 is no statement from the owner: it never confirms a closed node
                return
            signed = SignedTrustMessage(message=str(data["message"]), signature=str(data["signature"]))
            self.offer(signed)
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
            self._save()


validator_trust = ValidatorTrust(enabled=importlib.util.find_spec("core.config_override") is None)


async def run_validator_trust_refresh() -> None:
    """Background loop: the message every MESSAGE_REFRESH_SECONDS; the owner every round until a read succeeds,
    then about hourly with random jitter."""
    if not validator_trust.enabled:
        return
    # the state file is written beside the loop, so a slow volume never holds up a refresh
    writer = validator_trust.start_persisting()
    owner_due = asyncio.get_running_loop().time()
    try:
        while True:
            if asyncio.get_running_loop().time() >= owner_due:
                await validator_trust.refresh_owner()
                if validator_trust.owner_confirmed:
                    owner_due = asyncio.get_running_loop().time() + random.uniform(0.5, 1.5) * OWNER_REFRESH_SECONDS
            await validator_trust.refresh_message()
            await asyncio.sleep(MESSAGE_REFRESH_SECONDS)
    finally:
        await validator_trust.stop_persisting(writer)
