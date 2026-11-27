import asyncio
import base64
import binascii
import getpass
import hashlib
import ipaddress
import logging
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from http import HTTPStatus

from cryptography.hazmat.primitives.serialization import load_ssh_public_key
from pydantic import BaseModel

from core.config import VALIDATOR_SSH_SOURCES, settings
from dependencies.auth import match_validator_hotkey
from services.dial import trusted_get

logger = logging.getLogger(__name__)

# DAH-3394: every key /upload_ssh_key appends carries this trailing comment token with the upload
# time; the purge below expires marked lines and drops unmarked key lines. The previous release's
# uploads lack it and are marked once by the startup stamp (stamp_legacy_uploaded_keys). sshd treats
# everything after the key material as a comment, so the token changes nothing for logins.
UPLOADED_KEY_MARKER = b"lium-uploaded-at="

# The routes and the purge task all run on the event loop thread and do their file work
# synchronously, so today nothing contends for this lock; it is what keeps an append from landing
# on an inode the purge is replacing if any of them ever moves to a worker thread.
_authorized_keys_lock = threading.Lock()


def authorized_keys_path() -> str:
    return os.path.expanduser("~/.ssh/authorized_keys")


def _lacks_trailing_newline(path: str) -> bool:
    """True when the file exists, is not empty and its last byte is not a newline."""
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:
                return False
            file.seek(-1, os.SEEK_END)
            return file.read(1) != b"\n"
    except FileNotFoundError:
        return False


def _split_marker(line: bytes) -> tuple[bytes, bytes | None]:
    """(key without the marker, the marker's value) — value is None on an unmarked line.

    The marker is the line's last whitespace-separated token; plain splits, no regex, so a
    hostile line (the key text is peer-supplied) costs linear time on every purge tick. Bytes
    throughout: one non-UTF-8 byte anywhere in the file must not stop the purge or a removal.
    """
    stripped = line.strip()
    parts = stripped.rsplit(None, 1)
    if len(parts) != 2 or not parts[1].startswith(UPLOADED_KEY_MARKER):
        return stripped, None
    return parts[0], parts[1][len(UPLOADED_KEY_MARKER):]


def _in_cvm() -> bool:
    """A dstack CVM: QEMU user networking shows sshd its own gateway address, never the validator's.

    A socket, not just a path: docker-compose.app.yml bind-mounts /var/run/dstack.sock on every host,
    and on a host without dstack docker creates an empty directory there.
    """
    try:
        return stat.S_ISSOCK(os.stat(settings.DSTACK_SOCKET_PATH).st_mode)
    except OSError:
        return False


# The pinned addresses come from `<COMPUTE_REST_API_URL>/watchtower/ssh-sources`, signed by the validator as
# `ssh-sources:{sources}:{timestamp}` (the prefix keeps it apart from a signed digest), so a NAT change or turning
# the pin off fleet-wide needs no executor release.
SSH_SOURCES_CACHE_SECONDS = 300
SSH_SOURCES_TIMEOUT_SECONDS = 5.0
SSH_SOURCES_MAX_SKEW_SECONDS = 10 * 60



AUTHORIZED_KEY_TYPES = frozenset(
    {"ssh-ed25519", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521", "ssh-rsa"}
)


def check_authorized_key(pub_key: str) -> None:
    """Raise ValueError unless `pub_key` is `<type> <base64> [comment]` holding a real public key.

    sshd reads anything before the key type as options (command=, environment=, ...), so a line
    must start with an allowed type and the key must parse.
    """
    parts = pub_key.split()
    if len(parts) < 2 or parts[0] not in AUTHORIZED_KEY_TYPES:
        raise ValueError("an authorized_keys entry is a public key of an allowed type, with no options")
    try:
        load_ssh_public_key(f"{parts[0]} {parts[1]}".encode())
    except Exception as exc:
        raise ValueError(f"not a valid {parts[0]} public key") from exc

class SshSourcesAnswer(BaseModel):
    """The `/watchtower/ssh-sources` answer (lium-platform SignedSshSourcesResponse)."""

    sources: str
    timestamp: int
    signature: str


def verify_signed_ssh_sources(answer: SshSourcesAnswer) -> str:
    """The `ip,ip` list of a `/watchtower/ssh-sources` answer after checking its signature; "" = no pin."""
    now = int(time.time())
    if abs(answer.timestamp - now) > SSH_SOURCES_MAX_SKEW_SECONDS:
        raise ValueError(f"ssh-sources answer timestamp out of range (now={now}, ts={answer.timestamp})")
    if match_validator_hotkey(f"ssh-sources:{answer.sources}:{answer.timestamp}", answer.signature) is None:
        raise ValueError("ssh-sources answer is not signed by a configured validator hotkey")
    # the backend accepts `a, b`; sshd's from= takes no spaces, so pin the stripped entries
    entries = [entry.strip() for entry in answer.sources.split(",")] if answer.sources else []
    # a malformed from= makes sshd refuse the key: keep the last good list instead
    for entry in entries:
        ipaddress.ip_network(entry)  # an empty entry (`ip,`) is refused too
    return ",".join(entries)


class SignedSshSources:
    """The signed source list. A background task refreshes it (run_ssh_sources_refresh) so an upload never waits
    on the backend; a failed refresh keeps the last good list; a 404 and a node that never got one do not pin."""

    def __init__(self):
        self._sources = ""

    def get(self) -> str:
        return self._sources

    def refresh(self) -> None:
        """Blocking: run it off the event loop."""
        if not settings.COMPUTE_REST_API_URL:
            return
        url = settings.COMPUTE_REST_API_URL.rstrip("/") + "/watchtower/ssh-sources"
        try:
            response = trusted_get(url, timeout=SSH_SOURCES_TIMEOUT_SECONDS)
            if response.status_code == HTTPStatus.NOT_FOUND:
                # the backend has no list (endpoint reverted): unpin, as a node that never got one
                self._sources = ""
                return
            response.raise_for_status()
            self._sources = verify_signed_ssh_sources(SshSourcesAnswer.model_validate(response.json()))
        except Exception as exc:  # never fail an upload over this
            logger.warning("Validator ssh sources not refreshed from %s: %s: %s", url, type(exc).__name__, exc)


_signed_ssh_sources = SignedSshSources()


def _pin_applies() -> bool:
    return bool(VALIDATOR_SSH_SOURCES) and settings.EXECUTOR_SSH_SOURCE_PIN and not _in_cvm()


def _validator_source_pin_option() -> bytes:
    """The `from="..." ` option every uploaded key gets (I-78), or b"" when there is no pin.

    A build without a source list (non-prod, e2e) never pins; a prod build pins to the signed list.
    """
    if not _pin_applies():
        return b""
    sources = _signed_ssh_sources.get()
    return b'from="' + sources.encode() + b'" ' if sources else b""


def _key_without_marker_and_source_pin(line: bytes) -> bytes:
    """The key of a line without our marker and without a `from="..."` option a release put in front."""
    key = _split_marker(line)[0]
    if key.startswith(b'from="'):
        end = key.find(b'" ', len(b'from="'))
        if end != -1:
            key = key[end + 2:]
    return key


def _replace_authorized_keys(path: str, lines: list[bytes]) -> None:
    """Write `lines` next to the file and move it into place, keeping the file's mode.

    A crash mid-write leaves the old file whole rather than an empty authorized_keys that locks
    the validator out.
    """
    mode = os.stat(path).st_mode & 0o777
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".authorized_keys.")
    try:
        with os.fdopen(fd, "wb") as file:
            file.writelines(lines)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _rewrite_authorized_keys(path: str, keep: Callable[[bytes], bool]) -> int:
    """Rewrite `path` keeping the lines `keep(line)` accepts byte-for-byte; returns lines dropped."""
    with open(path, "rb") as file:
        lines = file.readlines()
    kept = [line for line in lines if keep(line)]
    dropped = len(lines) - len(kept)
    if dropped == 0:
        return 0
    _replace_authorized_keys(path, kept)
    return dropped


# Written next to authorized_keys (so on the disk reserve, where authorized_keys survives a
# container rebuild) once the startup stamp has run; its presence is what makes the stamp one-time.
LEGACY_STAMP_DONE_FILE = ".lium-uploaded-keys-stamped"


def legacy_stamp_done_path() -> str:
    return os.path.join(os.path.dirname(authorized_keys_path()), LEGACY_STAMP_DONE_FILE)


def stamp_legacy_uploaded_keys(now: float | None = None) -> int | None:
    """Once per executor: give every unmarked key line `lium-uploaded-at=<now>`; returns the count.

    Returns None when an earlier start already did it (LEGACY_STAMP_DONE_FILE exists). The keys the
    previous release appended carry no marker, and authorized_keys lives on the disk reserve, so
    they survive the upgrade and the purge never reaches them (I-78: on a long-running node that
    is hundreds of lines, every one of them the shape this executor uploads). Stamped with the
    start time they are on the same clock as a fresh upload: gone after EXECUTOR_UPLOADED_KEY_TTL_S
    unless the validator uploads them again. Lines already carrying the marker, blank lines and
    `#` comments are left as they are; a stamped line is rewritten as the stripped key, one space
    and the marker. The done-file is written after the rewrite, so a start that
    dies between the two stamps again next time; the second stamp finds nothing unmarked.
    """
    now = time.time() if now is None else now
    stamp = b" " + UPLOADED_KEY_MARKER + str(int(now)).encode() + b"\n"
    path = authorized_keys_path()
    done_path = legacy_stamp_done_path()
    with _authorized_keys_lock:
        if os.path.exists(done_path):
            return None
        try:
            with open(path, "rb") as file:
                lines = file.readlines()
        except FileNotFoundError:
            lines = []
        out: list[bytes] = []
        stamped = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(b"#") or _split_marker(line)[1] is not None:
                out.append(line)
            else:
                out.append(stripped + stamp)
                stamped += 1
        if stamped:
            _replace_authorized_keys(path, out)
        with open(done_path, "w", encoding="utf-8") as file:
            file.write(f"{int(now)} {stamped}\n")
    return stamped


def _key_fingerprint(line: bytes) -> str:
    """`SHA256:...` as ssh-keygen -l prints it, from the first token that decodes to an ssh key blob."""
    for token in line.split():
        try:
            blob = base64.b64decode(token, validate=True)
        except (binascii.Error, ValueError):
            continue
        type_len = int.from_bytes(blob[:4], "big")
        if blob[4 : 4 + type_len].startswith((b"ssh-", b"ecdsa-", b"sk-")):
            return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return "unparsed"


def purge_expired_uploaded_keys(ttl_s: int, now: float | None = None) -> int:
    """Drop every marked key uploaded more than `ttl_s` seconds ago and every unmarked key line; returns how many.

    Every key we install goes through /upload_ssh_key and carries UPLOADED_KEY_MARKER, so an unmarked key line
    was written past that route and is dropped (blank lines and `#` comments stay). A marked line whose
    timestamp cannot be read, or lies more than `ttl_s` in the future (the host clock was stepped back after
    the upload), is dropped too: the marker is ours, so an age we cannot bound is a key we cannot bound.
    Every marked key therefore lives at most 2 × ttl_s.
    """
    now = time.time() if now is None else now
    path = authorized_keys_path()
    unmarked: list[bytes] = []

    def keep(line: bytes) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith(b"#"):
            return True
        _, uploaded_at = _split_marker(line)
        if uploaded_at is None:
            unmarked.append(stripped)
            return False
        try:
            return abs(now - int(uploaded_at)) <= ttl_s
        except ValueError:
            return False

    with _authorized_keys_lock:
        try:
            removed = _rewrite_authorized_keys(path, keep)
        except FileNotFoundError:
            return 0
    if unmarked:
        # fingerprints only: the key text is never logged
        logger.warning(
            "removed %d unmarked ssh key line(s): %s",
            len(unmarked),
            ", ".join(_key_fingerprint(line) for line in unmarked),
        )
    return removed


async def run_ssh_sources_refresh() -> None:
    """Background loop: refresh the signed source list every SSH_SOURCES_CACHE_SECONDS, off the event loop.
    Not run where the pin never applies (no source list in the build, the switch off, a CVM)."""
    if not _pin_applies():
        return
    while True:
        await asyncio.to_thread(_signed_ssh_sources.refresh)
        await asyncio.sleep(SSH_SOURCES_CACHE_SECONDS)


async def run_uploaded_key_purge() -> None:
    """Background loop: every EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S remove the expired uploads."""
    ttl_s = settings.EXECUTOR_UPLOADED_KEY_TTL_S
    interval_s = settings.EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S
    logger.info("uploaded ssh keys expire after %ss (checked every %ss)", ttl_s, interval_s)
    while True:
        try:
            # every pass until it has run once: a failed stamp skips the purge, which would drop the unstamped keys
            stamped = stamp_legacy_uploaded_keys()
            if stamped is not None:
                # the count is the record that the upgrade closed the legacy access on this node
                (logger.warning if stamped else logger.info)("legacy keys stamped: %d", stamped)
            removed = purge_expired_uploaded_keys(ttl_s)
            if removed:
                # the count only: a key still present after its TTL is what this line reports
                logger.warning("removed %d ssh key line(s): expired after %ss or unmarked", removed, ttl_s)
        except Exception:
            logger.exception("uploaded ssh key stamp or purge failed; retrying next interval")
        await asyncio.sleep(interval_s)


class SSHService:
    def add_pubkey_to_host(self, pub_key: str):
        pub_key = pub_key.strip()
        if "\n" in pub_key or "\r" in pub_key:
            # a second line would be a second, unmarked key that no TTL reaches
            raise ValueError("an authorized_keys entry is one line")
        check_authorized_key(pub_key)
        line = _validator_source_pin_option() + pub_key.encode() + b" " + UPLOADED_KEY_MARKER + str(int(time.time())).encode() + b"\n"
        with _authorized_keys_lock:
            path = authorized_keys_path()
            # a last line without its newline would merge with this one, and the merged line —
            # marked — would take the other key with it at the purge
            if _lacks_trailing_newline(path):
                line = b"\n" + line
            with open(path, "ab") as file:
                file.write(line)

    def remove_pubkey_from_host(self, pub_key: str):
        wanted = pub_key.strip().encode()
        # a key uploaded by this release carries the marker; one uploaded by the previous
        # release does not — both must go when the validator asks
        with _authorized_keys_lock:
            _rewrite_authorized_keys(
                authorized_keys_path(), lambda line: _key_without_marker_and_source_pin(line) != wanted
            )

    def get_current_os_user(self) -> str:
        return getpass.getuser()

    def get_host_public_key(self) -> str | None:
        host_key_path = settings.SSH_HOST_KEY_PATH
        if not host_key_path:
            return None

        path = os.path.expanduser(host_key_path)
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    candidate = line.strip()
                    if candidate:
                        return candidate
        except FileNotFoundError:
            logger.warning("SSH host key file not found at %s", path)
        except OSError as exc:
            logger.warning("Failed to read SSH host key from %s: %s", path, exc)

        return None
