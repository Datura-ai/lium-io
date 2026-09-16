import asyncio
import getpass
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable

from core.config import settings

logger = logging.getLogger(__name__)

# DAH-3394: every key /upload_ssh_key appends carries this trailing comment token with the upload
# time; the purge below expires marked lines only. Lines without it are the previous release's
# uploads, which the startup stamp (stamp_legacy_uploaded_keys) marks once. sshd treats
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


def purge_expired_uploaded_keys(ttl_s: int, now: float | None = None) -> int:
    """Drop every marked key uploaded more than `ttl_s` seconds ago; returns how many.

    Only lines carrying UPLOADED_KEY_MARKER are candidates; every other line is kept unchanged.
    A marked line whose timestamp cannot be read, or lies more than `ttl_s` in the future (the
    host clock was stepped back after the upload), is dropped too: the marker is ours, so an age
    we cannot bound is a key we cannot bound. Every marked key therefore lives at most 2 × ttl_s.
    """
    now = time.time() if now is None else now
    path = authorized_keys_path()

    def keep(line: bytes) -> bool:
        _, uploaded_at = _split_marker(line)
        if uploaded_at is None:
            return True
        try:
            return abs(now - int(uploaded_at)) <= ttl_s
        except ValueError:
            return False

    with _authorized_keys_lock:
        try:
            return _rewrite_authorized_keys(path, keep)
        except FileNotFoundError:
            return 0


async def run_uploaded_key_purge() -> None:
    """Background loop: every EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S remove the expired uploads."""
    ttl_s = settings.EXECUTOR_UPLOADED_KEY_TTL_S
    interval_s = settings.EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S
    logger.info("uploaded ssh keys expire after %ss (checked every %ss)", ttl_s, interval_s)
    try:
        stamped = stamp_legacy_uploaded_keys()
        if stamped is not None:
            # the count is the record that the upgrade closed the legacy access on this node
            (logger.warning if stamped else logger.info)("legacy keys stamped: %d", stamped)
    except Exception:
        logger.exception("legacy ssh key stamp failed; the purge still runs")
    while True:
        try:
            removed = purge_expired_uploaded_keys(ttl_s)
            if removed:
                # the count only: a key still present after its TTL is what this line reports
                logger.warning("removed %d uploaded ssh key(s) older than %ss", removed, ttl_s)
        except Exception:
            logger.exception("uploaded ssh key purge failed; retrying next interval")
        await asyncio.sleep(interval_s)


class SSHService:
    def add_pubkey_to_host(self, pub_key: str):
        pub_key = pub_key.strip()
        if "\n" in pub_key or "\r" in pub_key:
            # a second line would be a second, unmarked key that no TTL reaches
            raise ValueError("an authorized_keys entry is one line")
        line = pub_key.encode() + b" " + UPLOADED_KEY_MARKER + str(int(time.time())).encode() + b"\n"
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
                authorized_keys_path(), lambda line: _split_marker(line)[0] != wanted
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
