"""DAH-3394: keys the previous release uploaded get the expiry marker once, at the first start.

authorized_keys lives on the disk reserve and survives the upgrade, so every key the validator
uploaded before this release is there with no marker and the purge never touches it (the staging
executor held 640 such lines, all bare two-field keys — the shape this executor uploads). At its
first start on this release the executor stamps every unmarked line with the start time; the TTL
purge then removes them on its normal path. A done-file next to authorized_keys makes it one-time.
"""

import asyncio
import logging

import pytest

import services.ssh_service as ssh_service
from services.ssh_service import (
    legacy_stamp_done_path,
    purge_expired_uploaded_keys,
    stamp_legacy_uploaded_keys,
)

TTL = 900
NOW = 1_800_000_000  # a fixed clock: the tests never read time.time() for the ages they assert on
MARKER = ssh_service.UPLOADED_KEY_MARKER.decode()

# what the validator's generate_ssh_key produced and the old add_pubkey_to_host appended: two fields
LEGACY_UPLOAD = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKeyTheOldReleaseAppended\n"
LEGACY_UPLOAD_2 = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAnotherOldUpload\n"
WITH_COMMENT = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKeyWithAComment someone@somewhere\n"
COMMENT = "# keys below this line were installed by the image\n"
FOREIGN_BYTES = b"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIWindowsPastedKey user@pc\r\n# Jos\xe9's key\n"


def _marked(key: str, uploaded_at) -> str:
    return f"{key.rstrip()} {MARKER}{uploaded_at}\n"


@pytest.fixture()
def authorized_keys(tmp_path, monkeypatch):
    """A private HOME so ~/.ssh/authorized_keys is a temp file; returns its path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    path = ssh_dir / "authorized_keys"
    path.write_text("")
    path.chmod(0o600)
    return path


def test_first_start_stamps_every_unmarked_key_line(authorized_keys):
    # regression: the stamp is written in a form the purge does not read (position, token), or
    # skips lines by shape (a comment after the key, a CRLF, a non-UTF-8 byte), leaving some of
    # the legacy access open
    authorized_keys.write_bytes((LEGACY_UPLOAD + COMMENT + WITH_COMMENT + "\n").encode() + FOREIGN_BYTES + LEGACY_UPLOAD_2.encode())
    before_mode = authorized_keys.stat().st_mode

    assert stamp_legacy_uploaded_keys(now=NOW) == 4

    expected = (
        _marked(LEGACY_UPLOAD, NOW) + COMMENT + _marked(WITH_COMMENT, NOW) + "\n"
    ).encode() + _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIWindowsPastedKey user@pc", NOW).encode() + b"# Jos\xe9's key\n" + _marked(LEGACY_UPLOAD_2, NOW).encode()
    assert authorized_keys.read_bytes() == expected
    assert authorized_keys.stat().st_mode == before_mode


def test_lines_already_carrying_the_marker_keep_their_own_time(authorized_keys):
    # regression: the stamp re-marks a fresh upload with the start time, extending or cutting the
    # window the validator's running job was given; or it rewrites a file with nothing to stamp
    fresh = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFreshUpload", NOW - 100)
    older = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOlderUpload", NOW - 800)
    authorized_keys.write_text(fresh + older)
    inode = authorized_keys.stat().st_ino

    assert stamp_legacy_uploaded_keys(now=NOW) == 0

    assert authorized_keys.read_text() == fresh + older
    assert authorized_keys.stat().st_ino == inode


def test_a_second_start_stamps_nothing_even_when_a_new_unmarked_line_appeared(authorized_keys):
    # regression: "once" implemented as "while unmarked lines exist" puts a line added after the
    # upgrade on a 15-minute clock at the next restart; or the done-file is not on the reserve
    # path and every restart re-stamps
    authorized_keys.write_text(LEGACY_UPLOAD)
    assert stamp_legacy_uploaded_keys(now=NOW) == 1
    after_first = authorized_keys.read_bytes()
    authorized_keys.write_bytes(after_first + LEGACY_UPLOAD_2.encode())

    assert stamp_legacy_uploaded_keys(now=NOW + 600) is None

    assert authorized_keys.read_bytes() == after_first + LEGACY_UPLOAD_2.encode()
    assert legacy_stamp_done_path() == str(authorized_keys.parent / ssh_service.LEGACY_STAMP_DONE_FILE)


def test_the_purge_removes_a_stamped_line_after_the_ttl_and_not_before(authorized_keys):
    # regression: the stamp's time is written in another unit or the purge treats a stamped line
    # as unbounded — the legacy keys go at once under a running job, or never
    authorized_keys.write_text(LEGACY_UPLOAD + LEGACY_UPLOAD_2)
    assert stamp_legacy_uploaded_keys(now=NOW) == 2

    assert purge_expired_uploaded_keys(TTL, now=NOW + TTL) == 0
    assert authorized_keys.read_text() == _marked(LEGACY_UPLOAD, NOW) + _marked(LEGACY_UPLOAD_2, NOW)

    assert purge_expired_uploaded_keys(TTL, now=NOW + TTL + 1) == 2
    assert authorized_keys.read_text() == ""


def test_the_done_file_is_written_only_after_the_rewrite_succeeded(authorized_keys, monkeypatch):
    # regression: the done-file is written first (or the two are swapped), so a start that dies
    # mid-rewrite records the stamp as done and the legacy keys are never stamped
    real_rewrite = ssh_service._replace_authorized_keys
    failures = iter([OSError("disk full")])

    def rewrite_fails_once(path, lines):
        error = next(failures, None)
        if error is not None:
            raise error
        real_rewrite(path, lines)

    monkeypatch.setattr(ssh_service, "_replace_authorized_keys", rewrite_fails_once)
    authorized_keys.write_text(LEGACY_UPLOAD)

    with pytest.raises(OSError):
        stamp_legacy_uploaded_keys(now=NOW)

    assert authorized_keys.read_text() == LEGACY_UPLOAD
    assert not (authorized_keys.parent / ssh_service.LEGACY_STAMP_DONE_FILE).exists()
    assert stamp_legacy_uploaded_keys(now=NOW) == 1
    assert authorized_keys.read_text() == _marked(LEGACY_UPLOAD, NOW)


def test_a_start_without_an_authorized_keys_file_records_the_stamp_as_done(authorized_keys):
    # regression: a fresh node (no upload yet) raises on the missing file and stamps at every
    # start until a key appears, then puts the validator's first upload of the day on the clock twice
    authorized_keys.unlink()

    assert stamp_legacy_uploaded_keys(now=NOW) == 0
    assert stamp_legacy_uploaded_keys(now=NOW + 1) is None
    assert not authorized_keys.exists()


def test_the_purge_loop_stamps_at_start_and_logs_the_count(authorized_keys, monkeypatch, caplog):
    # regression: the stamp is never wired into the lifespan task, or runs without the count line,
    # so nothing tells an operator whether the upgrade closed the legacy access on this node
    authorized_keys.write_text(LEGACY_UPLOAD + LEGACY_UPLOAD_2)

    def purge(ttl_s):
        raise asyncio.CancelledError  # the lifespan's cancel, on the first tick

    monkeypatch.setattr(ssh_service, "purge_expired_uploaded_keys", purge)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S", 0)

    with caplog.at_level(logging.WARNING, logger="services.ssh_service"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ssh_service.run_uploaded_key_purge())

    assert "legacy keys stamped: 2" in caplog.text
    assert authorized_keys.read_text().count(MARKER) == 2


def test_a_failing_stamp_does_not_stop_the_purge_loop(monkeypatch, caplog):
    # regression: an OSError at start (read-only /root/.ssh for a moment) ends the lifespan task
    # before its first tick, and no upload expires until the next restart
    ticks = []

    def stamp(now=None):
        raise OSError("authorized_keys busy")

    def purge(ttl_s):
        ticks.append(ttl_s)
        raise asyncio.CancelledError

    monkeypatch.setattr(ssh_service, "stamp_legacy_uploaded_keys", stamp)
    monkeypatch.setattr(ssh_service, "purge_expired_uploaded_keys", purge)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_UPLOADED_KEY_TTL_S", 7)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S", 0)

    with caplog.at_level(logging.WARNING, logger="services.ssh_service"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ssh_service.run_uploaded_key_purge())

    assert ticks == [7]
    assert "legacy ssh key stamp failed" in caplog.text
