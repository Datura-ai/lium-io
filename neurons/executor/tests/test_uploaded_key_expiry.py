"""DAH-3394: an ssh key installed through /upload_ssh_key expires unless the validator removes it.

The validator uploads a fresh keypair per job and removes it afterwards; a key that is still in the
executor's authorized_keys after EXECUTOR_UPLOADED_KEY_TTL_S is a leak (I-78: keys planted through a
validator-hotkey-signed upload stayed for months). The executor now appends every upload with a marker
carrying the upload time and purges marked lines past the TTL. A key line without the marker was written
past that route and is dropped too; comments and blank lines the purge never touches (the one-time startup
stamp of lines the previous release wrote is test_legacy_uploaded_key_stamp.py).
"""

import asyncio
import base64
import hashlib
import logging
import time

import pytest

import services.ssh_service as ssh_service
from services.ssh_service import SSHService, purge_expired_uploaded_keys

TTL = 900
NOW = 1_800_000_000  # a fixed clock: the tests never read time.time() for the ages they assert on

RENTER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJm96GoibO7Tqp4/aMTDItExnAQDTXWiByXDQXVD5wRv renter@laptop\n"
TEMPLATE_KEY = 'command="/usr/bin/tunnel",no-pty ssh-rsa AAAAB3NzaC1yc2ETemplateKey template\n'
COMMENT = "# keys below this line were installed by the image\n"
ODDLY_SPACED = "ssh-ed25519   AAAAC3NzaC1lZDI1NTE5AAAAIE2SSg2OW1LVkJplLWN5VjIo6fIwhr3bXIVmcGciSUbh\t  \n"
# a CRLF line and a comment that is not UTF-8 (a key pasted from a Windows box, a latin-1 name)
FOREIGN_BYTES = b"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILZtEKC4QnQmpIcaRB/PGs/w/go4UMu0/t2Wekz1Y2/u user@pc\r\n# Jos\xe9's key\n"


MARKER = ssh_service.UPLOADED_KEY_MARKER.decode()


def _marked(key: str, uploaded_at) -> str:
    return f"{key} {MARKER}{uploaded_at}\n"


@pytest.fixture()
def authorized_keys(tmp_path, monkeypatch):
    """A private HOME so ~/.ssh/authorized_keys is a temp file; returns its path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # the line format these tests assert on is the unpinned one; the I-78 pin is test_ssh_source_pin.py
    monkeypatch.setattr(ssh_service, "VALIDATOR_SSH_SOURCES", "")
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    path = ssh_dir / "authorized_keys"
    path.write_text("")
    path.chmod(0o600)
    return path


def test_purge_keeps_comments_and_blank_lines_byte_for_byte(authorized_keys):
    # regression: the purge re-serialises the file (strip(), split/join) and a comment or blank line is
    # altered or lost with the keys it drops
    stale = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOKkPQzAplh/+woVCo4K3R4UvBK3rSqYCwtQDdFKEcsd", NOW - 2 * TTL)
    fresh = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFreshValidatorJobKey", NOW - 5)
    untouchable = (COMMENT + "\n").encode() + b"# Jos\xe9's key\n" + fresh.encode()
    authorized_keys.write_bytes(
        (RENTER_KEY + COMMENT + stale + "\n").encode() + FOREIGN_BYTES + (TEMPLATE_KEY + ODDLY_SPACED + fresh).encode()
    )
    before_mode = authorized_keys.stat().st_mode

    removed = purge_expired_uploaded_keys(TTL, now=NOW)

    assert removed == 5  # the stale upload and the four unmarked key lines
    assert authorized_keys.read_bytes() == untouchable
    assert authorized_keys.stat().st_mode == before_mode


def test_an_unmarked_key_is_dropped_and_logged_by_fingerprint_only(authorized_keys, caplog):
    # a key written into the file past /upload_ssh_key carries no marker and so never expired
    blob = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(range(32))
    body = base64.b64encode(blob).decode()
    authorized_keys.write_text(f"ssh-ed25519 {body} someone@host\n")
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")

    with caplog.at_level(logging.WARNING, logger=ssh_service.logger.name):
        removed = purge_expired_uploaded_keys(TTL, now=NOW)

    assert removed == 1
    assert authorized_keys.read_text() == ""
    assert f"removed 1 unmarked ssh key line(s): {fingerprint}" in caplog.text
    assert body not in caplog.text


# --- through the real writer -----------------------------------------------------------------------


def test_an_uploaded_key_expires_after_the_ttl_and_not_before(authorized_keys):
    # regression: the writer and the purge disagree on the marker (format, position, units) —
    # uploads are written but never recognised, so nothing ever expires
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFuEaUMojWO6jd3EorXLg/kyt6oiIyzyoE+1Xw5mUVV+ validator@lium"
    uploaded_at = time.time()
    SSHService().add_pubkey_to_host(key)
    assert authorized_keys.read_text().startswith(key + " ")

    assert purge_expired_uploaded_keys(TTL, now=uploaded_at + TTL - 5) == 0
    assert authorized_keys.read_text().startswith(key + " ")

    assert purge_expired_uploaded_keys(TTL, now=uploaded_at + TTL + 5) == 1
    assert authorized_keys.read_text() == ""


def test_an_upload_onto_a_file_without_a_trailing_newline_does_not_merge_with_the_last_line(authorized_keys):
    # regression: the upload is appended straight after a provider key that has no newline; the
    # merged line carries the marker and the purge deletes the provider's key with ours
    authorized_keys.write_text(RENTER_KEY.rstrip("\n"))
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBzeUy9ipww+CN1oJ51zPvAM04/oS8Ps5n3FmBHT3Zs5 validator@lium"

    SSHService().add_pubkey_to_host(key)
    assert authorized_keys.read_text().startswith(RENTER_KEY + key + " ")

    # two lines, not one merged marked line; the unmarked provider key goes with the expired upload
    assert purge_expired_uploaded_keys(TTL, now=time.time() + TTL + 5) == 2
    assert authorized_keys.read_text() == ""



def test_remove_ssh_key_still_removes_a_key_uploaded_with_the_marker(authorized_keys):
    # regression: /remove_ssh_key compares the whole line to the bare key, so a marked line is
    # never matched and every validator key stays until the TTL — 15 minutes of access it did
    # not ask for, on every job
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAGgl6IerZSFlcnlYw/g6d4kaTRkeDbu/Jqg2B0fxXOP validator@lium"
    authorized_keys.write_text(RENTER_KEY)
    service = SSHService()
    service.add_pubkey_to_host(key)

    service.remove_pubkey_from_host(key)

    assert authorized_keys.read_text() == RENTER_KEY


def test_remove_ssh_key_removes_a_key_the_previous_release_wrote_without_a_marker(authorized_keys):
    # regression: authorized_keys lives on the disk reserve (setup_disk_reserve.sh mount_ssh) and
    # survives the restart onto this release, so a key the previous release appended (no marker)
    # is still there when the validator asks to remove it
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDE19e79FjEfwVyfX0MeovSOipEmFuvguBBljNtq2Ag+ validator@lium"
    authorized_keys.write_text(RENTER_KEY + key + "\n")

    SSHService().remove_pubkey_from_host(key)

    assert authorized_keys.read_text() == RENTER_KEY


def test_a_public_key_with_an_embedded_newline_is_refused_by_the_writer(authorized_keys):
    # regression: "key-a\nkey-b" appended verbatim puts key-b on its own unmarked line — a
    # permanent key smuggled past the TTL by whoever holds the validator hotkey
    with pytest.raises(ValueError):
        SSHService().add_pubkey_to_host("ssh-ed25519 AAAAFirst a\nssh-ed25519 AAAASecond b")

    assert authorized_keys.read_text() == ""




def test_a_failed_legacy_stamp_skips_the_purge_and_the_keys_expire_only_a_full_ttl_after_the_retry(
    authorized_keys, monkeypatch
):
    legacy = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILegacyValidatorKeyNoMarker validator@old\n"
    authorized_keys.write_text(legacy)
    real_stamp = ssh_service.stamp_legacy_uploaded_keys
    calls = []

    def stamp_failing_once():
        calls.append(1)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")
        return real_stamp()

    # pass 1 at NOW fails the stamp, pass 2 at NOW+60 stamps, then the clock reaches the TTL edges
    clock = iter([NOW, NOW + 60, NOW + 60 + TTL - 5, NOW + 60 + TTL + 5])
    now = [next(clock)]
    seen = []

    async def fake_sleep(_):
        seen.append(authorized_keys.read_text())
        now[0] = next(clock, None)
        if now[0] is None:
            raise asyncio.CancelledError

    monkeypatch.setattr(ssh_service, "stamp_legacy_uploaded_keys", stamp_failing_once)
    monkeypatch.setattr(ssh_service.time, "time", lambda: now[0])
    monkeypatch.setattr(ssh_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_UPLOADED_KEY_TTL_S", TTL)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ssh_service.run_uploaded_key_purge())

    assert seen[0] == legacy  # the failed stamp skipped the purge
    assert seen[1].startswith(legacy.strip() + " " + MARKER) and seen[2] == seen[1]
    assert seen[3] == ""
