"""DAH-3394: an ssh key installed through /upload_ssh_key expires unless the validator removes it.

The validator uploads a fresh keypair per job and removes it afterwards; a key that is still in the
executor's authorized_keys after EXECUTOR_UPLOADED_KEY_TTL_S is a leak (I-78: keys planted through a
validator-hotkey-signed upload stayed for months). The executor now appends every upload with a marker
carrying the upload time and purges marked lines past the TTL. Everything else in the file — a template's
key, a key the provider added by hand, comments, blank lines — is never touched.
"""

import asyncio
import logging
import time

import pytest

import services.ssh_service as ssh_service
from services.ssh_service import SSHService, purge_expired_uploaded_keys

TTL = 900
NOW = 1_800_000_000  # a fixed clock: the tests never read time.time() for the ages they assert on

RENTER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIRenterOwnKeyNotOursToTouch renter@laptop\n"
TEMPLATE_KEY = 'command="/usr/bin/tunnel",no-pty ssh-rsa AAAAB3NzaC1yc2ETemplateKey template\n'
COMMENT = "# keys below this line were installed by the image\n"
ODDLY_SPACED = "ssh-ed25519   AAAAC3NzaC1lZDI1NTE5AAAAIKeyWithTabsAndSpaces\t  \n"
# a CRLF line and a comment that is not UTF-8 (a key pasted from a Windows box, a latin-1 name)
FOREIGN_BYTES = b"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIWindowsPastedKey user@pc\r\n# Jos\xe9's key\n"


MARKER = ssh_service.UPLOADED_KEY_MARKER.decode()


def _marked(key: str, uploaded_at) -> str:
    return f"{key} {MARKER}{uploaded_at}\n"


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


def test_purge_removes_a_marked_key_past_the_ttl_and_keeps_a_fresh_one(authorized_keys):
    # regression: off-by-one or unit error on the age (ms vs s, TTL compared to the timestamp
    # instead of the age) removes a key the validator is still using, or never removes any
    stale = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIStaleValidatorJobKey", NOW - TTL - 1)
    fresh = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFreshValidatorJobKey", NOW - TTL)
    authorized_keys.write_text(stale + fresh)

    removed = purge_expired_uploaded_keys(TTL, now=NOW)

    assert removed == 1
    assert authorized_keys.read_text() == fresh


def test_purge_keeps_every_unmarked_line_byte_for_byte(authorized_keys):
    # regression: the purge re-serialises the file (strip(), split/join, dropped comments or
    # blank lines) and a renter's or template's key is altered or lost with the stale one
    stale = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIStaleValidatorJobKey", NOW - 2 * TTL)
    untouchable = (RENTER_KEY + COMMENT + "\n").encode() + FOREIGN_BYTES + (TEMPLATE_KEY + ODDLY_SPACED).encode()
    authorized_keys.write_bytes(
        (RENTER_KEY + COMMENT + stale + "\n").encode() + FOREIGN_BYTES + (TEMPLATE_KEY + ODDLY_SPACED).encode()
    )
    before_mode = authorized_keys.stat().st_mode

    removed = purge_expired_uploaded_keys(TTL, now=NOW)

    assert removed == 1
    assert authorized_keys.read_bytes() == untouchable
    assert authorized_keys.stat().st_mode == before_mode


def test_purge_drops_a_marked_key_whose_timestamp_it_cannot_read(authorized_keys):
    # regression: a marker the purge cannot parse is skipped as "not ours", and a key with a
    # garbled timestamp becomes the one permanent upload
    garbled = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGarbledMarkerKey", "yesterday")
    authorized_keys.write_text(RENTER_KEY + garbled)

    assert purge_expired_uploaded_keys(TTL, now=NOW) == 1
    assert authorized_keys.read_text() == RENTER_KEY


def test_purge_drops_a_marked_key_dated_more_than_ttl_in_the_future(authorized_keys):
    # regression: a negative age reads as "fresh", so a key uploaded while the host clock ran ahead
    # (then stepped back) lives until that future date — unbounded, like an unparsable marker
    far_future = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIClockSteppedBackKey", NOW + TTL + 1)
    slightly_ahead = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINtpSlewKey", NOW + 5)
    authorized_keys.write_text(far_future + slightly_ahead)

    assert purge_expired_uploaded_keys(TTL, now=NOW) == 1
    assert authorized_keys.read_text() == slightly_ahead


def test_purge_leaves_the_file_alone_when_nothing_expired(authorized_keys):
    # regression: a rewrite every interval on an unchanged file (mtime churn, a truncated file
    # if the process dies mid-write) where no rewrite was needed
    fresh = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFreshValidatorJobKey", NOW - 1)
    authorized_keys.write_text(RENTER_KEY + fresh)
    inode = authorized_keys.stat().st_ino

    assert purge_expired_uploaded_keys(TTL, now=NOW) == 0
    assert authorized_keys.stat().st_ino == inode
    assert authorized_keys.read_text() == RENTER_KEY + fresh


def test_purge_without_an_authorized_keys_file_is_a_no_op(authorized_keys):
    # regression: the loop dies on its first tick on an executor that has had no upload yet
    authorized_keys.unlink()

    assert purge_expired_uploaded_keys(TTL, now=NOW) == 0


def test_purge_and_removal_survive_a_non_utf8_byte_elsewhere_in_the_file(authorized_keys):
    # regression: text-mode reads raise UnicodeDecodeError on one latin-1 byte in someone else's
    # comment, so on that executor no upload ever expires and /remove_ssh_key is a 500
    stale = _marked("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIStaleValidatorJobKey", NOW - 2 * TTL)
    fresh_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFreshValidatorJobKey validator@lium"
    authorized_keys.write_bytes(FOREIGN_BYTES + (stale + _marked(fresh_key, NOW)).encode())

    assert purge_expired_uploaded_keys(TTL, now=NOW) == 1
    SSHService().remove_pubkey_from_host(fresh_key)

    assert authorized_keys.read_bytes() == FOREIGN_BYTES


def test_the_purge_loop_survives_one_failing_tick_and_logs_what_it_removed(monkeypatch, caplog):
    # regression: a narrowed `except`, or none, ends the loop on its first transient error (a
    # read-only filesystem moment, a permissions blip) and nothing expires until the next restart
    ticks = []

    def purge(ttl_s):
        ticks.append(ttl_s)
        if len(ticks) == 1:
            raise OSError("authorized_keys busy")
        if len(ticks) == 2:
            return 1
        raise asyncio.CancelledError  # the lifespan's cancel, on the third tick

    monkeypatch.setattr(ssh_service, "purge_expired_uploaded_keys", purge)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_UPLOADED_KEY_TTL_S", 7)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S", 0)

    with caplog.at_level(logging.WARNING, logger="services.ssh_service"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(ssh_service.run_uploaded_key_purge())

    assert ticks == [7, 7, 7]
    assert "uploaded ssh key purge failed" in caplog.text
    assert "removed 1 uploaded ssh key(s) older than 7s" in caplog.text


# --- through the real writer -----------------------------------------------------------------------


def test_an_uploaded_key_expires_after_the_ttl_and_not_before(authorized_keys):
    # regression: the writer and the purge disagree on the marker (format, position, units) —
    # uploads are written but never recognised, so nothing ever expires
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIUploadedThroughTheRoute validator@lium"
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
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIUploadedAfterNoNewline validator@lium"

    SSHService().add_pubkey_to_host(key)
    assert authorized_keys.read_text().startswith(RENTER_KEY + key + " ")

    assert purge_expired_uploaded_keys(TTL, now=time.time() + TTL + 5) == 1
    assert authorized_keys.read_text() == RENTER_KEY


def test_remove_ssh_key_still_removes_a_key_uploaded_with_the_marker(authorized_keys):
    # regression: /remove_ssh_key compares the whole line to the bare key, so a marked line is
    # never matched and every validator key stays until the TTL — 15 minutes of access it did
    # not ask for, on every job
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIValidatorJobKey validator@lium"
    authorized_keys.write_text(RENTER_KEY)
    service = SSHService()
    service.add_pubkey_to_host(key)

    service.remove_pubkey_from_host(key)

    assert authorized_keys.read_text() == RENTER_KEY


def test_remove_ssh_key_removes_a_key_the_previous_release_wrote_without_a_marker(authorized_keys):
    # regression: authorized_keys lives on the disk reserve (setup_disk_reserve.sh mount_ssh) and
    # survives the restart onto this release, so a key the previous release appended (no marker)
    # is still there when the validator asks to remove it
    key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKeyFromTheOldRelease validator@lium"
    authorized_keys.write_text(RENTER_KEY + key + "\n")

    SSHService().remove_pubkey_from_host(key)

    assert authorized_keys.read_text() == RENTER_KEY


def test_a_public_key_with_an_embedded_newline_is_refused_by_the_writer(authorized_keys):
    # regression: "key-a\nkey-b" appended verbatim puts key-b on its own unmarked line — a
    # permanent key smuggled past the TTL by whoever holds the validator hotkey
    with pytest.raises(ValueError):
        SSHService().add_pubkey_to_host("ssh-ed25519 AAAAFirst a\nssh-ed25519 AAAASecond b")

    assert authorized_keys.read_text() == ""


def _signed_upload(monkeypatch, public_key: str):
    """POST /upload_ssh_key through the real app (middleware + route + real SSHService) with real signatures."""
    import bittensor
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from middlewares.miner import MinerMiddleware
    from routes.apis import apis_router

    import dependencies.auth as auth
    from core.config import settings

    miner = bittensor.Keypair.create_from_uri("//LiumExpiryMiner")
    validator = bittensor.Keypair.create_from_uri("//LiumExpiryValidator")
    monkeypatch.setattr(settings, "MINER_HOTKEY_SS58_ADDRESS", miner.ss58_address)
    monkeypatch.setattr(settings, "DEFAULT_MINER_HOTKEY", miner.ss58_address)
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEYS_SS58", {"current": validator.ss58_address})
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    return TestClient(app).post(
        "/upload_ssh_key",
        json={
            "public_key": public_key,
            "data_to_sign": public_key,
            "signature": "0x" + miner.sign(public_key).hex(),
            "validator_signature": "0x" + validator.sign(public_key).hex(),
        },
    )


def test_upload_ssh_key_route_refuses_a_public_key_spanning_two_lines(authorized_keys, monkeypatch):
    # the same property at the door: a correctly signed two-line key is a 400, not a 500 from
    # the writer and not two lines in authorized_keys
    response = _signed_upload(monkeypatch, "ssh-ed25519 AAAAFirst a\nssh-ed25519 AAAASecond b")

    assert response.status_code == 400
    assert authorized_keys.read_text() == ""


def test_upload_ssh_key_route_refuses_a_public_key_over_the_line_bound(authorized_keys, monkeypatch):
    # regression: a correctly signed multi-megabyte "key" lands in authorized_keys, and the purge
    # re-reads the file every minute for as long as it lives
    response = _signed_upload(monkeypatch, "ssh-ed25519 " + "A" * 8192)

    assert response.status_code == 400
    assert authorized_keys.read_text() == ""
