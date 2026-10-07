"""DAH-3394: an ssh key installed through /upload_ssh_key expires unless the validator removes it.

The validator uploads a fresh keypair per job and removes it afterwards; a key that is still in the
executor's authorized_keys after EXECUTOR_UPLOADED_KEY_TTL_S is a leak (I-78: keys planted through a
validator-hotkey-signed upload stayed for months). The executor now appends every upload with a marker
carrying the upload time and purges marked lines past the TTL. Everything else in the file — a template's
key, a key the provider added by hand, comments, blank lines — the purge never touches (the one-time
startup stamp of lines the previous release wrote is test_legacy_uploaded_key_stamp.py).
"""

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


def test_a_public_key_with_an_embedded_newline_is_refused_by_the_writer(authorized_keys):
    # regression: "key-a\nkey-b" appended verbatim puts key-b on its own unmarked line — a
    # permanent key smuggled past the TTL by whoever holds the validator hotkey
    with pytest.raises(ValueError):
        SSHService().add_pubkey_to_host("ssh-ed25519 AAAAFirst a\nssh-ed25519 AAAASecond b")

    assert authorized_keys.read_text() == ""


