import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # the receiver sits beside run.sh, not in src/

import sshd_log_receiver  # noqa: E402
from sshd_log_receiver import record  # noqa: E402

ACCEPTED = "<38>Oct 18 06:27:30 sshd[812]: Accepted publickey for root from 203.0.113.9 port 51022 ssh2: ED25519 SHA256:aaa"


def test_a_login_is_kept_with_sshds_envelope(tmp_path):
    log = tmp_path / "sshd-auth.log"

    record(ACCEPTED, str(log))
    record(
        "<38>Oct 18 06:27:31 sshd[813]: Connection closed by 198.51.100.4 port 4022 [preauth]",
        str(log),
    )

    lines = log.read_text().splitlines()
    assert len(lines) == 1 and lines[0].endswith(ACCEPTED)


@pytest.mark.parametrize(
    "offer",
    [
        "<38>Oct 18 06:27:29 sshd[812]: Accepted key ED25519 SHA256:aaa found at /root/.ssh/authorized_keys:3",
        "<38>Oct 18 06:27:31 sshd[813]: Connection closed by 198.51.100.4 port 4022 [preauth]",
        # a client-chosen user name, unauthenticated
        "<38>Oct 18 06:27:32 sshd[815]: Invalid user Accepted publickey for root from 198.51.100.4 port 4023",
        "<38>Oct 18 06:27:32 sshd[815]: Invalid user x sshd[1]: Accepted publickey for root from 198.51.100.4",
    ],
)
def test_only_a_completed_login_is_recorded(tmp_path, offer):
    log = tmp_path / "sshd-auth.log"

    record(offer, str(log))

    assert not log.exists()


class FakeSocket:
    """Two datagrams, then stops the loop at the next blocking recv."""

    def __init__(self, datagrams):
        self.datagrams = list(datagrams)

    def settimeout(self, value):
        pass

    def recv(self, size):
        if not self.datagrams:
            raise KeyboardInterrupt
        return self.datagrams.pop(0).encode()


@pytest.mark.parametrize("noise_count", [0, sshd_log_receiver.MAX_PENDING + 1])
def test_a_failed_write_is_retried_instead_of_ending_the_receiver(
    tmp_path, monkeypatch, noise_count
):
    log = tmp_path / "sshd-auth.log"
    noise = "<38>Oct 18 06:27:31 sshd[813]: Connection closed by 198.51.100.4 port 4022 [preauth]"
    sock = FakeSocket([ACCEPTED] + [noise] * noise_count + [ACCEPTED.replace("812", "814")])

    def flaky(message, log=str(log)):
        if sock.datagrams:  # the disk stays full until the last login has arrived
            raise OSError(28, "No space left on device")
        record(message, log)

    monkeypatch.setattr(sshd_log_receiver, "record", flaky)

    with pytest.raises(KeyboardInterrupt):
        sshd_log_receiver.serve(sock)

    assert len(log.read_text().splitlines()) == 2


def test_a_full_log_is_rotated_before_the_next_login(tmp_path, monkeypatch):
    monkeypatch.setattr(sshd_log_receiver, "ROTATE_BYTES", 10)
    log = tmp_path / "sshd-auth.log"
    log.write_text("x" * 20 + "\n")

    record(ACCEPTED, str(log))

    assert (tmp_path / "sshd-auth.log.1").read_text() == "x" * 20 + "\n"
    assert log.read_text().endswith(f"{ACCEPTED}\n")


def test_an_unread_rotated_log_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(sshd_log_receiver, "ROTATE_BYTES", 10)
    log = tmp_path / "sshd-auth.log"
    (tmp_path / "sshd-auth.log.1").write_text("unread\n")
    log.write_text("x" * 20 + "\n")

    record(ACCEPTED, str(log))

    assert (tmp_path / "sshd-auth.log.1").read_text() == "unread\n"
    assert log.read_text().startswith("x" * 20 + "\n") and log.read_text().endswith(f"{ACCEPTED}\n")


def test_a_full_spool_drops_logins_with_a_marker_then_a_count(tmp_path, monkeypatch):
    log = tmp_path / "sshd-auth.log"
    record(ACCEPTED, str(log))
    monkeypatch.setattr(sshd_log_receiver, "MAX_SPOOL_BYTES", log.stat().st_size)

    for _ in range(3):
        record(ACCEPTED, str(log))
    full = log.read_text().splitlines()
    log.rename(f"{log}.1")
    (tmp_path / "sshd-auth.log.1").unlink()  # the validator drained it
    record(ACCEPTED, str(log))

    assert len(full) == 2 and full[1].endswith("spool full, dropping logins")
    lines = log.read_text().splitlines()
    assert lines[0].endswith("3 logins dropped while the spool was full") and lines[1].endswith(
        ACCEPTED
    )
