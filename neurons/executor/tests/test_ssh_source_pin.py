"""I-78: a key uploaded through /upload_ssh_key logs in only from the validator's addresses."""

import asyncio
import socket
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import core.config as config
import services.ssh_service as ssh_service
from services.ssh_service import SSHService, purge_expired_uploaded_keys

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIValidatorJobKey validator@lium"
RENTER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIRenterOwnKey renter@laptop\n"


@pytest.fixture(autouse=True)
def signed_sources(monkeypatch):
    """The list /watchtower/ssh-sources signs; a test sets `.value` to change it."""
    holder = SimpleNamespace(value=config.VALIDATOR_SSH_SOURCES)
    monkeypatch.setattr(ssh_service._signed_ssh_sources, "get", lambda: holder.value)
    return holder


@pytest.fixture()
def authorized_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir(mode=0o700)
    path = tmp_path / ".ssh" / "authorized_keys"
    path.write_text(RENTER_KEY)
    return path


def test_prod_build_pins_the_nat_gateways():
    assert config.VALIDATOR_SSH_SOURCES == "54.163.250.18,13.223.237.157"


def test_the_pin_is_on_by_default():
    assert config.Settings.model_fields["EXECUTOR_SSH_SOURCE_PIN"].default is True


def test_upload_is_pinned_and_removal_and_expiry_still_find_it(authorized_keys):
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + f'from="{config.VALIDATOR_SSH_SOURCES}" {KEY} ')

    SSHService().remove_pubkey_from_host(KEY)
    assert authorized_keys.read_text() == RENTER_KEY

    SSHService().add_pubkey_to_host(KEY)
    assert purge_expired_uploaded_keys(900, now=time.time() + 905) == 1
    assert authorized_keys.read_text() == RENTER_KEY


def test_a_key_pinned_by_an_older_list_is_still_removed(authorized_keys):
    authorized_keys.write_text(RENTER_KEY + f'from="10.0.0.1" {KEY} lium-uploaded-at=1\n')
    SSHService().remove_pubkey_from_host(KEY)
    assert authorized_keys.read_text() == RENTER_KEY


@pytest.mark.parametrize("sources,pin_on", [("", True), ("54.163.250.18", False)])
def test_no_pin_without_sources_or_with_the_switch_off(authorized_keys, monkeypatch, sources, pin_on):
    monkeypatch.setattr(ssh_service, "VALIDATOR_SSH_SOURCES", sources)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_SSH_SOURCE_PIN", pin_on)
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + KEY + " ")


def test_no_pin_in_a_cvm(authorized_keys, tmp_path, monkeypatch):
    # QEMU user networking shows sshd its own gateway address, never the validator's
    sock_path = tmp_path / "dstack.sock"
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(sock_path))
    monkeypatch.setattr(ssh_service.settings, "DSTACK_SOCKET_PATH", str(sock_path))
    try:
        SSHService().add_pubkey_to_host(KEY)
    finally:
        sock.close()
    assert authorized_keys.read_text().startswith(RENTER_KEY + KEY + " ")


def test_a_dstack_path_that_is_not_a_socket_still_pins(authorized_keys, tmp_path, monkeypatch):
    # docker-compose.app.yml bind-mounts the path on every host; without dstack docker makes a directory
    (tmp_path / "dstack.sock").mkdir()
    monkeypatch.setattr(ssh_service.settings, "DSTACK_SOCKET_PATH", str(tmp_path / "dstack.sock"))
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + f'from="{config.VALIDATOR_SSH_SOURCES}" ')


def test_a_malformed_source_fails_at_import(monkeypatch):
    monkeypatch.setattr(config, "_BUILTIN_VALIDATOR_SSH_SOURCES", "54.163.250.18,not-an-ip")
    with pytest.raises(ValueError):
        config._resolve_validator_ssh_sources()


def test_the_signed_list_is_what_gets_pinned(authorized_keys, signed_sources):
    signed_sources.value = "198.51.100.7"
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + 'from="198.51.100.7" ')


def test_an_empty_signed_list_keeps_the_pin_off(authorized_keys, signed_sources):
    signed_sources.value = ""
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + KEY + " ")


@pytest.fixture()
def clock(monkeypatch):
    monkeypatch.setattr(ssh_service.time, "time", lambda: 1000.0)


def answer(sources="198.51.100.7", timestamp=1000):
    return ssh_service.SshSourcesAnswer(sources=sources, timestamp=timestamp, signature="0xab")


@pytest.mark.parametrize(
    "sources,timestamp,signed",
    [
        ("198.51.100.7", 1000, False),
        ("198.51.100.7,not-an-ip", 1000, True),
        ("198.51.100.7,", 1000, True),
        ("198.51.100.7", 1000 - 3600, True),
    ],
)
def test_a_bad_answer_is_refused(monkeypatch, clock, sources, timestamp, signed):
    monkeypatch.setattr(ssh_service, "match_validator_hotkey", lambda message, sig: "current" if signed else None)
    with pytest.raises(ValueError):
        ssh_service.verify_signed_ssh_sources(answer(sources, timestamp))


def test_an_answer_without_sources_is_refused():
    with pytest.raises(ValueError):
        ssh_service.SshSourcesAnswer.model_validate({"timestamp": 1000, "signature": "0xab"})


def test_spaces_the_backend_accepts_are_stripped(monkeypatch, clock):
    # lium-platform accepts `a, b`; sshd's from= takes no spaces
    monkeypatch.setattr(ssh_service, "match_validator_hotkey", lambda message, sig: "current")
    assert ssh_service.verify_signed_ssh_sources(answer("198.51.100.7, 198.51.100.8")) == "198.51.100.7,198.51.100.8"


def test_a_failed_refresh_keeps_the_last_good_list(monkeypatch):
    monkeypatch.setattr(ssh_service.settings, "COMPUTE_REST_API_URL", "https://backend.test/api")
    monkeypatch.setattr(ssh_service, "match_validator_hotkey", lambda message, sig: "current")
    now = int(time.time())
    good = MagicMock(json=MagicMock(return_value={"sources": "198.51.100.7", "timestamp": now, "signature": "0xab"}))
    get = MagicMock(side_effect=[good, OSError("down")])
    monkeypatch.setattr(ssh_service.requests, "get", get)
    cache = ssh_service.SignedSshSources()

    cache.refresh()
    first = cache.get()
    cache.refresh()

    assert (first, cache.get()) == ("198.51.100.7", "198.51.100.7")
    assert get.call_count == 2
    assert get.call_args.args[0] == "https://backend.test/api/watchtower/ssh-sources"


def test_a_node_that_never_got_a_list_does_not_pin(monkeypatch):
    monkeypatch.setattr(ssh_service.settings, "COMPUTE_REST_API_URL", "https://backend.test/api")
    monkeypatch.setattr(ssh_service.requests, "get", MagicMock(side_effect=OSError("down")))
    cache = ssh_service.SignedSshSources()
    cache.refresh()
    assert cache.get() == ""


def test_the_signed_message_carries_the_ssh_sources_prefix(monkeypatch, clock):
    # a signed digest answer must never read as a signed source list, nor the other way round
    messages = []
    monkeypatch.setattr(ssh_service, "match_validator_hotkey", lambda message, sig: messages.append(message) or "current")

    assert ssh_service.verify_signed_ssh_sources(answer()) == "198.51.100.7"
    assert messages == ["ssh-sources:198.51.100.7:1000"]


@pytest.mark.parametrize("sources,pin_on", [("", True), ("198.51.100.7", False)])
def test_no_refresh_where_the_pin_never_applies(monkeypatch, sources, pin_on):
    monkeypatch.setattr(ssh_service, "VALIDATOR_SSH_SOURCES", sources)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_SSH_SOURCE_PIN", pin_on)
    get = MagicMock()
    monkeypatch.setattr(ssh_service.requests, "get", get)

    asyncio.run(asyncio.wait_for(ssh_service.run_ssh_sources_refresh(), timeout=1))  # returns, never loops

    get.assert_not_called()
