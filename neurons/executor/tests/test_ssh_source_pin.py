"""I-78: a key uploaded through /upload_ssh_key logs in only from the validator's addresses."""

import socket
import time

import pytest

import core.config as config
import services.ssh_service as ssh_service
from services.ssh_service import SSHService, purge_expired_uploaded_keys

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIValidatorJobKey validator@lium"
RENTER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIRenterOwnKey renter@laptop\n"


@pytest.fixture()
def authorized_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir(mode=0o700)
    path = tmp_path / ".ssh" / "authorized_keys"
    path.write_text(RENTER_KEY)
    return path


def test_prod_build_pins_the_nat_gateways():
    assert config.VALIDATOR_SSH_SOURCES == "54.163.250.18,13.223.237.157"


def test_the_pin_has_no_opt_out():
    assert "EXECUTOR_SSH_SOURCE_PIN" not in config.Settings.model_fields


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


def test_no_pin_without_sources(authorized_keys, monkeypatch):
    monkeypatch.setattr(ssh_service, "VALIDATOR_SSH_SOURCES", "")
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + KEY + " ")


def test_no_pin_in_a_cvm(authorized_keys, tmp_path, monkeypatch):
    # QEMU user networking shows sshd 10.0.2.2, never the validator's address
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
