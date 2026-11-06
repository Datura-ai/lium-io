"""I-78: a key uploaded through /upload_ssh_key logs in only from the validator's addresses."""

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


def test_the_pin_is_off_by_default():
    assert config.Settings.model_fields["EXECUTOR_SSH_SOURCE_PIN"].default is False


def test_upload_is_pinned_and_removal_and_expiry_still_find_it(authorized_keys, monkeypatch):
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_SSH_SOURCE_PIN", True)
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
def test_no_pin_without_sources_or_with_the_opt_out(authorized_keys, monkeypatch, sources, pin_on):
    monkeypatch.setattr(ssh_service, "VALIDATOR_SSH_SOURCES", sources)
    monkeypatch.setattr(ssh_service.settings, "EXECUTOR_SSH_SOURCE_PIN", pin_on)
    SSHService().add_pubkey_to_host(KEY)
    assert authorized_keys.read_text().startswith(RENTER_KEY + KEY + " ")


def test_a_malformed_source_fails_at_import(monkeypatch):
    monkeypatch.setattr(config, "_BUILTIN_VALIDATOR_SSH_SOURCES", "54.163.250.18,not-an-ip")
    with pytest.raises(ValueError):
        config._resolve_validator_ssh_sources()
