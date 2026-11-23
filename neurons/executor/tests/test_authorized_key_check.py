"""An uploaded key reaches authorized_keys only as `<type> <base64> [comment]`: sshd reads anything before
the type as options (command=, environment=), so those lines are refused at the route and at the writer."""

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import HTTPException
from payloads.miner import UploadSShKeyPayload
from routes.apis import _validate_ssh_key_consistency
from services.ssh_service import SSHService, check_authorized_key

KEY = (
    ed25519.Ed25519PrivateKey.generate()
    .public_key()
    .public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
    .decode()
)

REFUSED = [
    f'command="id" {KEY}',
    f"environment=\"A=1\" {KEY}",
    f"no-pty {KEY}",
    "ssh-ed25519 AAAAnotakey",
    "ssh-dss AAAAB3NzaC1kc3MAAACBAP",
    "ssh-ed25519",
    "",
]


@pytest.mark.parametrize("line", [KEY, f"{KEY} validator@lium"])
def test_a_plain_public_key_passes(line):
    check_authorized_key(line)


@pytest.mark.parametrize("line", REFUSED)
def test_options_or_a_key_that_does_not_parse_is_refused(line):
    with pytest.raises(ValueError):
        check_authorized_key(line)


@pytest.mark.parametrize("line", REFUSED[:4])
def test_the_route_answers_400(line):
    payload = UploadSShKeyPayload(
        data_to_sign=line, signature="s", public_key=line, validator_signature="v"
    )
    with pytest.raises(HTTPException) as refused:
        _validate_ssh_key_consistency(payload)
    assert refused.value.status_code == 400


def test_the_writer_refuses_an_option_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ssh_service.authorized_keys_path", lambda: str(tmp_path / "authorized_keys"))
    with pytest.raises(ValueError):
        SSHService().add_pubkey_to_host(f'command="id" {KEY}')
    assert not (tmp_path / "authorized_keys").exists()
