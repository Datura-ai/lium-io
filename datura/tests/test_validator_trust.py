import pytest

from bittensor_wallet import Keypair

from datura.validator_trust import (
    RELEASE_SUBNET_OWNER_COLDKEY,
    SignedTrustMessage,
    override_trust,
    subnet_owner_from_reply,
    subnet_owner_request,
    subnet_owner_storage_key,
    trust_line,
)


def test_storage_key_is_the_identity_hashed_netuid():
    assert subnet_owner_storage_key(51).endswith("3300")
    assert subnet_owner_storage_key(51).startswith("0x658faa385070e074c85bf6b568cf0555")


def test_reads_and_encodes_the_owner():
    from scalecodec.utils.ss58 import ss58_decode

    assert subnet_owner_request(51)["params"] == [subnet_owner_storage_key(51)]
    reply = {"result": "0x" + ss58_decode(RELEASE_SUBNET_OWNER_COLDKEY)}
    assert subnet_owner_from_reply(reply) == RELEASE_SUBNET_OWNER_COLDKEY


@pytest.mark.parametrize("reply", [{}, {"result": None}, {"result": "0x00"}, ["x"]])
def test_refuses_any_other_reply(reply):
    with pytest.raises(ValueError):
        subnet_owner_from_reply(reply)


def _override(tmp_path, monkeypatch, name, text):
    (tmp_path / f"{name}.py").write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))


def test_override_trust_release_build_staging_owner_and_plain_non_prod(tmp_path, monkeypatch):
    assert override_trust("no_such_override_module_6670") == {}   # release build: unchanged
    _override(tmp_path, monkeypatch, "plain_override_6670", '_VALIDATOR_HOTKEY_SS58 = "x"\n')
    assert override_trust("plain_override_6670") == {"enabled": False}   # non-prod without an owner: as before
    staging, validator = Keypair.create_from_uri("//StagingOwner6670"), Keypair.create_from_uri("//StagingValidator6670")
    line = trust_line(validator.ss58_address, 1)
    signed = {"message": line, "signature": "0x" + staging.sign(line).hex()}
    _override(tmp_path, monkeypatch, "staging_override_6670",
              f"_STAGING_TRUST_OWNER_SS58 = {staging.ss58_address!r}\n_STAGING_TRUST_MESSAGE = {signed!r}\n")
    assert override_trust("staging_override_6670") == {
        "owner": staging.ss58_address, "read_chain": False, "message": SignedTrustMessage(**signed)}
    other = Keypair.create_from_uri("//NotTheStagingOwner6670")
    _override(tmp_path, monkeypatch, "badsig_override_6670",
              f"_STAGING_TRUST_OWNER_SS58 = {other.ss58_address!r}\n_STAGING_TRUST_MESSAGE = {signed!r}\n")
    with pytest.raises(ValueError):
        override_trust("badsig_override_6670")
    _override(tmp_path, monkeypatch, "badowner_override_6670", '_STAGING_TRUST_OWNER_SS58 = "not-an-address"\n')
    with pytest.raises(ValueError):
        override_trust("badowner_override_6670")
