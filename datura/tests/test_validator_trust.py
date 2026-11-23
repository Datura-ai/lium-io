import pytest

from bittensor_wallet import Keypair

from datura.validator_trust import (
    RELEASE_SUBNET_OWNER_COLDKEY,
    STAGING_NETWORK,
    SignedTrustMessage,
    override_trust,
    subnet_owner_from_reply,
    subnet_owner_request,
    subnet_owner_storage_key,
    trust_line,
    verify_trust_message,
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
    line = trust_line(validator.ss58_address, 1, STAGING_NETWORK)
    signed = {"message": line, "signature": "0x" + staging.sign(line).hex()}
    url = "https://staging.example/api"
    _override(tmp_path, monkeypatch, "staging_override_6670",
              f"_STAGING_TRUST_OWNER_SS58 = {staging.ss58_address!r}\n_STAGING_TRUST_MESSAGE = {signed!r}\n"
              f"_STAGING_TRUST_API_URL = {url!r}\n")
    assert override_trust("staging_override_6670") == {
        "owner": staging.ss58_address, "read_chain": False, "message": SignedTrustMessage(**signed), "api_url": url,
        "network": STAGING_NETWORK}
    # a release line signed by the staging owner is not a staging build's start message
    release_line = trust_line(validator.ss58_address, 1)
    release_signed = {"message": release_line, "signature": "0x" + staging.sign(release_line).hex()}
    _override(tmp_path, monkeypatch, "relline_override_6670",
              f"_STAGING_TRUST_OWNER_SS58 = {staging.ss58_address!r}\n_STAGING_TRUST_MESSAGE = {release_signed!r}\n"
              f"_STAGING_TRUST_API_URL = {url!r}\n")
    with pytest.raises(ValueError, match="of this build"):
        override_trust("relline_override_6670")
    # a staging owner without the backend that serves its messages would read the host's (prod) backend: refused
    for bad in ("", '_STAGING_TRUST_API_URL = "http://staging.example/api"\n', '_STAGING_TRUST_API_URL = "https://x/a\\"b"\n'):
        _override(tmp_path, monkeypatch, f"nourl_override_6670_{len(bad)}",
                  f"_STAGING_TRUST_OWNER_SS58 = {staging.ss58_address!r}\n_STAGING_TRUST_MESSAGE = {signed!r}\n{bad}")
        with pytest.raises(ValueError, match="_STAGING_TRUST_API_URL"):
            override_trust(f"nourl_override_6670_{len(bad)}")
    other = Keypair.create_from_uri("//NotTheStagingOwner6670")
    _override(tmp_path, monkeypatch, "badsig_override_6670",
              f"_STAGING_TRUST_OWNER_SS58 = {other.ss58_address!r}\n_STAGING_TRUST_MESSAGE = {signed!r}\n")
    with pytest.raises(ValueError):
        override_trust("badsig_override_6670")
    _override(tmp_path, monkeypatch, "badowner_override_6670", '_STAGING_TRUST_OWNER_SS58 = "not-an-address"\n')
    with pytest.raises(ValueError):
        override_trust("badowner_override_6670")


def test_a_line_signed_for_one_network_never_verifies_as_the_other():
    owner, validator = Keypair.create_from_uri("//NetOwner6670"), Keypair.create_from_uri("//NetValidator6670").ss58_address
    staging_line, release_line = trust_line(validator, 7, STAGING_NETWORK), trust_line(validator, 7)
    assert staging_line == f"lium-validator:test:{validator}:7" and release_line == f"lium-validator:{validator}:7"
    staging_sig = SignedTrustMessage(staging_line, "0x" + owner.sign(staging_line).hex())
    release_sig = SignedTrustMessage(release_line, "0x" + owner.sign(release_line).hex())
    assert verify_trust_message(staging_sig, owner.ss58_address, STAGING_NETWORK) == (validator, 7)
    assert verify_trust_message(release_sig, owner.ss58_address) == (validator, 7)
    for signed, network in ((staging_sig, ""), (release_sig, STAGING_NETWORK)):
        with pytest.raises(ValueError, match="of this build"):
            verify_trust_message(signed, owner.ss58_address, network)
