"""DAH-6590: the executor trusts the validator named in a message the subnet owner signed.

Real signatures with ephemeral keypairs; a temp dir stands in for the reserve_data volume.
"""

import json
from unittest.mock import MagicMock

import pytest
import requests
from bittensor_wallet import Keypair

import dependencies.auth as auth
import services.validator_trust_service as trust_service
from datura.validator_trust import SignedTrustMessage, trust_line, verify_trust_message
from services.validator_trust_service import MESSAGE_FILE, OWNER_FILE, ValidatorTrust


@pytest.fixture(scope="module")
def owner():
    return Keypair.create_from_uri("//LiumTrustOwner")


@pytest.fixture(scope="module")
def old_validator():
    return Keypair.create_from_uri("//LiumTrustOldValidator")


@pytest.fixture(scope="module")
def new_validator():
    return Keypair.create_from_uri("//LiumTrustNewValidator")


def _signed(owner: Keypair, validator: Keypair, revision: int) -> SignedTrustMessage:
    line = trust_line(validator.ss58_address, revision)
    return SignedTrustMessage(message=line, signature="0x" + owner.sign(line).hex())


def _trust(tmp_path, owner: Keypair, image_message=None) -> ValidatorTrust:
    (tmp_path / OWNER_FILE).write_text(f'{{"owner": "{owner.ss58_address}"}}')
    trust = ValidatorTrust(state_dir=tmp_path, image_message=image_message or tmp_path / "absent.json")
    trust.load()
    return trust


def _accepts(monkeypatch, trust: ValidatorTrust, validator: Keypair, baked: dict[str, str]) -> str | None:
    monkeypatch.setattr(auth, "validator_trust", trust)
    monkeypatch.setattr(auth, "VALIDATOR_HOTKEYS_SS58", baked)
    return auth.match_validator_hotkey("ping_request", "0x" + validator.sign("ping_request").hex())


def test_acceptance_new_validator_survives_recreate_without_backend_or_chain(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    baked = {"current": old_validator.ss58_address}
    running = _trust(tmp_path, owner)
    assert _accepts(monkeypatch, running, old_validator, baked) == "current"

    # the backend serves the owner's message naming the new validator
    answer = MagicMock(status_code=200)
    answer.json.return_value = _signed(owner, new_validator, 1)._asdict()
    monkeypatch.setattr(trust_service.requests, "get", lambda *a, **k: answer)
    running.refresh_message()
    assert _accepts(monkeypatch, running, new_validator, baked) == "signed"
    assert _accepts(monkeypatch, running, old_validator, baked) is None

    # container recreated; backend and chain both unreachable
    def down(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(trust_service.requests, "get", down)
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", down)
    recreated = _trust(tmp_path, owner)
    recreated.refresh_message()
    recreated.refresh_owner()
    assert recreated.report() == {
        "validator": new_validator.ss58_address,
        "revision": 1,
        "subnet_owner": owner.ss58_address,
    }
    assert _accepts(monkeypatch, recreated, new_validator, baked) == "signed"
    assert _accepts(monkeypatch, recreated, old_validator, baked) is None


def test_refuses_another_signer_and_a_revision_not_higher(tmp_path, owner, old_validator, new_validator):
    trust = _trust(tmp_path, owner)
    assert not trust.offer(_signed(new_validator, new_validator, 5))  # not the owner
    assert trust.offer(_signed(owner, new_validator, 2))
    assert not trust.offer(_signed(owner, old_validator, 2))  # same revision
    assert not trust.offer(_signed(owner, old_validator, 1))  # rollback
    assert trust.offer(_signed(owner, old_validator, 3))  # two changes on one day both pass
    assert trust.held.validator == old_validator.ss58_address


def test_start_takes_the_higher_revision_of_image_and_volume(tmp_path, monkeypatch, owner, old_validator, new_validator):
    # the image copy is checked against the owner at release
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, old_validator, 4)))
    (tmp_path / MESSAGE_FILE).write_text(_dump(_signed(owner, new_validator, 3)))
    assert _trust(tmp_path, owner, image).held.validator == old_validator.ss58_address
    (tmp_path / MESSAGE_FILE).write_text(_dump(_signed(owner, new_validator, 5)))
    assert _trust(tmp_path, owner, image).held.validator == new_validator.ss58_address


def test_a_404_or_garbage_keeps_the_held_message(tmp_path, monkeypatch, owner, new_validator):
    trust = _trust(tmp_path, owner)
    trust.offer(_signed(owner, new_validator, 1))
    for answer in (MagicMock(status_code=404), MagicMock(status_code=200, json=MagicMock(return_value={"x": 1}))):
        monkeypatch.setattr(trust_service.requests, "get", lambda *a, **k: answer)
        trust.refresh_message()
        assert trust.held.validator == new_validator.ss58_address


def test_owner_change_is_kept_and_checks_the_next_message(tmp_path, monkeypatch, owner, new_validator):
    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout: new_owner.ss58_address)
    trust.refresh_owner()
    assert _trust(tmp_path, new_owner).owner == new_owner.ss58_address  # kept on the volume
    assert trust.offer(_signed(new_owner, new_validator, 1))


def test_no_message_keeps_the_baked_hotkeys(tmp_path, monkeypatch, owner, old_validator):
    assert _accepts(monkeypatch, _trust(tmp_path, owner), old_validator, {"current": old_validator.ss58_address}) == "current"


def test_a_non_prod_build_never_takes_a_message(tmp_path, owner, new_validator):
    trust = ValidatorTrust(state_dir=tmp_path, image_message=tmp_path / "absent.json", enabled=False)
    assert not trust.offer(_signed(owner, new_validator, 1))
    assert trust.hotkeys({"current": "x"}) == {"current": "x"}


@pytest.mark.parametrize(
    "message",
    ["lium-validator:{v}:01", "lium-validator:{v}:1 ", "lium-validator:{v}", "other:{v}:1", "lium-validator:5Fbad:1"],
)
def test_the_line_is_exact(owner, new_validator, message):
    line = message.format(v=new_validator.ss58_address)
    with pytest.raises(ValueError):
        verify_trust_message(SignedTrustMessage(line, "0x" + owner.sign(line).hex()), owner.ss58_address)


def _dump(signed: SignedTrustMessage) -> str:
    import json

    return json.dumps(signed._asdict())


def test_an_owner_change_keeps_the_held_validator_across_a_restart(tmp_path, monkeypatch, owner, new_validator):
    # review finding: owner A signed the held message, the chain then names owner C, the node restarts
    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    assert trust.offer(_signed(owner, new_validator, 7))
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout: new_owner.ss58_address)
    trust.refresh_owner()
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=tmp_path / "absent.json")
    restarted.load()
    assert restarted.owner == new_owner.ss58_address
    assert restarted.report()["validator"] == new_validator.ss58_address
    assert restarted.held.revision == 7


def test_a_new_owner_starts_its_own_revisions(tmp_path, monkeypatch, owner, old_validator, new_validator):
    new_owner = Keypair.create_from_uri("//LiumTrustNewOwner")
    trust = _trust(tmp_path, owner)
    assert trust.offer(_signed(owner, new_validator, 50))
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout: new_owner.ss58_address)
    trust.refresh_owner()
    assert not trust.offer(_signed(owner, old_validator, 51))  # the old owner no longer signs
    assert trust.offer(_signed(new_owner, old_validator, 1))
    assert not trust.offer(_signed(new_owner, new_validator, 1))  # and the floor holds within the new owner


def test_a_failed_write_is_written_once_the_volume_takes_it(tmp_path, monkeypatch, owner, new_validator):
    # review finding: a full volume at acceptance must not leave the revision unsaved for good
    trust = _trust(tmp_path, owner)
    real_write = trust_service._write_json

    def full(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trust_service, "_write_json", full)
    assert trust.offer(_signed(owner, new_validator, 7))
    assert not (tmp_path / MESSAGE_FILE).exists()
    monkeypatch.setattr(trust_service, "_write_json", real_write)
    monkeypatch.setattr(trust_service.requests, "get", lambda *a, **k: MagicMock(status_code=404))
    trust.refresh_message()
    restarted = _trust(tmp_path, owner)
    assert restarted.held == trust.held


def test_one_public_node_cannot_name_an_owner(tmp_path, monkeypatch, owner):
    trust = _trust(tmp_path, owner)
    liar = Keypair.create_from_uri("//LiumTrustLiar").ss58_address
    answers = iter([liar, owner.ss58_address])
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout: next(answers))
    trust.refresh_owner()
    assert trust.owner == owner.ss58_address

    def one_down(url, timeout):
        if url == trust_service.PUBLIC_CHAIN_NODES[0]:
            raise requests.ConnectionError("down")
        return liar

    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", one_down)
    trust.refresh_owner()
    assert trust.owner == owner.ss58_address


def test_the_image_copy_needs_the_release_owner(tmp_path, owner, new_validator):
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, new_validator, 1)))  # not signed by the release owner
    assert _trust(tmp_path, owner, image).held is None


def test_owner_a_b_a_prefers_the_current_owners_message(tmp_path, monkeypatch, owner, old_validator, new_validator):
    # review finding: image copy signed by A (revision 8), kept message signed by B, the owner file still says B
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, new_validator, 8)))
    (tmp_path / MESSAGE_FILE).write_text(
        json.dumps({**_signed(owner_b, old_validator, 3)._asdict(), "owner": owner_b.ss58_address})
    )
    trust = _trust(tmp_path, owner_b, image)
    assert trust.held.validator == old_validator.ss58_address  # B is still the owner the node knows
    # the chain names A again: the image message A signed is held at once, with no backend
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout: owner.ss58_address)
    monkeypatch.setattr(trust_service.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    trust.refresh_owner()
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    # a restart with the owner file now saying A keeps it, and A's lower revisions stay refused
    restarted = _trust(tmp_path, owner, image)
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 8)
    assert not restarted.offer(_signed(owner, old_validator, 5))


def test_an_owner_change_never_restores_an_older_message_of_the_former_owner(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    # review finding: owner A, image revision 8 (replacement), kept revision 3 (revoked); the chain then names B
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, new_validator, 8)))
    (tmp_path / MESSAGE_FILE).write_text(
        json.dumps({**_signed(owner, old_validator, 3)._asdict(), "owner": owner.ss58_address})
    )
    trust = _trust(tmp_path, owner, image)
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout: owner_b.ss58_address)
    trust.refresh_owner()
    assert (trust.held.validator, trust.held.revision) == (new_validator.ss58_address, 8)
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=image)
    restarted.load()
    assert restarted.owner == owner_b.ss58_address
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 8)


def test_a_restart_never_compares_revisions_of_different_owners(
    tmp_path, monkeypatch, owner, old_validator, new_validator
):
    # review finding: image copy A revision 8 (old validator), B then signs revision 1 (new validator), the chain
    # names C, who has published nothing; the node restarts
    owner_b = Keypair.create_from_uri("//LiumTrustNewOwner")
    owner_c = Keypair.create_from_uri("//LiumTrustThirdOwner")
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_HOTKEY", owner.ss58_address)
    image = tmp_path / "image.json"
    image.write_text(_dump(_signed(owner, old_validator, 8)))
    trust = _trust(tmp_path, owner, image)
    for new_owner in (owner_b, owner_c):
        monkeypatch.setattr(trust_service, "read_subnet_owner_hotkey", lambda url, timeout, o=new_owner: o.ss58_address)
        trust.refresh_owner()
        if new_owner is owner_b:
            assert trust.offer(_signed(owner_b, new_validator, 1))
    assert trust.held.validator == new_validator.ss58_address
    restarted = ValidatorTrust(state_dir=tmp_path, image_message=image)
    restarted.load()
    assert restarted.owner == owner_c.ss58_address
    assert (restarted.held.validator, restarted.held.revision) == (new_validator.ss58_address, 1)
