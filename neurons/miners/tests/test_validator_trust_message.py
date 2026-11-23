"""The miner trusts the validator named in a message the subnet owner signed, as the executor does.

Real signatures with ephemeral keypairs; a temp SQLite database stands in for the miner database.
"""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from bittensor_wallet import Keypair
from datura.validator_trust import RELEASE_SUBNET_OWNER_COLDKEY, SignedTrustMessage, trust_line, verify_trust_message
from sqlmodel import Session, SQLModel, create_engine, select

import services.executor_service as executor_service_module
import services.validator_service as validator_service_module
import services.validator_trust_service as trust_service
from daos.executor import ExecutorDao
from models.executor import Executor
from models.validator_trust import ValidatorTrustMessage
from services.executor_service import ExecutorService
from services.validator_service import ValidatorService
from services.validator_trust_service import IMAGE_MESSAGE_PATH, ValidatorTrust

STRANGER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@pytest.fixture(scope="module")
def owner():
    return Keypair.create_from_uri("//LiumTrustOwner")


@pytest.fixture(scope="module")
def old_validator():
    return Keypair.create_from_uri("//LiumTrustOldValidator")


@pytest.fixture(scope="module")
def new_validator():
    return Keypair.create_from_uri("//LiumTrustNewValidator")


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'miner.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def release_owner(monkeypatch, owner):
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_COLDKEY", owner.ss58_address)
    monkeypatch.setattr(trust_service.settings, "COMPUTE_REST_API_URL", "http://backend")
    monkeypatch.setattr(trust_service.settings, "CENTRAL_MODE", False)
    monkeypatch.setattr(trust_service.settings.debug, "SKIP_VALIDATOR_REGISTRATION_CHECK", False)


def _signed(owner: Keypair, validator: Keypair, revision: int) -> SignedTrustMessage:
    line = trust_line(validator.ss58_address, revision)
    return SignedTrustMessage(message=line, signature="0x" + owner.sign(line).hex())


def _serves(monkeypatch, reply) -> None:
    """The backend answers `reply()` (a dict, or None for a 404), or it raises."""

    async def fetch(url, body=None):
        return reply()

    monkeypatch.setattr(trust_service, "fetch_json", fetch)


def _down(*args, **kwargs):
    raise ConnectionError("down")


def _image(tmp_path, signed: SignedTrustMessage):
    image = tmp_path / "image.json"
    image.write_text(json.dumps(signed._asdict()))
    return image


def _start(monkeypatch, engine, image) -> ValidatorTrust:
    """A miner (re)starting on the database `engine` with the image copy `image`."""
    trust = ValidatorTrust(engine=engine, image_message=image)
    trust.load()
    monkeypatch.setattr(validator_service_module, "validator_trust", trust)
    monkeypatch.setattr(executor_service_module, "validator_trust", trust)
    return trust


def _signs_in(validator: str) -> bool:
    return ValidatorService(validator_dao=None).is_valid_validator(validator)


async def _nodes(engine, validator: str) -> set[str]:
    service = ExecutorService.__new__(ExecutorService)
    with Session(engine) as session:
        service.executor_dao = ExecutorDao(session=session)
        return {e.address for e in await service.get_executors_for_validator(validator, "miner")}


@pytest.mark.asyncio
async def test_acceptance_new_validator_survives_recreate_without_backend_or_chain(
    tmp_path, monkeypatch, engine, owner, old_validator, new_validator
):
    with Session(engine) as session:
        session.add(Executor(address="node-a", port=8001, validator=old_validator.ss58_address))
        session.commit()
    image = _image(tmp_path, _signed(owner, old_validator, 1))
    trust = _start(monkeypatch, engine, image)
    assert _signs_in(old_validator.ss58_address)

    # the owner signs a new validator; the miner takes it from the backend and stores it
    _serves(monkeypatch, lambda: _signed(owner, new_validator, 2)._asdict())
    await trust.refresh_message()
    assert trust.report()["validator"] == new_validator.ss58_address

    # the container is recreated from the same image (still naming the old validator); backend and chain are down
    _serves(monkeypatch, _down)
    monkeypatch.setattr(trust_service, "read_subnet_owner_coldkey", AsyncMock(side_effect=ConnectionError("down")))
    trust = _start(monkeypatch, engine, image)
    assert await trust.refresh_owner() is False
    await trust.refresh_message()

    assert trust.report() == {"validator": new_validator.ss58_address, "revision": 2, "subnet_owner": owner.ss58_address}
    assert _signs_in(new_validator.ss58_address)
    assert not _signs_in(old_validator.ss58_address)
    # the node keyed to the old validator is served to the new one, and no longer to the old one
    assert await _nodes(engine, new_validator.ss58_address) == {"node-a"}


@pytest.mark.asyncio
async def test_a_validator_change_moves_the_nodes_but_not_another_validators(
    tmp_path, monkeypatch, engine, owner, old_validator, new_validator
):
    with Session(engine) as session:
        session.add(Executor(address="node-baked", port=8001, validator=trust_service.BAKED_VALIDATOR_HOTKEYS[0]))
        session.add(Executor(address="node-old", port=8001, validator=old_validator.ss58_address))
        session.add(Executor(address="node-other", port=8001, validator=STRANGER))
        session.commit()
    trust = _start(monkeypatch, engine, _image(tmp_path, _signed(owner, old_validator, 1)))
    _serves(monkeypatch, lambda: _signed(owner, new_validator, 2)._asdict())
    await trust.refresh_message()

    assert await _nodes(engine, new_validator.ss58_address) == {"node-baked", "node-old"}
    # a validator the miner does not trust (signed in through the debug bypass) gets only its own rows
    assert await _nodes(engine, old_validator.ss58_address) == {"node-old"}
    assert await _nodes(engine, STRANGER) == {"node-other"}


@pytest.mark.asyncio
async def test_a_central_miner_serves_the_portal_nodes_of_a_validator_trusted_before(
    tmp_path, monkeypatch, engine, owner, old_validator, new_validator
):
    from clients.miner_portal_api import MinerPortalAPI

    trust = _start(monkeypatch, engine, _image(tmp_path, _signed(owner, old_validator, 1)))
    _serves(monkeypatch, lambda: _signed(owner, new_validator, 2)._asdict())
    await trust.refresh_message()
    monkeypatch.setattr(trust_service.settings, "CENTRAL_MODE", True)
    portal = [
        {"id": str(uuid4()), "validator_hotkey": validator, "executor_ip_address": address, "executor_ip_port": "8001"}
        for validator, address in ((old_validator.ss58_address, "node-old"), (STRANGER, "node-other"))
    ]
    with patch.object(MinerPortalAPI, "fetch_executors", AsyncMock(return_value=portal)):
        assert await _nodes(engine, new_validator.ss58_address) == {"node-old"}


def test_start_takes_the_higher_revision_of_image_and_database(tmp_path, monkeypatch, engine, owner, old_validator,
                                                               new_validator):
    with Session(engine) as session:
        signed = _signed(owner, new_validator, 5)
        session.add(ValidatorTrustMessage(revision=5, signer=owner.ss58_address, **signed._asdict()))
        session.commit()

    assert _start(monkeypatch, engine, _image(tmp_path, _signed(owner, old_validator, 3))).held.revision == 5
    assert _start(monkeypatch, engine, _image(tmp_path, _signed(owner, old_validator, 7))).held.revision == 7


@pytest.mark.asyncio
async def test_an_image_message_taken_at_start_is_stored_at_the_first_refresh(
    tmp_path, monkeypatch, engine, owner, new_validator
):
    trust = _start(monkeypatch, engine, _image(tmp_path, _signed(owner, new_validator, 4)))
    _serves(monkeypatch, _down)
    await trust.refresh_message()

    assert _start(monkeypatch, engine, tmp_path / "absent.json").held.revision == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [lambda: None, lambda: {"message": "garbage", "signature": "0x00"}, _down])
async def test_a_404_garbage_or_an_outage_keeps_the_held_message(tmp_path, monkeypatch, engine, owner, new_validator,
                                                                reply):
    trust = _start(monkeypatch, engine, _image(tmp_path, _signed(owner, new_validator, 1)))
    _serves(monkeypatch, reply)
    await trust.refresh_message()

    assert trust.report()["validator"] == new_validator.ss58_address


def test_refuses_another_signer_and_a_revision_not_higher(tmp_path, monkeypatch, engine, owner, old_validator,
                                                          new_validator):
    trust = _start(monkeypatch, engine, _image(tmp_path, _signed(owner, old_validator, 2)))

    assert trust.offer(_signed(new_validator, new_validator, 3)) is False
    assert trust.offer(_signed(owner, new_validator, 2)) is False
    assert trust.report()["validator"] == old_validator.ss58_address


def test_a_badly_signed_stored_row_is_ignored(tmp_path, monkeypatch, engine, owner, new_validator):
    with Session(engine) as session:
        forged = _signed(new_validator, new_validator, 9)
        session.add(ValidatorTrustMessage(revision=9, signer=owner.ss58_address, **forged._asdict()))
        session.commit()

    trust = _start(monkeypatch, engine, tmp_path / "absent.json")
    assert trust.held is None
    assert trust.hotkeys() == trust_service.BAKED_VALIDATOR_HOTKEYS


@pytest.mark.asyncio
async def test_a_failed_write_is_retried_at_the_next_refresh(tmp_path, monkeypatch, engine, owner, new_validator):
    trust = _start(monkeypatch, engine, tmp_path / "absent.json")
    _serves(monkeypatch, lambda: _signed(owner, new_validator, 1)._asdict())
    with patch.object(ValidatorTrust, "_write", side_effect=OSError("database down")):
        await trust.refresh_message()
    await trust.refresh_message()

    with Session(engine) as session:
        assert [row.revision for row in session.exec(select(ValidatorTrustMessage)).all()] == [1]


def test_a_non_prod_build_never_takes_a_message(tmp_path, engine, owner, new_validator):
    trust = ValidatorTrust(engine=engine, image_message=_image(tmp_path, _signed(owner, new_validator, 1)), enabled=False)
    trust.load()

    assert trust.offer(_signed(owner, new_validator, 2)) is False
    assert trust.hotkeys() == trust_service.BAKED_VALIDATOR_HOTKEYS


def test_the_image_copy_is_signed_by_the_release_time_owner_coldkey():
    signed = SignedTrustMessage(**json.loads(IMAGE_MESSAGE_PATH.read_text()))
    assert verify_trust_message(signed, RELEASE_SUBNET_OWNER_COLDKEY).revision >= 1
