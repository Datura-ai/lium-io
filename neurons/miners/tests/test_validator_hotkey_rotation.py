"""The validator hotkeys a miner trusts with no signed message taken (the baked fallback), and the CLI that re-keys
executor rows by hand. The signed message itself: test_validator_trust_message.py.
"""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import services.validator_service as validator_service
from daos.executor import ExecutorDao
from models.executor import Executor
from services.validator_service import ValidatorService, migrate_validator_hotkey_rows
from services.validator_trust_service import ValidatorTrust

CURRENT = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
NEXT = "5F1aH9124FPvDNMWKgPCCJ8WeBBV3B3vpijDCyxZVyZgrEV9"
STRANGER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@pytest.fixture()
def no_message(monkeypatch, tmp_path):
    """A miner that never took a signed message."""
    monkeypatch.setattr(validator_service.settings.debug, "SKIP_VALIDATOR_REGISTRATION_CHECK", False)
    monkeypatch.setattr(validator_service, "validator_trust", ValidatorTrust(image_message=tmp_path / "absent.json"))


def _service() -> ValidatorService:
    return ValidatorService(validator_dao=None)


def test_with_no_message_both_baked_lium_hotkeys_sign_in(no_message):
    assert _service().is_valid_validator(CURRENT) is True
    assert _service().is_valid_validator(NEXT) is True


def test_a_third_hotkey_is_refused(no_message):
    # regression: the check becomes "anything non-empty" or a substring match once it is a set
    assert _service().is_valid_validator(STRANGER) is False
    assert _service().is_valid_validator("") is False


def test_an_env_entry_cannot_name_the_validator(no_message, monkeypatch):
    # regression: a local .env repoints the miner at another validator
    monkeypatch.setenv("DEFAULT_VALIDATOR_HOTKEY", STRANGER)
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY", STRANGER)
    assert _service().is_valid_validator(STRANGER) is False


# --- cli.py migrate-validator-hotkey: re-keys the executor rows by hand ------------------------------------------


@pytest.fixture()
def rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rotation.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Executor(address="10.0.0.1", port=22, validator=CURRENT))
        session.add(Executor(address="10.0.0.2", port=22, validator=CURRENT))
        session.add(Executor(address="10.0.0.3", port=22, validator=STRANGER))
        session.commit()
    with Session(engine) as session:
        yield session


def _validators(session) -> dict[str, str]:
    return {e.address: e.validator for e in session.exec(select(Executor)).all()}


def test_the_dry_run_counts_the_rows_keyed_to_the_old_hotkey_and_moves_none(rows):
    result = migrate_validator_hotkey_rows(ExecutorDao(session=rows), CURRENT, NEXT, dry_run=True)

    assert result == (2, 0)
    assert _validators(rows) == {"10.0.0.1": CURRENT, "10.0.0.2": CURRENT, "10.0.0.3": STRANGER}


def test_the_migration_moves_exactly_the_rows_keyed_to_the_old_hotkey(rows):
    # regression: the command keys on a hard-coded previous-rotation address and moves 0 rows (the
    # miner's nodes stay listed under a validator that no longer asks), or re-keys every row
    result = migrate_validator_hotkey_rows(ExecutorDao(session=rows), CURRENT, NEXT, dry_run=False)

    assert result == (2, 2)
    assert _validators(rows) == {"10.0.0.1": NEXT, "10.0.0.2": NEXT, "10.0.0.3": STRANGER}


def test_the_migration_refuses_equal_or_blank_hotkeys_before_touching_the_rows(rows):
    with pytest.raises(ValueError, match="the same"):
        migrate_validator_hotkey_rows(ExecutorDao(session=rows), NEXT, NEXT, dry_run=False)
    with pytest.raises(ValueError, match="required"):
        migrate_validator_hotkey_rows(ExecutorDao(session=rows), CURRENT, " ", dry_run=False)
    assert _validators(rows) == {"10.0.0.1": CURRENT, "10.0.0.2": CURRENT, "10.0.0.3": STRANGER}
