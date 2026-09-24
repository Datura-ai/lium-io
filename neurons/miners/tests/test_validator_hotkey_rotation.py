"""The validator hotkey rotation on the miner: both Lium validator hotkeys sign in, only the current one is
active (new executors are listed under it), the swap-day flip is the config pair swap that keeps both
accepted, the drop is a blank VALIDATOR_NEXT_HOTKEY, and the executor rows move with one CLI command.

Settings are built with a clean environment per test so a developer's .env cannot change the answer.
"""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import core.config as core_config
from daos.executor import ExecutorDao
from models.executor import Executor
from services.validator_service import ValidatorService, migrate_validator_hotkey_rows

CURRENT = "5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p"
NEXT = "5DZhu7LLGGc7qRa8ZPFArt7KV2XEKMTr5Q7ZuM9LNdTaoNfK"
STRANGER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@pytest.fixture()
def settings(monkeypatch):
    for name in ("DEFAULT_VALIDATOR_HOTKEY", "VALIDATOR_NEXT_HOTKEY", "DEBUG_SKIP_VALIDATOR_REGISTRATION_CHECK"):
        monkeypatch.delenv(name, raising=False)
    built = core_config.Settings(_env_file=None)
    monkeypatch.setattr("services.validator_service.settings", built)
    return built


def _service() -> ValidatorService:
    return ValidatorService(validator_dao=None)


def test_the_defaults_are_the_two_lium_hotkeys_with_current_active(settings):
    # regression: the new address lands in DEFAULT_VALIDATOR_HOTKEY (new executors are listed under a
    # validator that does not serve yet; the live validator's sign-in is refused) instead of NEXT
    assert settings.DEFAULT_VALIDATOR_HOTKEY == CURRENT
    assert settings.VALIDATOR_NEXT_HOTKEY == NEXT
    assert settings.accepted_validator_hotkeys == frozenset({CURRENT, NEXT})


def test_the_current_hotkey_signs_in(settings):
    assert _service().is_valid_validator(CURRENT) is True


def test_the_next_hotkey_signs_in_before_the_swap(settings):
    # regression: is_valid_validator stays an equality check on DEFAULT_VALIDATOR_HOTKEY, so the
    # chain swap is refused by every miner until each one is reconfigured
    assert _service().is_valid_validator(NEXT) is True


def test_a_third_hotkey_is_refused_with_two_accepted(settings):
    # regression: the check becomes "anything non-empty" or a substring match once it is a set
    assert _service().is_valid_validator(STRANGER) is False
    assert _service().is_valid_validator("") is False


def _built(monkeypatch, default: str, nxt: str) -> core_config.Settings:
    monkeypatch.setenv("DEFAULT_VALIDATOR_HOTKEY", default)
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY", nxt)
    monkeypatch.delenv("DEBUG_SKIP_VALIDATOR_REGISTRATION_CHECK", raising=False)
    return core_config.Settings(_env_file=None)


def test_after_the_pair_swap_flip_both_hotkeys_stay_accepted_and_the_new_one_is_active(monkeypatch):
    # the swap-day flip: DEFAULT_VALIDATOR_HOTKEY=<new>, VALIDATOR_NEXT_HOTKEY=<old> — the validator now
    # signs with the new hotkey and new executors are listed under it, while the old one still signs in
    # (its last cycle, the rollback path). Regression: the flip drops the old hotkey the same hour.
    built = _built(monkeypatch, default=NEXT, nxt=CURRENT)

    assert built.DEFAULT_VALIDATOR_HOTKEY == NEXT
    assert built.accepted_validator_hotkeys == frozenset({CURRENT, NEXT})


def test_flipping_default_alone_drops_the_old_hotkey_at_once(monkeypatch):
    # the hazard the pair swap avoids: DEFAULT_VALIDATOR_HOTKEY=<new> with VALIDATOR_NEXT_HOTKEY still <new>
    # leaves one accepted hotkey, so a validator still signing with the old one is refused immediately
    built = _built(monkeypatch, default=NEXT, nxt=NEXT)

    assert built.accepted_validator_hotkeys == frozenset({NEXT})


def test_the_old_hotkey_is_dropped_by_blanking_next_after_the_flip(monkeypatch):
    # the follow-up release once the old hotkey has left the chain — VALIDATOR_NEXT_HOTKEY="" — the new
    # hotkey stays active, the old one no longer signs in
    built = _built(monkeypatch, default=NEXT, nxt="")

    assert built.DEFAULT_VALIDATOR_HOTKEY == NEXT
    assert built.accepted_validator_hotkeys == frozenset({NEXT})


def test_a_blank_next_hotkey_adds_no_signer(monkeypatch):
    # rolling back (VALIDATOR_NEXT_HOTKEY="") leaves the miner exactly as before this release
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY", " ")
    monkeypatch.delenv("DEFAULT_VALIDATOR_HOTKEY", raising=False)
    built = core_config.Settings(_env_file=None)

    assert built.accepted_validator_hotkeys == frozenset({CURRENT})


def test_a_miner_serving_another_validator_does_not_accept_the_lium_next_hotkey(monkeypatch):
    # regression: a staging or e2e miner that sets only DEFAULT_VALIDATOR_HOTKEY also accepts the prod
    # validator's new hotkey (the executor and watchtower staging builds trust their own validator only)
    monkeypatch.delenv("VALIDATOR_NEXT_HOTKEY", raising=False)
    monkeypatch.setenv("DEFAULT_VALIDATOR_HOTKEY", STRANGER)
    built = core_config.Settings(_env_file=None)

    assert built.accepted_validator_hotkeys == frozenset({STRANGER})


# --- cli.py migrate-validator-hotkey: the executor rows move with the flip ------------------------------------


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
