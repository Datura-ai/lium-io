"""The validator hotkey rotation on the miner: both Lium validator hotkeys sign in, only the current one is
active (new executors are listed under it), and the swap is a config value, not a release.

Settings are built with a clean environment per test so a developer's .env cannot change the answer.
"""

import pytest

import core.config as core_config
from services.validator_service import ValidatorService

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


def test_the_swap_is_one_config_value(monkeypatch):
    # the flip named in the PR body: DEFAULT_VALIDATOR_HOTKEY=<NEXT>; both stay accepted, NEXT is now
    # the one new executors are listed under
    monkeypatch.setenv("DEFAULT_VALIDATOR_HOTKEY", NEXT)
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY", NEXT)
    monkeypatch.delenv("DEBUG_SKIP_VALIDATOR_REGISTRATION_CHECK", raising=False)
    built = core_config.Settings(_env_file=None)

    assert built.DEFAULT_VALIDATOR_HOTKEY == NEXT
    assert built.accepted_validator_hotkeys == frozenset({NEXT})


def test_a_blank_next_hotkey_adds_no_signer(monkeypatch):
    # rollback: VALIDATOR_NEXT_HOTKEY="" leaves the miner exactly as before this release
    monkeypatch.setenv("VALIDATOR_NEXT_HOTKEY", " ")
    monkeypatch.delenv("DEFAULT_VALIDATOR_HOTKEY", raising=False)
    built = core_config.Settings(_env_file=None)

    assert built.accepted_validator_hotkeys == frozenset({CURRENT})
