"""A settings validation error at startup names the failing field and never repeats the input value."""

import pytest
from pydantic import ValidationError

from core.config import Settings

MARKER = "q7x-settings-marker-4f1c"


def _startup_error(monkeypatch: pytest.MonkeyPatch, **env: str) -> str:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    return str(exc.value)


def test_type_error_names_the_field_and_omits_the_value(monkeypatch: pytest.MonkeyPatch):
    text = _startup_error(monkeypatch, BITTENSOR_NETUID=f"not-a-number-{MARKER}")

    assert "BITTENSOR_NETUID" in text
    assert MARKER not in text
    assert "input_value" not in text


def test_custom_validator_error_names_the_field_and_omits_the_value(monkeypatch: pytest.MonkeyPatch):
    text = _startup_error(
        monkeypatch,
        ENABLE_VOLUME_ENCRYPTION="true",
        VOLUME_MASTER_SECRET=f"x{MARKER}",
    )

    assert "VOLUME_MASTER_SECRET" in text
    assert MARKER not in text
    assert "input_value" not in text


@pytest.mark.parametrize("secret", ["x" * 32, "password" * 4, "abc" * 11])
def test_a_volume_secret_that_repeats_a_short_unit_is_refused_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch, secret: str
):
    text = _startup_error(monkeypatch, ENABLE_VOLUME_ENCRYPTION="true", VOLUME_MASTER_SECRET=secret)

    assert "VOLUME_MASTER_SECRET" in text
    assert "repeats" in text
    assert secret not in text


@pytest.mark.parametrize(
    "secret",
    [
        "4b1d9e07c2a85f36e0d7b9a41c6f2e8d",
        # weak, yet not provably so: a phrase or a 16-character unit stays accepted, as before
        "test-master-secret-32-chars-long!!",
        "0123456789abcdef" * 2,
    ],
)
def test_a_volume_secret_without_a_short_repeated_unit_is_accepted(monkeypatch: pytest.MonkeyPatch, secret: str):
    monkeypatch.setenv("ENABLE_VOLUME_ENCRYPTION", "true")
    monkeypatch.setenv("VOLUME_MASTER_SECRET", secret)

    assert Settings(_env_file=None).VOLUME_MASTER_SECRET == secret
