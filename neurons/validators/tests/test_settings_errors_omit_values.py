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
