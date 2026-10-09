"""A settings validation error at startup names the failing field and never repeats the input value."""

import logging

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



@pytest.mark.parametrize(
    ("secret", "warned"),
    [
        ("x" * 32, True),
        ("password" * 4, True),
        ("abc" * 11, True),
        ("4b1d9e07c2a85f36e0d7b9a41c6f2e8d", False),
        # weak, yet not provably so: a phrase or a 16-byte unit draws no warning
        ("test-master-secret-32-chars-long!!", False),
        ("0123456789abcdef" * 2, False),
        # 8 characters of 3 UTF-8 bytes each: a 24-byte unit, not a short one
        ("\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b" * 4, False),
    ],
)
def test_a_volume_secret_that_repeats_a_unit_of_at_most_8_bytes_starts_with_a_warning_that_omits_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, secret: str, warned: bool
):
    monkeypatch.setenv("ENABLE_VOLUME_ENCRYPTION", "true")
    monkeypatch.setenv("VOLUME_MASTER_SECRET", secret)

    with caplog.at_level(logging.WARNING, logger="core.config"):
        settings = Settings(_env_file=None)

    warnings = [record.getMessage() for record in caplog.records if "VOLUME_MASTER_SECRET" in record.getMessage()]
    assert settings.VOLUME_MASTER_SECRET == secret
    assert len(warnings) == (1 if warned else 0)
    assert all(secret not in warning for warning in warnings)
