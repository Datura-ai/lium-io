"""A settings validation error at startup names the failing field and never repeats the input value."""

import pytest
from pydantic import ValidationError

from core.config import Settings

MARKER = "q7x-settings-marker-4f1c"


def test_type_error_names_the_field_and_omits_the_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INTERNAL_PORT", f"not-a-port-{MARKER}")

    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    text = str(exc.value)

    assert "INTERNAL_PORT" in text
    assert MARKER not in text
    assert "input_value" not in text
