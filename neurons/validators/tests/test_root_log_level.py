"""LOG_LEVEL sets the validator's root log level; an unknown name falls back to INFO."""

import logging

import pytest

from core.config import settings
from core.utils import root_log_level


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("INFO", logging.INFO),
        ("debug", logging.DEBUG),
        (" Warning ", logging.WARNING),
        ("nonsense", logging.INFO),
    ],
)
def test_root_log_level_reads_the_setting(monkeypatch, value, expected):
    monkeypatch.setattr(settings, "LOG_LEVEL", value)

    assert root_log_level() == expected
