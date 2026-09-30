"""LOG_LEVEL sets the validator's root log level; an unknown name falls back to INFO."""

import logging

import pytest

from core.config import settings
from core import utils
from core.utils import PROTOCOL_LOGGERS, configure_logs_of_other_modules, get_logger, root_log_level


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("INFO", logging.INFO),
        ("debug", logging.DEBUG),
        (" Warning ", logging.WARNING),
        ("nonsense", logging.INFO),
        ("DEBGU", logging.INFO),
        ("10", logging.INFO),
    ],
)
def test_root_log_level_reads_the_setting(monkeypatch, caplog, value, expected):
    monkeypatch.setattr(settings, "LOG_LEVEL", value)
    monkeypatch.setattr(utils, "_warned_log_levels", set())

    with caplog.at_level(logging.WARNING, logger="core.utils"):
        assert root_log_level() == expected
        root_log_level()

    warnings = [r.getMessage() for r in caplog.records if r.name == "core.utils"]
    assert warnings == ([] if expected != logging.INFO or value == "INFO" else [f"LOG_LEVEL={value!r} is not a level name; logging at INFO"])


@pytest.fixture
def restore_logging():
    names = ["", "connector", "asyncssh", "sqlalchemy", *PROTOCOL_LOGGERS]
    saved = {name: (logging.getLogger(name).level, logging.getLogger(name).handlers[:]) for name in names}
    yield
    for name, (level, handlers) in saved.items():
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.handlers[:] = handlers


@pytest.mark.parametrize("value", ["DEBUG", "INFO", "WARNING"])
def test_log_level_debug_never_reaches_the_protocol_loggers(monkeypatch, restore_logging, value):
    # a websocket frame to a miner can carry an SSH key or a registry password
    monkeypatch.setattr(settings, "LOG_LEVEL", value)
    for setup in (lambda: get_logger("test.protocol"), configure_logs_of_other_modules):
        setup()
        for name in PROTOCOL_LOGGERS:
            level = logging.getLogger(name).getEffectiveLevel()
            assert level >= logging.INFO and level >= root_log_level(), (name, level)
    assert logging.getLogger("test.protocol").getEffectiveLevel() == root_log_level()
