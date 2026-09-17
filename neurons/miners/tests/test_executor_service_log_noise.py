"""DAH-3593 — an executor that does not answer its SSH-key call is one WARNING per window.

`send_pubkey_to_executor` wrote 27,894 ERROR lines in two days, every one with `"error": ""`
(aiohttp's TimeoutError has no text), and `remove_pubkey_from_executor` logged its 3,066 failures
as "failed to register". The line now names the exception class, says register or remove, and
repeats per executor only once per window; a real HTTP error from a live executor is still ERROR.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import services.executor_service as es_module
from models.executor import Executor
from services.executor_service import _UnreachableExecutorLog
from tests.test_executor_service import (
    _SSH_KEY,
    _make_mock_session,
    executor_service,
    miner_keypair,
)

# pytest resolves fixtures by name from the module namespace; the tuple keeps the import "used"
_SHARED_FIXTURES = (executor_service, miner_keypair)


@pytest.fixture
def executor():
    return Executor(uuid=uuid4(), validator="//TestValidator", address="203.0.113.21", port=8001)


def _records(caplog, message):
    # the miner's structured message renders "text >>> {json}", so match the text part
    return [r for r in caplog.records if getattr(r.msg, "message", r.getMessage()) == message]


def _extra(record):
    return getattr(record.msg, "extra", {})


def _session_raising(exc: BaseException):
    session = _make_mock_session()
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(side_effect=exc)
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(return_value=post_ctx)
    return session


@pytest.fixture
def fresh_window(monkeypatch):
    log = _UnreachableExecutorLog(window_seconds=300.0)
    monkeypatch.setattr(es_module, "_unreachable_executor_log", log)
    return log


@pytest.mark.asyncio
async def test_register_timeout_is_one_warning_naming_the_exception_class(
    executor_service, executor, caplog, fresh_window
):
    caplog.set_level(logging.DEBUG, logger="services.executor_service")
    session = _session_raising(TimeoutError())

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        first = await executor_service.send_pubkey_to_executor(executor, _SSH_KEY, "0xsig")
        second = await executor_service.send_pubkey_to_executor(executor, _SSH_KEY, "0xsig")
        third = await executor_service.send_pubkey_to_executor(executor, _SSH_KEY, "0xsig")

    assert first is None and second is None and third is None
    lines = _records(caplog, "API request failed to register SSH key - executor did not answer")
    assert [r.levelno for r in lines] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert _extra(lines[0])["error"] == "TimeoutError"  # was "" before
    assert _extra(lines[0])["reason"] == "executor_unreachable"
    assert _extra(lines[0])["folded_since_last_line"] == 0
    assert logging.ERROR not in {r.levelno for r in caplog.records}


@pytest.mark.asyncio
async def test_next_window_warns_again_with_the_folded_count(
    executor_service, executor, caplog, fresh_window
):
    caplog.set_level(logging.DEBUG, logger="services.executor_service")
    session = _session_raising(ConnectionRefusedError(111, "Connection refused"))

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        for _ in range(4):
            await executor_service.send_pubkey_to_executor(executor, _SSH_KEY, "0xsig")
        # the window passes
        fresh_window._last_warned_at[str(executor.uuid)] -= 301.0
        await executor_service.send_pubkey_to_executor(executor, _SSH_KEY, "0xsig")

    warnings = [
        r
        for r in _records(caplog, "API request failed to register SSH key - executor did not answer")
        if r.levelno == logging.WARNING
    ]
    assert len(warnings) == 2
    assert _extra(warnings[1])["folded_since_last_line"] == 3
    assert _extra(warnings[1])["error"].startswith("ConnectionRefusedError: ")


@pytest.mark.asyncio
async def test_remove_path_says_remove(executor_service, executor, caplog, fresh_window):
    caplog.set_level(logging.DEBUG, logger="services.executor_service")
    session = _session_raising(TimeoutError())

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session):
        await executor_service.remove_pubkey_from_executor(executor, _SSH_KEY, "0xsig")

    lines = _records(caplog, "API request failed to remove SSH key - executor did not answer")
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert not [r for r in caplog.records if "register" in getattr(r.msg, "message", r.getMessage()) and r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_live_executor_refusing_the_key_is_still_an_error(
    executor_service, executor, caplog, fresh_window
):
    caplog.set_level(logging.DEBUG, logger="services.executor_service")

    with patch(
        "services.executor_service.aiohttp.ClientSession", return_value=_make_mock_session(response_status=401)
    ):
        assert await executor_service.send_pubkey_to_executor(executor, _SSH_KEY, "0xsig") is None
        await executor_service.remove_pubkey_from_executor(executor, _SSH_KEY, "0xsig")

    register = _records(caplog, "API request failed to register SSH key - HTTP error")
    remove = _records(caplog, "API request failed to remove SSH key - HTTP error")
    assert [r.levelno for r in register] == [logging.ERROR]
    assert [r.levelno for r in remove] == [logging.ERROR]
    assert _extra(register[0])["status"] == 401
