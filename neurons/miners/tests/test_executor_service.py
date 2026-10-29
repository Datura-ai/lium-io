"""
Tests for ExecutorService SSH key payload construction.

Focus: verify that validator_signature is correctly forwarded in the POST
body sent to the executor when registering or removing SSH public keys.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import bittensor
import aiohttp
import pytest

from models.executor import Executor
from protocol.miner_portal_request import (
    DeleteExecutorRequest,
    ExecutorDeleted,
    SyncExecutorPayload,
)
import services.executor_service as es
from services.executor_service import ExecutorService


_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestMiner")


@pytest.fixture
def executor_service(miner_keypair, monkeypatch):
    """ExecutorService with mocked DAO and wallet — no real DB or wallet needed."""
    service = ExecutorService.__new__(ExecutorService)
    service.executor_dao = MagicMock()
    service.ssh_service = MagicMock()

    # Replace settings in executor_service with a mock (Pydantic Settings does not allow
    # setattr of non-fields like get_bittensor_wallet).
    import services.executor_service as es_module

    mock_wallet = MagicMock()
    mock_wallet.get_hotkey.return_value = miner_keypair
    mock_settings = MagicMock()
    mock_settings.get_bittensor_wallet.return_value = mock_wallet
    mock_settings.CENTRAL_MODE = False
    monkeypatch.setattr(es_module, "settings", mock_settings)

    return service


@pytest.fixture
def test_executor():
    return Executor(
        uuid=uuid4(),
        validator="//TestValidator",
        address="127.0.0.1",
        port=8001,
    )


def _make_mock_session(response_status: int = 500):
    """Return a mock aiohttp.ClientSession that captures the POST payload."""
    mock_response = MagicMock()
    mock_response.status = response_status
    mock_response.text = AsyncMock(return_value="test error")

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.post = MagicMock(return_value=mock_post_ctx)

    return mock_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_pubkey_to_executor_includes_validator_signature(
    executor_service, test_executor, validator_keypair
):
    """validator_signature is forwarded verbatim in the POST body to the executor."""
    # Arrange
    validator_sig = "0x" + validator_keypair.sign(_SSH_KEY).hex()
    mock_session = _make_mock_session()

    # Act
    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, validator_sig)

    # Assert — the executor was called with a JSON body
    assert mock_session.post.called
    sent_payload = mock_session.post.call_args.kwargs["json"]

    # Assert — validator_signature is present and unchanged in the forwarded payload
    assert "validator_signature" in sent_payload
    assert sent_payload["validator_signature"] == validator_sig


@pytest.mark.asyncio
async def test_send_pubkey_to_executor_relays_the_miner_bound_signature(executor_service, test_executor):
    """miner_hotkey and validator_miner_signature reach the executor verbatim; absent when not sent."""
    mock_session = _make_mock_session()
    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", miner_hotkey="5Miner", validator_miner_signature="0xbb"
        )
        sent = mock_session.post.call_args.kwargs["json"]
        assert (sent["miner_hotkey"], sent["validator_miner_signature"]) == ("5Miner", "0xbb")

        await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa")
        sent = mock_session.post.call_args.kwargs["json"]
        assert "miner_hotkey" not in sent and "validator_miner_signature" not in sent


@pytest.mark.parametrize(
    "status, body_error, expected_posts, accepted",
    [
        (None, None, 2, True),  # the request timed out before any answer
        (200, aiohttp.ClientPayloadError("Response payload is not completed"), 2, True),
        (200, json.JSONDecodeError("invalid JSON", "{", 1), 1, False),
        (503, None, 1, False),
    ],
)
@pytest.mark.asyncio
async def test_send_pubkey_to_executor_retries_an_executor_that_did_not_answer(
    executor_service, test_executor, status, body_error, expected_posts, accepted
):
    """A timed-out or cut-off upload is asked again, so one dropped request does not leave the node out of
    the answer; a complete bad answer is not; a node that never answers gives up inside the deadline."""
    answered = MagicMock(status=200)
    answered.json = AsyncMock(return_value={"ssh_username": "root", "ssh_port": 22, "python_path": "/p", "root_dir": "/r"})
    if status is None:
        first = asyncio.TimeoutError()
    else:
        first = MagicMock(status=status)
        first.json = AsyncMock(side_effect=body_error)
        first.text = AsyncMock(return_value="unavailable")
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(side_effect=[first, answered])
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = _make_mock_session()
    session.post = MagicMock(return_value=post_ctx)

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service.asyncio.sleep", AsyncMock()):
        info = await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa")
        assert (info is not None and info.uuid == str(test_executor.uuid)) is accepted
        assert session.post.call_count == expected_posts

        post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        session.post.reset_mock()
        assert await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa") is None
        assert session.post.call_count == es.UPLOAD_SSH_KEY_ATTEMPTS

        # no time left for a second attempt: one try, then the miner answers without the node
        session.post.reset_mock()
        assert await executor_service.send_pubkey_to_executor(
            test_executor, _SSH_KEY, "0xaa", deadline=time.monotonic() + 1
        ) is None
        assert session.post.call_count == 1



@pytest.mark.parametrize(
    "is_rental_request, hook, lookup_seconds, expected_left",
    [
        (False, None, 0, 25), (False, None, 15, 10),
        (False, "http://hook", 15, 10), (True, None, 15, 10),
        (True, "http://hook", 0, 20), (True, "http://hook", 15, 5),
    ],
)
@pytest.mark.asyncio
async def test_register_pubkey_reserves_the_hook_time_only_when_the_hook_runs(
    executor_service, test_executor, is_rental_request, hook, lookup_seconds, expected_left
):
    """Ordinary checks never call the rental hook, so their uploads keep the whole wait minus the reply margin."""
    clock = [100.0]

    async def slow_lookup(*_args):
        clock[0] += lookup_seconds
        return [test_executor]

    es.settings.RENTAL_REQUEST_HOOK = hook
    executor_service.get_executors_for_validator = slow_lookup
    executor_service.send_pubkey_to_executor = AsyncMock(return_value=None)
    with patch("services.executor_service.time.monotonic", side_effect=lambda: clock[0]):
        await executor_service.register_pubkey("v", "m", _SSH_KEY.encode(), "0xaa", is_rental_request=is_rental_request)

    deadline = executor_service.send_pubkey_to_executor.call_args.kwargs["deadline"]
    assert deadline - clock[0] == expected_left
    # uploads plus a stalled hook still end inside the validator's 30 s wait, leaving its connect/auth and reply time
    hook_time = es.RENTAL_REQUEST_HOOK_TIMEOUT_SECONDS if is_rental_request and hook else 0
    overhead = es.UPLOAD_SSH_KEY_RESPONSE_MARGIN_SECONDS
    assert lookup_seconds + expected_left + hook_time + overhead <= es.VALIDATOR_SSH_SUBMIT_TIMEOUT_SECONDS


@pytest.mark.parametrize("is_rental_request", [False, True])
@pytest.mark.parametrize("hook", [None, "http://hook"])
@pytest.mark.parametrize("lookup_seconds", [0, 15, 30])
@pytest.mark.asyncio
async def test_register_pubkey_first_upload_keeps_mains_full_timeout(
    executor_service, test_executor, is_rental_request, hook, lookup_seconds
):
    """Retries must not cost a first-attempt success main accepted: attempt 1 always gets the full 10 s."""
    clock = [100.0]

    async def slow_lookup(*_args):
        clock[0] += lookup_seconds
        return [test_executor]

    answered = MagicMock(status=200)
    answered.json = AsyncMock(return_value={"ssh_username": "root", "ssh_port": 22, "python_path": "/p", "root_dir": "/r"})
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(return_value=answered)
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = _make_mock_session()
    session.post = MagicMock(return_value=post_ctx)
    es.settings.RENTAL_REQUEST_HOOK = hook
    executor_service.get_executors_for_validator = slow_lookup

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service.time.monotonic", side_effect=lambda: clock[0]):
        registration = await executor_service.register_pubkey(
            "v", "m", _SSH_KEY.encode(), "0xaa", is_rental_request=is_rental_request
        )

    assert len(registration.accepted) == 1
    assert session.post.call_args.kwargs["timeout"].total == es.UPLOAD_SSH_KEY_ATTEMPT_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_send_pubkey_does_not_retry_when_the_delay_overran_the_deadline(executor_service, test_executor):
    """A retry delay that ended past the deadline gives up instead of starting an attempt the validator cannot wait for."""
    clock = [100.0]

    async def late_sleep(_seconds):
        clock[0] += 30  # the event loop was busy: the 1 s retry delay ended well after the deadline

    answered = MagicMock(status=200)
    answered.json = AsyncMock(return_value={"ssh_username": "root", "ssh_port": 22, "python_path": "/p", "root_dir": "/r"})
    post_ctx = MagicMock()
    post_ctx.__aenter__ = AsyncMock(side_effect=[asyncio.TimeoutError(), answered])
    post_ctx.__aexit__ = AsyncMock(return_value=None)
    session = _make_mock_session()
    session.post = MagicMock(return_value=post_ctx)

    with patch("services.executor_service.aiohttp.ClientSession", return_value=session), \
            patch("services.executor_service.asyncio.sleep", late_sleep), \
            patch("services.executor_service.time.monotonic", side_effect=lambda: clock[0]):
        info = await executor_service.send_pubkey_to_executor(test_executor, _SSH_KEY, "0xaa", deadline=clock[0] + 20)

    assert info is None
    assert session.post.call_count == 1


# ---------------------------------------------------------------------------
# A failed write rolls the shared session back
# ---------------------------------------------------------------------------


def _payload(test_executor):
    return SyncExecutorPayload(
        uuid=test_executor.uuid,
        validator=test_executor.validator,
        address=test_executor.address,
        port=test_executor.port,
        price_per_gpu=0.5,
    )


# ---------------------------------------------------------------------------
# A portal delete removes only the node it names (ticket-0591)
# ---------------------------------------------------------------------------


def test_delete_removes_the_named_node_not_another_at_its_address(executor_service, test_executor):
    dao = executor_service.executor_dao
    dao.delete_by_uuid.return_value = False

    payload = _payload(test_executor)

    result = executor_service.delete(DeleteExecutorRequest(executor=payload))

    assert isinstance(result, ExecutorDeleted)
    dao.delete_by_uuid.assert_called_once_with(payload.uuid)
    dao.delete_by_address_port.assert_not_called()
