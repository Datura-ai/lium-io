"""
Tests for ExecutorService SSH key payload construction.

Focus: verify that validator_signature is correctly forwarded in the POST
body sent to the executor when registering or removing SSH public keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import bittensor
import pytest

from models.executor import Executor
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
async def test_remove_pubkey_from_executor_includes_validator_signature(
    executor_service, test_executor, validator_keypair
):
    """validator_signature is forwarded verbatim in the POST body on SSH key removal."""
    # Arrange
    validator_sig = "0x" + validator_keypair.sign(_SSH_KEY).hex()
    mock_session = _make_mock_session()

    # Act
    with patch("services.executor_service.aiohttp.ClientSession", return_value=mock_session):
        await executor_service.remove_pubkey_from_executor(test_executor, _SSH_KEY, validator_sig)

    # Assert — the executor was called with a JSON body
    assert mock_session.post.called
    sent_payload = mock_session.post.call_args.kwargs["json"]

    # Assert — validator_signature is present and unchanged in the forwarded payload
    assert "validator_signature" in sent_payload
    assert sent_payload["validator_signature"] == validator_sig


# ---------------------------------------------------------------------------
# DAH-3338: register_pubkey tells the validator which executors it KNOWS apart from
# which ACCEPTED the key, so an unreachable node stops reading as an invalid id.
# ---------------------------------------------------------------------------


def _ssh_info(executor: Executor):
    from datura.requests.miner_requests import ExecutorSSHInfo

    return ExecutorSSHInfo(
        uuid=str(executor.uuid),
        address=executor.address,
        port=executor.port,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


@pytest.mark.asyncio
async def test_register_pubkey_splits_known_from_accepted(executor_service, validator_keypair):
    reachable = Executor(uuid=uuid4(), validator="//TestValidator", address="203.0.113.10", port=8001)
    dead = Executor(uuid=uuid4(), validator="//TestValidator", address="203.0.113.11", port=8001)
    executor_service.executor_dao.get_executors_for_validator.return_value = [reachable, dead]

    async def only_the_reachable_one_answers(executor, pubkey, signature, nonce=None):
        return _ssh_info(executor) if executor is reachable else None

    executor_service.send_pubkey_to_executor = only_the_reachable_one_answers

    registration = await executor_service.register_pubkey(
        validator_keypair.ss58_address, "miner-hotkey", _SSH_KEY.encode(), "0xsig"
    )

    assert registration.known_executor_ids == [str(reachable.uuid), str(dead.uuid)]
    assert [info.uuid for info in registration.accepted] == [str(reachable.uuid)]


@pytest.mark.asyncio
async def test_register_pubkey_knows_nothing_for_an_id_the_miner_does_not_list(
    executor_service, validator_keypair
):
    executor_service.executor_dao.get_executors_for_validator.return_value = []

    registration = await executor_service.register_pubkey(
        validator_keypair.ss58_address, "miner-hotkey", _SSH_KEY.encode(), "0xsig", str(uuid4())
    )

    assert registration.known_executor_ids == []
    assert registration.accepted == []
