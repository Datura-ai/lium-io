"""The latest-set-weights report: never posted without a chain block, and a refusal is logged, not dropped silently."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients import subtensor_client
from clients.subtensor_client import SubtensorClient

PAYLOAD = {
    "netuid": 51,
    "uids": [0, 1],
    "weights": [1, 2],
    "version_key": 10000,
    "wait_for_finalization": False,
    "wait_for_inclusion": False,
    "current_block": 123456,
    "updated_at": "2026-10-12T00:00:00Z",
}


def _session(status: int, text: str = "") -> MagicMock:
    response = MagicMock(status=status, text=AsyncMock(return_value=text))
    post = MagicMock()
    post.return_value.__aenter__ = AsyncMock(return_value=response)
    post.return_value.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock(post=post)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def session(monkeypatch):
    wallet = MagicMock()
    wallet.get_hotkey.return_value = MagicMock(
        ss58_address="validator-hotkey", sign=lambda blob: b"\x01"
    )
    settings = SimpleNamespace(
        get_bittensor_wallet=lambda: wallet, COMPUTE_REST_API_URL="http://lium.test"
    )
    monkeypatch.setattr(subtensor_client, "settings", settings)
    sessions = []

    def make(*args, **kwargs):
        sessions.append(_session(422, '{"detail": "refused"}'))
        return sessions[-1]

    monkeypatch.setattr(subtensor_client.aiohttp, "ClientSession", make)
    return sessions


@pytest.mark.asyncio
async def test_a_report_without_a_current_block_is_not_posted(session):
    client = SubtensorClient.__new__(SubtensorClient)

    await client.send_weights_to_lium({**PAYLOAD, "current_block": None})

    assert session == []


@pytest.mark.asyncio
async def test_a_refused_report_is_logged_with_its_status(session, monkeypatch):
    errors = []
    monkeypatch.setattr(
        subtensor_client.logger, "error", lambda message, *args, **kwargs: errors.append(message)
    )
    client = SubtensorClient.__new__(SubtensorClient)

    await client.send_weights_to_lium(PAYLOAD)

    assert session[0].post.call_args.kwargs["json"] == PAYLOAD
    assert len(errors) == 1
    assert "refused" in str(errors[0])
