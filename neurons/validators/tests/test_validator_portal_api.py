"""Tests for the ValidatorPortalAPI last-good cache (DAH-2518).

A portal outage used to return an empty opted-in list, which `fetch_miners` could not tell
apart from a real mass opt-out: the opt-in miners then lost their central-miner axon and
dropped out of the validated fleet. The cache must serve the last successful answer on
failure, report None only when nothing was ever cached, and keep a populated list when the
portal answers 200 with an empty body.
"""

from unittest.mock import AsyncMock, Mock

import aiohttp
import bittensor
import pytest
from pydantic import ValidationError

from clients.subtensor_client import SubtensorClient
from clients.validator_portal_api import OptedInMiner, ValidatorPortalAPI
from core.config import settings

OPTED_IN_RECORDS = [
    {
        "miner_hotkey": "hotkey-a",
        "miner_coldkey": "cold-a",
        "central_miner_ip": "10.0.0.1",
        "central_miner_port": 8000,
    },
    {
        "miner_hotkey": "hotkey-b",
        "miner_coldkey": "cold-b",
        "central_miner_ip": "10.0.0.1",
        "central_miner_port": 8000,
    },
]
OPTED_IN_MINERS = [OptedInMiner.model_validate(record) for record in OPTED_IN_RECORDS]


@pytest.fixture(autouse=True)
def reset_portal_cache(monkeypatch):
    ValidatorPortalAPI._last_good_opted_in_miners = None
    ValidatorPortalAPI._last_good_fetched_at = None
    monkeypatch.setattr(settings, "MINER_PORTAL_REST_API_URL", "https://portal.test/")
    yield


def _install_fake_fetch(monkeypatch, *results):
    # each call returns the next result; an Exception instance is raised instead
    fetch = AsyncMock(side_effect=list(results))
    monkeypatch.setattr(ValidatorPortalAPI, "_fetch_opted_in_miners", fetch)
    return fetch


@pytest.mark.asyncio
async def test_successful_fetch_is_cached(monkeypatch):
    _install_fake_fetch(monkeypatch, OPTED_IN_MINERS)

    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == OPTED_IN_MINERS
    assert ValidatorPortalAPI._last_good_opted_in_miners == OPTED_IN_MINERS


@pytest.mark.asyncio
async def test_failed_fetch_serves_last_good(monkeypatch):
    _install_fake_fetch(monkeypatch, OPTED_IN_MINERS, TimeoutError("portal timed out"))

    await ValidatorPortalAPI.get_opted_in_miners()
    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == OPTED_IN_MINERS


@pytest.mark.asyncio
async def test_failed_fetch_without_cache_returns_none(monkeypatch):
    _install_fake_fetch(monkeypatch, TimeoutError("portal timed out"))

    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners is None


@pytest.mark.asyncio
async def test_empty_response_keeps_populated_cache(monkeypatch):
    _install_fake_fetch(monkeypatch, OPTED_IN_MINERS, [])

    await ValidatorPortalAPI.get_opted_in_miners()
    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == OPTED_IN_MINERS


@pytest.mark.asyncio
async def test_empty_response_without_cache_is_accepted(monkeypatch):
    _install_fake_fetch(monkeypatch, [])

    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == []


@pytest.mark.asyncio
async def test_blank_portal_url_returns_empty_list(monkeypatch):
    monkeypatch.setattr(settings, "MINER_PORTAL_REST_API_URL", "")
    fetch = _install_fake_fetch(monkeypatch, OPTED_IN_MINERS)

    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == []
    assert fetch.await_count == 0


class _FakeResponse:
    def __init__(self, status: int, json_body=None, text_body: str = ""):
        self.status = status
        self._json_body = json_body
        self._text_body = text_body

    async def json(self):
        return self._json_body

    async def text(self) -> str:
        return self._text_body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *args) -> bool:
        return False


class _FakeSession:
    last_url: str | None = None
    last_headers: dict[str, str] | None = None

    def __init__(self, response: _FakeResponse):
        self._response = response

    def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        _FakeSession.last_url = url
        _FakeSession.last_headers = headers
        return self._response

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args) -> bool:
        return False


@pytest.fixture
def install_portal_response(monkeypatch):
    # replaces the wallet and aiohttp session so the real fetch runs against a fake portal
    keypair = bittensor.Keypair.create_from_uri("//LiumTestValidator")
    fake_wallet = Mock()
    fake_wallet.get_hotkey.return_value = keypair
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: fake_wallet)

    def install(response: _FakeResponse) -> None:
        monkeypatch.setattr(aiohttp, "ClientSession", lambda timeout: _FakeSession(response))

    return install


@pytest.mark.asyncio
async def test_fetch_raises_on_http_error(install_portal_response):
    install_portal_response(_FakeResponse(status=502, text_body="Bad gateway"))

    with pytest.raises(RuntimeError, match="502"):
        await ValidatorPortalAPI._fetch_opted_in_miners("https://portal.test/validators/opted-in")


@pytest.mark.asyncio
async def test_fetch_raises_on_non_list_body(install_portal_response):
    install_portal_response(
        _FakeResponse(status=200, json_body={"success": True, "data": OPTED_IN_RECORDS})
    )

    with pytest.raises(RuntimeError, match="expected a list"):
        await ValidatorPortalAPI._fetch_opted_in_miners("https://portal.test/validators/opted-in")


@pytest.mark.asyncio
async def test_fetch_raises_on_record_missing_a_field(install_portal_response):
    install_portal_response(_FakeResponse(status=200, json_body=[{"miner_hotkey": "hotkey-a"}]))

    with pytest.raises(ValidationError):
        await ValidatorPortalAPI._fetch_opted_in_miners("https://portal.test/validators/opted-in")


@pytest.mark.asyncio
async def test_fetch_sends_signature_headers_and_returns_parsed_miners(install_portal_response):
    install_portal_response(_FakeResponse(status=200, json_body=OPTED_IN_RECORDS))

    miners = await ValidatorPortalAPI._fetch_opted_in_miners(
        "https://portal.test/validators/opted-in"
    )

    assert miners == OPTED_IN_MINERS
    assert _FakeSession.last_url.endswith("/validators/opted-in")
    assert set(_FakeSession.last_headers) == {"hotkey", "timestamp", "signature"}


def _neuron(hotkey: str, uid: int, is_serving: bool) -> Mock:
    return Mock(
        hotkey=hotkey,
        uid=uid,
        axon_info=Mock(ip="0.0.0.0", port=0, is_serving=is_serving),
    )


def _subtensor_client(monkeypatch, previous_miners, metagraph_neurons) -> SubtensorClient:
    client = SubtensorClient.__new__(SubtensorClient)
    client.default_extra = {}
    client.debug_miner = None
    client.miners = previous_miners
    monkeypatch.setattr(client, "get_metagraph", lambda: Mock(neurons=metagraph_neurons))
    return client


@pytest.mark.asyncio
async def test_fetch_miners_keeps_previous_list_when_portal_has_no_cache(monkeypatch):
    monkeypatch.setattr(ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=None))
    previous = [_neuron("hotkey-a", 1, True), _neuron("hotkey-b", 2, True)]
    client = _subtensor_client(monkeypatch, previous, [_neuron("hotkey-c", 3, True)])

    await client.fetch_miners()

    assert client.miners == previous


@pytest.mark.asyncio
async def test_fetch_miners_falls_back_to_metagraph_when_no_previous_list(monkeypatch):
    monkeypatch.setattr(ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=None))
    serving = _neuron("hotkey-c", 3, True)
    client = _subtensor_client(monkeypatch, [], [serving, _neuron("hotkey-a", 1, False)])

    await client.fetch_miners()

    assert client.miners == [serving]


@pytest.mark.asyncio
async def test_fetch_miners_overrides_axon_with_central_miner_address(monkeypatch):
    monkeypatch.setattr(
        ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=OPTED_IN_MINERS)
    )
    opted_in = _neuron("hotkey-a", 1, True)
    plain = _neuron("hotkey-c", 3, True)
    client = _subtensor_client(monkeypatch, [], [opted_in, plain])

    await client.fetch_miners()

    assert (opted_in.axon_info.ip, opted_in.axon_info.port) == ("10.0.0.1", 8000)
    assert (plain.axon_info.ip, plain.axon_info.port) == ("0.0.0.0", 0)
