"""Tests for the ValidatorPortalAPI last-good list and its use in `fetch_miners` (DAH-2518).

A portal outage used to return an empty opted-in list, which `fetch_miners` could not tell
apart from a real mass opt-out: the opt-in miners then lost their central-miner axon and
dropped out of the validated fleet. The stored answer must serve the last successful list on
failure, report None whenever no populated list was ever stored, and survive a 200 that
carries an empty list.
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
    # miner_coldkey is sent by the portal and deliberately not modelled by the validator
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
VALIDATOR_KEYPAIR = bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(autouse=True)
def reset_portal_cache(monkeypatch):
    # monkeypatch, not plain assignment: the last-good list lives on the class, so a test
    # that populates it would otherwise leak into every later test in the session
    monkeypatch.setattr(ValidatorPortalAPI, "_last_good_opted_in_miners", None)
    monkeypatch.setattr(ValidatorPortalAPI, "_last_good_fetched_at", None)
    monkeypatch.setattr(settings, "MINER_PORTAL_REST_API_URL", "https://portal.test/")


def _install_fake_fetch(monkeypatch, *results: list[OptedInMiner] | Exception) -> AsyncMock:
    # AsyncMock raises a side effect that is an Exception and returns anything else
    fetch = AsyncMock(side_effect=list(results))
    monkeypatch.setattr(ValidatorPortalAPI, "_fetch_opted_in_miners", fetch)
    return fetch


@pytest.mark.asyncio
async def test_successful_fetch_returns_the_parsed_list(monkeypatch):
    _install_fake_fetch(monkeypatch, OPTED_IN_MINERS)

    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == OPTED_IN_MINERS


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
async def test_recovered_portal_replaces_the_cached_list(monkeypatch):
    moved_miner = [
        OptedInMiner.model_validate({**OPTED_IN_RECORDS[0], "central_miner_ip": "10.0.0.9"})
    ]
    fetch = _install_fake_fetch(
        monkeypatch, OPTED_IN_MINERS, TimeoutError("portal timed out"), moved_miner
    )

    await ValidatorPortalAPI.get_opted_in_miners()
    await ValidatorPortalAPI.get_opted_in_miners()
    miners = await ValidatorPortalAPI.get_opted_in_miners()

    assert miners == moved_miner
    assert fetch.await_count == 3


@pytest.mark.asyncio
async def test_empty_response_is_not_kept_as_last_good(monkeypatch):
    _install_fake_fetch(monkeypatch, [], TimeoutError("portal timed out"))

    await ValidatorPortalAPI.get_opted_in_miners()
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
    def __init__(self, status: int, json_body: list | dict | None = None, text_body: str = ""):
        self.status = status
        self._json_body = json_body
        self._text_body = text_body

    async def json(self) -> list | dict | None:
        return self._json_body

    async def text(self) -> str:
        return self._text_body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *args) -> bool:
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None

    def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        self.last_url = url
        self.last_headers = headers
        return self._response

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args) -> bool:
        return False


@pytest.fixture
def install_portal_response(monkeypatch):
    # replaces the wallet and aiohttp session so the real fetch runs against a fake portal
    fake_wallet = Mock()
    fake_wallet.get_hotkey.return_value = VALIDATOR_KEYPAIR
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: fake_wallet)

    def install(response: _FakeResponse) -> _FakeSession:
        session = _FakeSession(response)
        monkeypatch.setattr(aiohttp, "ClientSession", lambda timeout: session)
        return session

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
    session = install_portal_response(_FakeResponse(status=200, json_body=OPTED_IN_RECORDS))

    miners = await ValidatorPortalAPI._fetch_opted_in_miners(
        "https://portal.test/validators/opted-in"
    )

    assert miners == OPTED_IN_MINERS
    assert session.last_url.endswith("/validators/opted-in")
    assert set(session.last_headers) == {"hotkey", "timestamp", "signature"}
    # signing the wrong payload would be a permanent 401, i.e. a permanent fallback
    assert VALIDATOR_KEYPAIR.verify(
        session.last_headers["timestamp"], bytes.fromhex(session.last_headers["signature"][2:])
    )


class _FakeAxonInfo:
    """Stand-in for `bittensor.AxonInfo`, where `is_serving` is a property over `ip`.

    That link is the whole mechanism: overwriting `ip` with the central-miner address is
    what keeps an opt-in miner inside the `is_serving` filter. A Mock with a fixed
    `is_serving` attribute would let the filter regress while the tests stayed green.
    """

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    @property
    def is_serving(self) -> bool:
        return self.ip != "0.0.0.0"


def _neuron(hotkey: str, uid: int, ip: str, port: int = 4444) -> Mock:
    return Mock(hotkey=hotkey, uid=uid, axon_info=_FakeAxonInfo(ip, port))


def _subtensor_client(
    monkeypatch, previous_miners: list[Mock], metagraph_neurons: list[Mock]
) -> SubtensorClient:
    client = SubtensorClient.__new__(SubtensorClient)
    client.default_extra = {}
    client.debug_miner = None
    client.miners = previous_miners
    monkeypatch.setattr(client, "get_metagraph", lambda: Mock(neurons=metagraph_neurons))
    return client


@pytest.mark.asyncio
async def test_fetch_miners_still_tracks_the_metagraph_when_portal_has_no_cache(monkeypatch):
    # None only ever happens before the portal answered once, so the previous list was
    # itself built without opt-in routing - freezing it would cost metagraph churn for free
    monkeypatch.setattr(ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=None))
    serving = _neuron("hotkey-c", 3, "1.2.3.4")
    previous = [_neuron("hotkey-a", 1, "10.0.0.1")]
    client = _subtensor_client(monkeypatch, previous, [serving, _neuron("hotkey-b", 2, "0.0.0.0")])

    await client.fetch_miners()

    assert client.miners == [serving]


@pytest.mark.asyncio
async def test_fetch_miners_keeps_opted_in_miner_that_serves_no_axon_of_its_own(monkeypatch):
    monkeypatch.setattr(
        ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=OPTED_IN_MINERS)
    )
    # an opt-in miner publishes 0.0.0.0 on chain and is reachable only via the central miner
    opted_in = _neuron("hotkey-a", 1, "0.0.0.0", port=0)
    plain = _neuron("hotkey-c", 3, "1.2.3.4")
    client = _subtensor_client(monkeypatch, [], [opted_in, plain])

    await client.fetch_miners()

    assert (opted_in.axon_info.ip, opted_in.axon_info.port) == ("10.0.0.1", 8000)
    assert (plain.axon_info.ip, plain.axon_info.port) == ("1.2.3.4", 4444)
    assert client.miners == [opted_in, plain]
