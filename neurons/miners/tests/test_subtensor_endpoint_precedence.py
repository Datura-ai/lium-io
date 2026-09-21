"""Chain endpoint precedence for the miner (DAH-3579).

Providers run the miner with `BITTENSOR_NETWORK=finney` and no `BITTENSOR_CHAIN_ENDPOINT`
(`.env.template`), so for them the resolved endpoint must stay the public finney node. Our
central miner (lium-io-deployment) sets both; there the endpoint must win, as in the
validator. Resolution goes through the real `AsyncSubtensor.setup_config` so a bittensor
upgrade that changes the rule shows up here.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from bittensor.core.async_subtensor import AsyncSubtensor

import core.miner as miner_module
from core.config import Settings
from core.miner import Miner

OWN_ENDPOINT = "ws://archive-node-proxy.proxy"
PUBLIC_FINNEY = "wss://entrypoint-finney.opentensor.ai:443"


def _resolve(settings: Settings) -> tuple[str, str]:
    return AsyncSubtensor.setup_config(
        settings.get_chain_endpoint_or_network_name(), settings.get_bittensor_config()
    )


def test_provider_shape_network_name_only_stays_on_the_public_node():
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)

    assert settings.get_chain_endpoint_or_network_name() == "finney"
    assert _resolve(settings) == (PUBLIC_FINNEY, "finney")


def test_both_set_resolves_to_the_endpoint():
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)

    endpoint, _network = _resolve(settings)

    assert endpoint == OWN_ENDPOINT


class _RecordingAsyncSubtensor:
    """Stands in for `bittensor.AsyncSubtensor`: resolves the endpoint the way the real
    constructor does, without opening a websocket."""

    calls: list[dict] = []

    def __init__(self, network=None, config=None, **_kwargs):
        self.chain_endpoint, self.network = AsyncSubtensor.setup_config(network, config)
        _RecordingAsyncSubtensor.calls.append({"network": network, "config": config})

    async def initialize(self):
        return self

    async def close(self):
        return None


def _make_miner(settings: Settings) -> Miner:
    miner = Miner.__new__(Miner)
    miner.config = settings.get_bittensor_config()
    miner.netuid = 51
    miner.subtensor = None
    miner.bootstrap_complete = False
    miner.should_exit = False
    miner.default_extra = {"external_ip": "203.0.113.10", "external_port": 8000}
    miner.wallet = SimpleNamespace(
        get_hotkey=lambda: SimpleNamespace(ss58_address="test-hotkey")
    )
    miner.check_registered = AsyncMock()
    return miner


def _connected_extra(caplog) -> dict:
    return next(
        record.msg.extra
        for record in caplog.records
        if getattr(record.msg, "message", None) == "Subtensor connected"
    )


@pytest.fixture
def recording_subtensor(monkeypatch):
    _RecordingAsyncSubtensor.calls = []
    monkeypatch.setattr(miner_module.bittensor, "AsyncSubtensor", _RecordingAsyncSubtensor)
    return _RecordingAsyncSubtensor


@pytest.mark.asyncio
async def test_initialize_subtensor_with_endpoint_dials_it_and_logs_it(recording_subtensor, caplog):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    miner = _make_miner(settings)

    with patch.object(miner_module, "settings", settings), caplog.at_level(logging.INFO):
        await miner.initialize_subtensor()

    assert miner.subtensor.chain_endpoint == OWN_ENDPOINT
    assert recording_subtensor.calls[0]["network"] == OWN_ENDPOINT
    extra = _connected_extra(caplog)
    assert extra["chain_endpoint"] == OWN_ENDPOINT
    assert extra["endpoint_source"] == "BITTENSOR_CHAIN_ENDPOINT"


@pytest.mark.asyncio
async def test_initialize_subtensor_provider_shape_is_unchanged(recording_subtensor, caplog):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)
    miner = _make_miner(settings)

    with patch.object(miner_module, "settings", settings), caplog.at_level(logging.INFO):
        await miner.initialize_subtensor()

    assert (miner.subtensor.chain_endpoint, miner.subtensor.network) == (PUBLIC_FINNEY, "finney")
    extra = _connected_extra(caplog)
    assert extra["endpoint_source"] == "BITTENSOR_NETWORK"


class _RefusingThenRecordingAsyncSubtensor(_RecordingAsyncSubtensor):
    """The first dial (our own endpoint) refuses the way a proxy outage does; later dials record."""

    refused: list[str] = []

    def __init__(self, network=None, config=None, **kwargs):
        self._refuse = network == OWN_ENDPOINT
        if self._refuse:
            _RefusingThenRecordingAsyncSubtensor.refused.append(network)
            return
        super().__init__(network=network, config=config, **kwargs)

    async def initialize(self):
        if self._refuse:
            raise ConnectionRefusedError("[Errno 111] Connect call failed")
        return self


@pytest.mark.asyncio
async def test_initialize_subtensor_falls_back_to_the_public_node_when_our_endpoint_fails(
    monkeypatch, caplog
):
    """Rustam's review (21 Sep): the central miner depends on the proxy too. The own endpoint is
    tried first; when it fails and one is configured, the public `BITTENSOR_NETWORK` node is
    dialled and the log says why. Providers set no endpoint, so nothing changes for them."""
    _RecordingAsyncSubtensor.calls = []
    _RefusingThenRecordingAsyncSubtensor.refused = []
    monkeypatch.setattr(
        miner_module.bittensor, "AsyncSubtensor", _RefusingThenRecordingAsyncSubtensor
    )
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    miner = _make_miner(settings)

    with patch.object(miner_module, "settings", settings), caplog.at_level(logging.INFO):
        await miner.initialize_subtensor()

    assert _RefusingThenRecordingAsyncSubtensor.refused == [OWN_ENDPOINT]
    assert (miner.subtensor.chain_endpoint, miner.subtensor.network) == (PUBLIC_FINNEY, "finney")
    assert _RecordingAsyncSubtensor.calls[-1]["network"] == "finney"
    extra = _connected_extra(caplog)
    assert extra["chain_endpoint"] == PUBLIC_FINNEY
    assert extra["endpoint_source"] == "BITTENSOR_NETWORK (own endpoint failed)"
