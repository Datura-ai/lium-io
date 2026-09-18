"""Our own chain endpoint must win over the network name (DAH-3579).

Prod (lium-io-deployment) sets both `BITTENSOR_NETWORK=finney` and
`BITTENSOR_CHAIN_ENDPOINT=ws://archive-node-proxy.proxy`. bittensor 10.5 resolves a
Config object by keeping the LAST set candidate, and `bittensor.Config()` defaults
`subtensor.network` to finney, so the endpoint placed in the Config was ignored and the
validator dialled the public, throttled finney node. These tests go through the real
`Subtensor.setup_config`, so a bittensor upgrade that changes the resolution shows up here.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from bittensor.core.subtensor import Subtensor

from clients import subtensor_client as subtensor_client_module
from clients.subtensor_client import SubtensorClient
from core.config import Settings

OWN_ENDPOINT = "ws://archive-node-proxy.proxy"
PUBLIC_FINNEY = "wss://entrypoint-finney.opentensor.ai:443"
PUBLIC_TEST = "wss://test.finney.opentensor.ai:443"


def _resolve(settings: Settings) -> tuple[str, str]:
    """What `bittensor.Subtensor(network=..., config=...)` connects to for these settings."""
    return Subtensor.setup_config(
        settings.get_subtensor_network(), settings.get_bittensor_config()
    )


def test_config_object_alone_dials_the_public_node():
    """Negative control: the pre-fix call `Subtensor(config=...)` with both set resolves
    to the public node. This is the behaviour the fix works around."""
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)

    endpoint, network = Subtensor.setup_config(None, settings.get_bittensor_config())

    assert (endpoint, network) == (PUBLIC_FINNEY, "finney")


def test_both_set_resolves_to_our_endpoint():
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)

    endpoint, _network = _resolve(settings)

    assert endpoint == OWN_ENDPOINT


def test_endpoint_unset_resolves_to_the_named_network():
    settings = Settings(BITTENSOR_NETWORK="test", BITTENSOR_CHAIN_ENDPOINT=None)

    assert _resolve(settings) == (PUBLIC_TEST, "test")


class _RecordingSubtensor:
    """Stands in for `bittensor.Subtensor`: resolves the endpoint the way the real
    constructor does, without opening a websocket."""

    calls: list[dict] = []

    def __init__(self, network=None, config=None, **_kwargs):
        self.chain_endpoint, self.network = Subtensor.setup_config(network, config)
        _RecordingSubtensor.calls.append({"network": network, "config": config})


def _bare_client(settings: Settings) -> SubtensorClient:
    client = SubtensorClient.__new__(SubtensorClient)
    client.config = settings.get_bittensor_config()
    client.default_extra = {"version_key": 0}
    client.check_registered = MagicMock()
    return client


def _connected_extra(caplog) -> dict:
    return next(
        record.msg.extra
        for record in caplog.records
        if getattr(record.msg, "message", None) == "Subtensor connected"
    )


@pytest.fixture
def recording_subtensor(monkeypatch):
    _RecordingSubtensor.calls = []
    monkeypatch.setattr(subtensor_client_module.bittensor, "Subtensor", _RecordingSubtensor)
    monkeypatch.setattr(SubtensorClient, "_subtensor", None)
    return _RecordingSubtensor


def test_initialize_subtensor_dials_our_endpoint_and_logs_it(recording_subtensor, caplog):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()

    assert client.subtensor.chain_endpoint == OWN_ENDPOINT
    assert recording_subtensor.calls[0]["network"] == OWN_ENDPOINT
    extra = _connected_extra(caplog)
    assert extra["chain_endpoint"] == OWN_ENDPOINT
    assert extra["endpoint_source"] == "BITTENSOR_CHAIN_ENDPOINT"


def test_initialize_subtensor_without_endpoint_dials_the_named_network(recording_subtensor, caplog):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()

    assert (client.subtensor.chain_endpoint, client.subtensor.network) == (PUBLIC_FINNEY, "finney")
    extra = _connected_extra(caplog)
    assert extra["chain_endpoint"] == PUBLIC_FINNEY
    assert extra["endpoint_source"] == "BITTENSOR_NETWORK"
