"""The miner CLI walks the same ordered endpoint list as the miner process (DAH-3579)."""

import logging
from unittest.mock import patch

import pytest
from bittensor.core.subtensor import Subtensor

from core.config import Settings
from services import cli_service as cli_service_module
from services.cli_service import CliService

OWN_ENDPOINT = "ws://archive-node-proxy.proxy"
PUBLIC_FINNEY = "wss://entrypoint-finney.opentensor.ai:443"


class _RecordingSubtensor:
    calls: list[str] = []

    def __init__(self, network=None, config=None, **_kwargs):
        self.chain_endpoint, self.network = Subtensor.setup_config(network, config)
        self.substrate = object()
        _RecordingSubtensor.calls.append(network)


class _RefusingThenRecording(_RecordingSubtensor):
    def __init__(self, network=None, config=None, **kwargs):
        if network == OWN_ENDPOINT:
            raise ConnectionRefusedError("[Errno 111] Connect call failed")
        super().__init__(network=network, config=config, **kwargs)


def _make_cli(settings: Settings) -> CliService:
    cli = CliService.__new__(CliService)
    cli.config = settings.get_bittensor_config()
    cli.default_extra = {"hotkey": "test-hotkey", "netuid": 51}
    cli.logger = logging.getLogger("cli-endpoint-test")
    cli.subtensor = None
    return cli


@pytest.fixture
def recording_subtensor(monkeypatch):
    _RecordingSubtensor.calls = []
    monkeypatch.setattr(cli_service_module.bt, "Subtensor", _RecordingSubtensor)
    return _RecordingSubtensor


def test_get_node_dials_our_endpoint_first(recording_subtensor):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    cli = _make_cli(settings)

    with patch.object(cli_service_module, "settings", settings):
        node = cli.get_node()

    assert node is cli.subtensor.substrate
    assert cli.subtensor.chain_endpoint == OWN_ENDPOINT
    assert recording_subtensor.calls == [OWN_ENDPOINT]


def test_get_node_falls_back_to_the_public_node_when_the_proxy_is_down(monkeypatch, caplog):
    _RecordingSubtensor.calls = []
    monkeypatch.setattr(cli_service_module.bt, "Subtensor", _RefusingThenRecording)
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    cli = _make_cli(settings)

    with patch.object(cli_service_module, "settings", settings), caplog.at_level(logging.INFO):
        node = cli.get_node()

    assert node is cli.subtensor.substrate
    assert cli.subtensor.chain_endpoint == PUBLIC_FINNEY
    assert _RecordingSubtensor.calls == ["finney"]
    assert any(
        "Subtensor endpoint switched" in (getattr(r.msg, "message", None) or str(r.msg))
        for r in caplog.records
    )


def test_get_node_without_an_own_endpoint_dials_the_public_node_once(recording_subtensor):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)
    cli = _make_cli(settings)

    with patch.object(cli_service_module, "settings", settings):
        cli.get_node()

    assert recording_subtensor.calls == ["finney"]
    assert cli.subtensor.chain_endpoint == PUBLIC_FINNEY
