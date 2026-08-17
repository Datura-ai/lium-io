import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import bittensor
import pytest

from clients.subtensor_client import SubtensorClient
from clients.validator_portal_api import ValidatorPortalAPI
from core.config import settings


def _neuron(uid: int, hotkey: str, *, is_serving: bool) -> bittensor.NeuronInfo:
    neuron = MagicMock(spec=bittensor.NeuronInfo)
    neuron.uid = uid
    neuron.hotkey = hotkey
    neuron.coldkey = "coldkey"
    neuron.axon_info = MagicMock()
    neuron.axon_info.ip = "192.0.2.1"
    neuron.axon_info.port = 8091
    neuron.axon_info.is_serving = is_serving
    return neuron


@pytest.mark.asyncio
async def test_portal_timeout_is_not_reported_as_empty_opt_in_list():
    keypair = MagicMock()
    keypair.ss58_address = "validator-hotkey"
    keypair.sign.return_value = b"signed"
    wallet = MagicMock()
    wallet.get_hotkey.return_value = keypair

    session = MagicMock()
    session.get.side_effect = asyncio.TimeoutError
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("core.config.Settings.get_bittensor_wallet", return_value=wallet),
        patch.object(settings, "MINER_PORTAL_REST_API_URL", "https://portal.test"),
        patch(
            "clients.validator_portal_api.aiohttp.ClientSession",
            return_value=session_context,
        ),
    ):
        result = await ValidatorPortalAPI.get_opted_in_miners()

    assert result is None


@pytest.mark.asyncio
async def test_failed_portal_refresh_keeps_previous_miner_snapshot():
    cached_provider = _neuron(100, "portal-provider", is_serving=True)
    on_chain_miner = _neuron(2, "on-chain-miner", is_serving=True)
    registered_provider = _neuron(100, "portal-provider", is_serving=False)

    client = SubtensorClient.__new__(SubtensorClient)
    client.debug_miner = None
    client.default_extra = {}
    client.miners = [cached_provider, on_chain_miner]
    previous_snapshot = client.miners
    client.get_metagraph = MagicMock(
        return_value=MagicMock(neurons=[registered_provider, on_chain_miner])
    )

    with patch.object(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        new=AsyncMock(return_value=None),
    ):
        with pytest.raises(RuntimeError, match="provider portal"):
            await client.fetch_miners()

    assert client.miners is previous_snapshot
    assert [miner.hotkey for miner in client.miners] == [
        "portal-provider",
        "on-chain-miner",
    ]


@pytest.mark.asyncio
async def test_cold_start_portal_failure_does_not_submit_restored_scores():
    provider_hotkey = "portal-provider"
    registered_provider = _neuron(100, provider_hotkey, is_serving=False)

    client = SubtensorClient.__new__(SubtensorClient)
    client.debug_miner = None
    client.default_extra = {}
    client.miners = []
    client.get_metagraph = MagicMock(
        return_value=MagicMock(neurons=[registered_provider])
    )

    subtensor = MagicMock()
    SubtensorClient._subtensor = subtensor

    with patch.object(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        new=AsyncMock(return_value=None),
    ):
        with pytest.raises(RuntimeError, match="provider portal"):
            await client.set_weights(miner_scores={provider_hotkey: 1.0})

    assert client.miners == []
    subtensor.set_weights.assert_not_called()


@pytest.mark.asyncio
async def test_set_weights_stops_before_chain_when_registered_positive_score_is_missing():
    provider_hotkey = "portal-provider"
    on_chain_miner = _neuron(2, "on-chain-miner", is_serving=True)
    registered_provider = _neuron(100, provider_hotkey, is_serving=False)

    client = SubtensorClient.__new__(SubtensorClient)
    client.netuid = 1
    client.default_extra = {}
    client.version_key = 10000
    client.wallet = MagicMock()
    client.redis_service = AsyncMock()
    client.send_weights_to_lium = AsyncMock()
    client.get_current_block = MagicMock(return_value=100)
    client.get_miners = AsyncMock(return_value=[on_chain_miner])
    client.get_metagraph = MagicMock(
        return_value=MagicMock(neurons=[on_chain_miner, registered_provider])
    )

    subtensor = MagicMock()
    subtensor.set_weights.return_value = (True, "ok")
    SubtensorClient._subtensor = subtensor

    def preserve_weights(*, uids, weights, **_kwargs):
        return uids, weights

    with patch(
        "clients.subtensor_client.process_weights_for_netuid",
        side_effect=preserve_weights,
    ):
        with pytest.raises(RuntimeError, match="positive-score"):
            await client.set_weights(
                miner_scores={"on-chain-miner": 0.5, provider_hotkey: 0.5},
                active_hotkeys={provider_hotkey},
            )

    client.redis_service.publish.assert_not_awaited()
    client.send_weights_to_lium.assert_not_awaited()
    subtensor.set_weights.assert_not_called()
