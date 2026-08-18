import asyncio
import json
import logging
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from clients.subtensor_client import (
    PORTAL_MINERS_CACHE_ALERT_SECONDS,
    PORTAL_MINERS_CACHE_KEY,
    OptedInMinerSnapshot,
    ProviderPortalDataUnavailable,
    SubtensorClient,
)
from clients.validator_portal_api import OptedInMiner, ValidatorPortalAPI
from core.config import settings
from core.validator import MINER_SCORES_KEY, Validator


@dataclass
class _AxonInfo:
    ip: str = "0.0.0.0"
    port: int = 0

    @property
    def is_serving(self) -> bool:
        return self.ip != "0.0.0.0"


@dataclass
class _Neuron:
    uid: int
    hotkey: str
    coldkey: str = "coldkey"
    axon_info: _AxonInfo = field(default_factory=_AxonInfo)


def _portal_miner(hotkey: str = "portal-provider") -> OptedInMiner:
    return OptedInMiner(
        miner_hotkey=hotkey,
        central_miner_ip="192.0.2.10",
        central_miner_port=8091,
    )


def _client(*, miners: list[_Neuron] | None = None) -> SubtensorClient:
    client = SubtensorClient.__new__(SubtensorClient)
    client.debug_miner = None
    client.default_extra = {}
    client.miners = miners or []
    client.redis_service = AsyncMock()
    client.redis_service.get.return_value = None
    return client


def _cache_payload(*miners: OptedInMiner, cached_at: float) -> str:
    return OptedInMinerSnapshot(
        cached_at=cached_at,
        miners=list(miners),
    ).model_dump_json()


def _structured_extra(record: logging.LogRecord) -> dict:
    return record.msg.extra


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
async def test_portal_rejects_response_with_a_malformed_provider():
    keypair = MagicMock(ss58_address="validator-hotkey")
    keypair.sign.return_value = b"signed"
    wallet = MagicMock()
    wallet.get_hotkey.return_value = keypair

    response = MagicMock(status=200)
    response.json = AsyncMock(
        return_value=[
            {
                "miner_hotkey": "portal-provider",
                "central_miner_ip": None,
                "central_miner_port": 8091,
            }
        ]
    )
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get.return_value = response_context
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


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("miner_hotkey", ""),
        ("central_miner_ip", None),
        ("central_miner_ip", ""),
        ("central_miner_ip", "0.0.0.0"),
        ("central_miner_port", None),
        ("central_miner_port", 0),
        ("central_miner_port", 65536),
    ],
)
def test_opted_in_miner_requires_a_usable_endpoint(field_name, invalid_value):
    data = {
        "miner_hotkey": "portal-provider",
        "central_miner_ip": "192.0.2.10",
        "central_miner_port": 8091,
    }
    data[field_name] = invalid_value

    with pytest.raises(ValidationError):
        OptedInMiner.model_validate(data)


@pytest.mark.asyncio
async def test_live_portal_snapshot_is_cached_and_used(monkeypatch):
    provider = _Neuron(uid=100, hotkey="portal-provider")
    client = _client()
    client.get_metagraph = MagicMock(return_value=MagicMock(neurons=[provider]))
    monkeypatch.setattr(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        AsyncMock(return_value=[_portal_miner()]),
    )

    await client.fetch_miners()

    assert client.miners == [provider]
    assert provider.axon_info.ip == "192.0.2.10"
    client.redis_service.set.assert_awaited_once()
    cache_key, cache_json = client.redis_service.set.await_args.args
    assert cache_key == PORTAL_MINERS_CACHE_KEY
    cached = OptedInMinerSnapshot.model_validate_json(cache_json)
    assert cached.miners == [_portal_miner()]


@pytest.mark.asyncio
async def test_failed_portal_refresh_uses_redis_snapshot_after_restart(monkeypatch):
    provider = _Neuron(uid=100, hotkey="portal-provider")
    client = _client()
    client.get_metagraph = MagicMock(return_value=MagicMock(neurons=[provider]))
    client.redis_service.get.return_value = _cache_payload(
        _portal_miner(),
        cached_at=1_000,
    )
    monkeypatch.setattr(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr("clients.subtensor_client.time.time", lambda: 1_100)

    await client.fetch_miners()

    assert client.miners == [provider]
    assert provider.axon_info.ip == "192.0.2.10"
    client.redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_portal_and_cache_refresh_keeps_in_memory_snapshot(monkeypatch):
    previous_snapshot = [
        _Neuron(
            uid=100,
            hotkey="portal-provider",
            axon_info=_AxonInfo("192.0.2.10", 8091),
        )
    ]
    client = _client(miners=previous_snapshot)
    client.redis_service.get.side_effect = RuntimeError("redis unavailable")
    monkeypatch.setattr(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        AsyncMock(return_value=None),
    )

    await client.fetch_miners()

    assert client.miners is previous_snapshot


@pytest.mark.asyncio
async def test_portal_failure_without_any_snapshot_is_explicit(monkeypatch):
    client = _client()
    monkeypatch.setattr(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        AsyncMock(return_value=None),
    )

    with pytest.raises(ProviderPortalDataUnavailable):
        await client.fetch_miners()


@pytest.mark.asyncio
async def test_sync_skips_cleanly_when_no_provider_snapshot_exists(caplog):
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.miner_scores = {"portal-provider": 1.0}
    validator.subtensor_client = MagicMock()
    validator.subtensor_client.get_miners = AsyncMock(
        side_effect=ProviderPortalDataUnavailable("no snapshot")
    )
    validator.redis_service = AsyncMock()

    with caplog.at_level(logging.ERROR):
        await validator.sync()

    assert "[sync] No reliable provider snapshot, skipping iteration" in caplog.text
    assert "[sync] Unknown error" not in caplog.text
    cache_key, scores_json = validator.redis_service.set.await_args.args
    assert cache_key == MINER_SCORES_KEY
    assert json.loads(scores_json) == validator.miner_scores


@pytest.mark.asyncio
async def test_stale_redis_snapshot_is_used_and_pages_once(monkeypatch, caplog):
    provider = _Neuron(uid=100, hotkey="portal-provider")
    client = _client()
    client.get_metagraph = MagicMock(return_value=MagicMock(neurons=[provider]))
    client.redis_service.get.return_value = _cache_payload(
        _portal_miner(),
        cached_at=1_000,
    )
    monkeypatch.setattr(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "clients.subtensor_client.time.time",
        lambda: 1_000 + PORTAL_MINERS_CACHE_ALERT_SECONDS,
    )

    with caplog.at_level(logging.CRITICAL):
        await client.fetch_miners()
        await client.fetch_miners()

    alerts = [
        record
        for record in caplog.records
        if record.getMessage() == "[fetch_miners] provider snapshot cache exceeded alert threshold"
    ]
    assert len(alerts) == 1
    assert _structured_extra(alerts[0])["cache_age_seconds"] == PORTAL_MINERS_CACHE_ALERT_SECONDS
    assert client.miners == [provider]


@pytest.mark.asyncio
async def test_live_provider_count_transition_to_zero_warns_and_caches_empty(monkeypatch, caplog):
    client = _client()
    opted_out_provider = _Neuron(
        uid=100,
        hotkey="opted-out-provider",
        axon_info=_AxonInfo("192.0.2.100", 8091),
    )
    client.get_metagraph = MagicMock(
        return_value=MagicMock(neurons=[opted_out_provider])
    )
    client.redis_service.get.return_value = _cache_payload(
        _portal_miner(),
        cached_at=1_000,
    )
    monkeypatch.setattr(
        ValidatorPortalAPI,
        "get_opted_in_miners",
        AsyncMock(return_value=[]),
    )

    with caplog.at_level(logging.WARNING):
        await client.fetch_miners()

    warnings = [
        record
        for record in caplog.records
        if record.getMessage() == "[fetch_miners] opted-in provider count dropped to zero"
    ]
    assert len(warnings) == 1
    cached = OptedInMinerSnapshot.model_validate_json(
        client.redis_service.set.await_args.args[1]
    )
    assert cached.miners == []
    assert client.miners == []


@pytest.mark.parametrize(
    ("active_hotkeys", "expected_level", "expected_active", "expected_inactive"),
    [
        ({"portal-provider"}, logging.CRITICAL, ["portal-provider"], []),
        (set(), logging.WARNING, [], ["portal-provider"]),
    ],
)
@pytest.mark.asyncio
async def test_missing_scored_provider_is_diagnosed_without_blocking_submission(
    monkeypatch,
    caplog,
    active_hotkeys,
    expected_level,
    expected_active,
    expected_inactive,
):
    provider_hotkey = "portal-provider"
    snapshot_miner = _Neuron(
        uid=2,
        hotkey="snapshot-miner",
        axon_info=_AxonInfo("192.0.2.2", 8091),
    )
    registered_provider = _Neuron(uid=100, hotkey=provider_hotkey)
    client = _client()
    client.netuid = 1
    client.version_key = 10000
    client.wallet = MagicMock()
    client.send_weights_to_lium = AsyncMock()
    client.get_current_block = MagicMock(return_value=100)
    client.get_miners = AsyncMock(return_value=[snapshot_miner])
    client.get_metagraph = MagicMock(
        return_value=MagicMock(neurons=[snapshot_miner, registered_provider])
    )

    subtensor = MagicMock()
    subtensor.set_weights.return_value = (True, "ok")
    monkeypatch.setattr(SubtensorClient, "_subtensor", subtensor)

    def preserve_weights(*, uids, weights, **_kwargs):
        return uids, weights

    monkeypatch.setattr(
        "clients.subtensor_client.process_weights_for_netuid",
        preserve_weights,
    )

    with caplog.at_level(expected_level):
        await client.set_weights(
            miner_scores={"snapshot-miner": 0.5, provider_hotkey: 0.5},
            active_hotkeys=active_hotkeys,
        )

    diagnostics = [
        record
        for record in caplog.records
        if record.getMessage() == "[set_weights] scored miners missing from provider snapshot"
    ]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.levelno == expected_level
    assert _structured_extra(diagnostic)["active_missing_hotkeys"] == expected_active
    assert _structured_extra(diagnostic)["inactive_missing_hotkeys"] == expected_inactive
    client.redis_service.publish.assert_awaited_once()
    client.send_weights_to_lium.assert_awaited_once()
    subtensor.set_weights.assert_called_once()


@pytest.mark.asyncio
async def test_legitimate_opt_out_keeps_preexisting_vector_behavior(monkeypatch):
    snapshot_miner = _Neuron(
        uid=2,
        hotkey="snapshot-miner",
        axon_info=_AxonInfo("192.0.2.2", 8091),
    )
    opted_out_provider = _Neuron(
        uid=100,
        hotkey="opted-out-provider",
        axon_info=_AxonInfo("192.0.2.100", 8091),
    )
    client = _client()
    client.netuid = 1
    client.version_key = 10000
    client.wallet = MagicMock()
    client.send_weights_to_lium = AsyncMock()
    client.get_current_block = MagicMock(return_value=100)
    client.get_miners = AsyncMock(return_value=[snapshot_miner])
    client.get_metagraph = MagicMock(
        return_value=MagicMock(neurons=[snapshot_miner, opted_out_provider])
    )

    subtensor = MagicMock()
    subtensor.set_weights.return_value = (True, "ok")
    monkeypatch.setattr(SubtensorClient, "_subtensor", subtensor)
    monkeypatch.setattr(
        "clients.subtensor_client.process_weights_for_netuid",
        lambda *, uids, weights, **_kwargs: (uids, weights),
    )

    await client.set_weights(
        miner_scores={"snapshot-miner": 0.5, "opted-out-provider": 0.5},
    )

    assert list(subtensor.set_weights.call_args.kwargs["uids"]) == [2]
