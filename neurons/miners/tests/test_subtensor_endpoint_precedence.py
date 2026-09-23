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

import datura.chain as chain_module
from datura.chain import EndpointCursor, PUBLIC_NODE_SOURCE, is_chain_error

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
    miner._endpoint_cursor = EndpointCursor(
        settings.get_chain_endpoints(),
        retry_after_seconds=settings.BITTENSOR_CHAIN_ENDPOINT_RETRY_AFTER_SECONDS,
    )
    miner.last_cycle_ran_on_fallback = False
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
def clock(monkeypatch) -> list[float]:
    """The monotonic clock the endpoint cursor reads; move it with `clock[0] += seconds`."""
    now = [1000.0]
    monkeypatch.setattr(chain_module, "monotonic", lambda: now[0])
    return now


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
    assert extra["endpoint_source"] == PUBLIC_NODE_SOURCE


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


def _switch_lines(caplog) -> list[str]:
    return [
        r.msg.message
        for r in caplog.records
        if getattr(r.msg, "message", "").startswith("Subtensor endpoint switched")
    ]


@pytest.mark.asyncio
async def test_initialize_subtensor_falls_back_to_the_public_node_when_our_endpoint_fails(
    monkeypatch, caplog
):
    """taiberium (22 Sep): proxy refused → the public node answers → ONE
    `Subtensor endpoint switched from=… to=…` line. Providers set no endpoint, so nothing
    changes for them."""
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
    assert _switch_lines(caplog) == [f"Subtensor endpoint switched from={OWN_ENDPOINT} to=finney"]


SECOND_ENDPOINT = "ws://archive-node-proxy-2.proxy"


def test_chain_endpoints_list_is_ordered_with_the_public_node_last():
    settings = Settings(
        BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINTS=f"{OWN_ENDPOINT},{SECOND_ENDPOINT}"
    )

    assert [e.value for e in settings.get_chain_endpoints()] == [OWN_ENDPOINT, SECOND_ENDPOINT, "finney"]
    assert settings.get_chain_endpoint_or_network_name() == OWN_ENDPOINT


@pytest.mark.asyncio
async def test_sync_read_failure_rests_the_endpoint_for_the_retry_window_then_returns_to_it(
    recording_subtensor, clock, caplog
):
    """`sync()` fails on a read after connecting to the proxy: the redial goes to the public node
    (one switch line). Syncs inside the retry window stay on the public node and do not redial;
    the first sync after the window goes back to the proxy."""
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    miner = _make_miner(settings)
    miner.bootstrap = AsyncMock(side_effect=[TimeoutError("metagraph read timed out"), None, None])

    with patch.object(miner_module, "settings", settings), caplog.at_level(logging.INFO):
        await miner.sync()  # connects to the proxy, bootstrap read fails → switched, redialled
        assert miner.subtensor.chain_endpoint == PUBLIC_FINNEY
        assert _switch_lines(caplog) == [f"Subtensor endpoint switched from={OWN_ENDPOINT} to=finney"]

        await miner.sync()  # completes on the public node
        on_public = miner.subtensor
        assert miner.last_cycle_ran_on_fallback

        clock[0] += settings.BITTENSOR_CHAIN_ENDPOINT_RETRY_AFTER_SECONDS - 1
        await miner.sync()  # inside the window: stays on the public node
        assert miner.subtensor is on_public

        clock[0] += 1
        await miner.sync()  # the window is over: back to the proxy
        assert miner.subtensor.chain_endpoint == OWN_ENDPOINT
        assert not miner.last_cycle_ran_on_fallback

    assert [c["network"] for c in recording_subtensor.calls] == [OWN_ENDPOINT, "finney", OWN_ENDPOINT]
    assert _switch_lines(caplog) == [f"Subtensor endpoint switched from={OWN_ENDPOINT} to=finney"]


@pytest.mark.asyncio
async def test_sync_read_failure_without_an_own_endpoint_redials_the_same_node(
    recording_subtensor, caplog
):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)
    miner = _make_miner(settings)
    miner.last_cycle_ran_on_fallback = False
    miner.bootstrap = AsyncMock(side_effect=[TimeoutError("read timed out"), None])

    with patch.object(miner_module, "settings", settings), caplog.at_level(logging.INFO):
        await miner.sync()

    assert miner.subtensor.chain_endpoint == PUBLIC_FINNEY
    assert [c["network"] for c in recording_subtensor.calls] == ["finney", "finney"]
    assert _switch_lines(caplog) == []


def test_is_chain_error_is_true_only_for_chain_client_faults():
    class RedisError(Exception):
        __module__ = "redis.exceptions"

    class PortalError(Exception):
        __module__ = "aiohttp.client_exceptions"

    class DatabaseError(Exception):
        __module__ = "sqlalchemy.exc"

    assert is_chain_error(TimeoutError("metagraph read timed out"))
    assert is_chain_error(ConnectionRefusedError("[Errno 111] Connect call failed"))
    assert not is_chain_error(RedisError("redis down"))
    assert not is_chain_error(PortalError("portal 502"))
    assert not is_chain_error(DatabaseError("disk full"))
    assert not is_chain_error(RuntimeError("save_validators failed"))


@pytest.mark.asyncio
async def test_sync_database_error_leaves_a_healthy_proxy_in_place(
    recording_subtensor, caplog
):
    """A SQL error in save_validators is not a chain fault: the cursor stays on the proxy
    and the next sync does not log `read failed` or dial the public node."""
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    miner = _make_miner(settings)
    miner.bootstrap = AsyncMock(side_effect=RuntimeError("save_validators: disk full"))

    with patch.object(miner_module, "settings", settings), caplog.at_level(logging.INFO):
        await miner.sync()

    assert miner.subtensor.chain_endpoint == OWN_ENDPOINT
    assert miner._endpoints.on_first
    assert [c["network"] for c in recording_subtensor.calls] == [OWN_ENDPOINT]
    assert _switch_lines(caplog) == []
