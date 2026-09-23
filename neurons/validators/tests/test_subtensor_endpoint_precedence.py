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

from datura.chain import EndpointCursor, PUBLIC_NODE_SOURCE, is_chain_error

from clients import subtensor_client as subtensor_client_module
from clients.subtensor_client import ProviderPortalDataUnavailable, SubtensorClient
from core.config import Settings

OWN_ENDPOINT = "ws://archive-node-proxy.proxy"
PUBLIC_FINNEY = "wss://entrypoint-finney.opentensor.ai:443"
PUBLIC_TEST = "wss://test.finney.opentensor.ai:443"


def _resolve(settings: Settings) -> tuple[str, str]:
    """What `bittensor.Subtensor(network=..., config=...)` connects to for these settings."""
    return Subtensor.setup_config(
        settings.get_chain_endpoint_or_network_name(), settings.get_bittensor_config()
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
        self.closed = False
        _RecordingSubtensor.calls.append({"network": network, "config": config})

    def close(self):
        self.closed = True


def _bare_client(settings: Settings) -> SubtensorClient:
    client = SubtensorClient.__new__(SubtensorClient)
    client.config = settings.get_bittensor_config()
    client.default_extra = {"version_key": 0}
    client._endpoint_cursor = EndpointCursor(settings.get_chain_endpoints())
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
    assert extra["endpoint_source"] == PUBLIC_NODE_SOURCE


class _RefusingThenRecordingSubtensor(_RecordingSubtensor):
    """The first dial (our own endpoint) refuses the way a proxy outage does; later dials record."""

    refused: list[str] = []

    def __init__(self, network=None, config=None, **kwargs):
        if network == OWN_ENDPOINT:
            _RefusingThenRecordingSubtensor.refused.append(network)
            raise ConnectionRefusedError("[Errno 111] Connect call failed")
        super().__init__(network=network, config=config, **kwargs)


def _switch_lines(caplog) -> list[str]:
    return [
        r.msg.message
        for r in caplog.records
        if getattr(r.msg, "message", "").startswith("Subtensor endpoint switched")
    ]


def test_initialize_subtensor_falls_back_to_the_public_node_when_our_endpoint_fails(
    monkeypatch, caplog
):
    """taiberium (22 Sep): proxy refused → the public node answers → ONE
    `Subtensor endpoint switched from=… to=…` line. Without it the validator had no chain
    client and metagraph sync and set_weights stopped."""
    _RecordingSubtensor.calls = []
    _RefusingThenRecordingSubtensor.refused = []
    monkeypatch.setattr(subtensor_client_module.bittensor, "Subtensor", _RefusingThenRecordingSubtensor)
    monkeypatch.setattr(SubtensorClient, "_subtensor", None)
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()

    assert _RefusingThenRecordingSubtensor.refused == [OWN_ENDPOINT]
    assert (client.subtensor.chain_endpoint, client.subtensor.network) == (PUBLIC_FINNEY, "finney")
    assert _RecordingSubtensor.calls[-1]["network"] == "finney"
    extra = _connected_extra(caplog)
    assert extra["chain_endpoint"] == PUBLIC_FINNEY
    assert extra["endpoint_source"] == "BITTENSOR_NETWORK (own endpoint failed)"
    assert _switch_lines(caplog) == [f"Subtensor endpoint switched from={OWN_ENDPOINT} to=finney"]


SECOND_ENDPOINT = "ws://archive-node-proxy-2.proxy"


def test_chain_endpoints_list_is_ordered_with_the_public_node_last():
    settings = Settings(
        BITTENSOR_NETWORK="finney",
        BITTENSOR_CHAIN_ENDPOINTS=f"{OWN_ENDPOINT}, {SECOND_ENDPOINT},,finney",
        BITTENSOR_CHAIN_ENDPOINT="ws://ignored-when-the-list-is-set",
    )

    assert [e.value for e in settings.get_chain_endpoints()] == [OWN_ENDPOINT, SECOND_ENDPOINT, "finney"]
    assert [e.source for e in settings.get_chain_endpoints()] == [
        "BITTENSOR_CHAIN_ENDPOINTS[0]",
        "BITTENSOR_CHAIN_ENDPOINTS[1]",
        PUBLIC_NODE_SOURCE,
    ]
    assert settings.get_chain_endpoint_or_network_name() == OWN_ENDPOINT
    assert _resolve(settings)[0] == OWN_ENDPOINT


def test_endpoint_list_walks_every_own_node_before_the_public_one(monkeypatch, caplog):
    """Two proxies down: two switch lines, the public node answers, the cursor sits on it."""
    refused: list[str] = []

    class _RefusingOwnNodes(_RecordingSubtensor):
        def __init__(self, network=None, config=None, **kwargs):
            if network in (OWN_ENDPOINT, SECOND_ENDPOINT):
                refused.append(network)
                raise ConnectionRefusedError("[Errno 111] Connect call failed")
            super().__init__(network=network, config=config, **kwargs)

    _RecordingSubtensor.calls = []
    monkeypatch.setattr(subtensor_client_module.bittensor, "Subtensor", _RefusingOwnNodes)
    monkeypatch.setattr(SubtensorClient, "_subtensor", None)
    settings = Settings(
        BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINTS=f"{OWN_ENDPOINT},{SECOND_ENDPOINT}"
    )
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()

    assert refused == [OWN_ENDPOINT, SECOND_ENDPOINT]
    assert client.subtensor.chain_endpoint == PUBLIC_FINNEY
    assert _switch_lines(caplog) == [
        f"Subtensor endpoint switched from={OWN_ENDPOINT} to={SECOND_ENDPOINT}",
        f"Subtensor endpoint switched from={SECOND_ENDPOINT} to=finney",
    ]
    assert client._endpoints.current.value == "finney"


def test_read_failure_switches_to_the_next_endpoint_and_the_next_sync_returns_to_the_first(
    recording_subtensor, caplog
):
    """A read on the proxy fails after connecting: the client is dropped, the redial goes to
    the public node (one switch line); after a completed sync cycle the cursor is back on the
    proxy and the next dial tries it first."""
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()
        assert client.subtensor.chain_endpoint == OWN_ENDPOINT

        client._switch_endpoint_after_read_failure(TimeoutError("metagraph read timed out"))
        assert client.subtensor is None
        client.set_subtensor()
        assert client.subtensor.chain_endpoint == PUBLIC_FINNEY
        assert _switch_lines(caplog) == [f"Subtensor endpoint switched from={OWN_ENDPOINT} to=finney"]
        assert not client._endpoints.on_first

        client._return_to_first_endpoint()
        assert client.subtensor is None and client._endpoints.on_first
        client.set_subtensor()
        assert client.subtensor.chain_endpoint == OWN_ENDPOINT

    assert [c["network"] for c in recording_subtensor.calls] == [OWN_ENDPOINT, "finney", OWN_ENDPOINT]


def test_switch_and_return_close_the_dropped_client(recording_subtensor):
    """A dropped client still holds an open websocket; it must be closed, not just forgotten."""
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings):
        client.initialize_subtensor()
        on_proxy = client.subtensor
        client._switch_endpoint_after_read_failure(TimeoutError("read timed out"))
        client.set_subtensor()
        on_public = client.subtensor
        client._return_to_first_endpoint()

    assert on_proxy.closed and on_public.closed


def test_read_failure_without_an_own_endpoint_does_not_switch(recording_subtensor, caplog):
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()
        client._switch_endpoint_after_read_failure(TimeoutError("read timed out"))

    assert client.subtensor is not None
    assert _switch_lines(caplog) == []


def test_initialize_subtensor_without_endpoint_does_not_retry_on_failure(monkeypatch, caplog):
    """No own endpoint configured: a failure is the failure it always was (logged by
    initialize_subtensor, no client), not a second dial of the same public node."""
    calls: list[str] = []

    class _Refusing:
        def __init__(self, network=None, config=None, **_kwargs):
            calls.append(network)
            raise ConnectionRefusedError("[Errno 111] Connect call failed")

    monkeypatch.setattr(subtensor_client_module.bittensor, "Subtensor", _Refusing)
    monkeypatch.setattr(SubtensorClient, "_subtensor", None)
    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=None)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()

    assert calls == ["finney"]
    assert client.subtensor is None


def test_redis_or_portal_error_leaves_a_healthy_endpoint_in_place(recording_subtensor, caplog):
    """A Redis miss or a portal HTTP failure is not a chain fault: the cursor stays
    on the proxy and the next cycle does not log `read failed`."""
    class RedisError(Exception):
        __module__ = "redis.exceptions"

    settings = Settings(BITTENSOR_NETWORK="finney", BITTENSOR_CHAIN_ENDPOINT=OWN_ENDPOINT)
    client = _bare_client(settings)

    with patch.object(subtensor_client_module, "settings", settings), caplog.at_level(logging.INFO):
        client.initialize_subtensor()
        on_proxy = client.subtensor
        client._switch_endpoint_after_read_failure(RedisError("redis down"))
        client._switch_endpoint_after_read_failure(
            ProviderPortalDataUnavailable("no snapshot")
        )

    assert client.subtensor is on_proxy
    assert client._endpoints.on_first
    assert _switch_lines(caplog) == []
    assert not is_chain_error(RedisError("redis down"))
    assert not is_chain_error(ProviderPortalDataUnavailable("no snapshot"))
    assert is_chain_error(TimeoutError("metagraph read timed out"))
