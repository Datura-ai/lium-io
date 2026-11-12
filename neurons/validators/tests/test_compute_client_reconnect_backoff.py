import itertools
from unittest.mock import MagicMock

import clients.compute_client as compute_client
from clients.compute_client import (
    WS_RECONNECT_MAX_DELAY,
    ComputeClient,
    reconnect_delays,
)


def test_reconnect_waits_stay_under_the_backends_rent_wait() -> None:
    delays = list(itertools.islice(reconnect_delays(), 50))
    assert all(0 < d <= WS_RECONNECT_MAX_DELAY for d in delays)
    assert (
        WS_RECONNECT_MAX_DELAY < 10
    )  # the backend's rent wait for a validator connection
    assert max(delays[10:]) >= WS_RECONNECT_MAX_DELAY / 2  # grows to the cap, jittered


def test_a_new_sequence_starts_short_again() -> None:
    # websockets takes a new sequence after each successful connect, so this is the reset
    for _ in range(20):
        seq = reconnect_delays()
        assert next(seq) <= 1
        assert next(seq) <= 2


def test_connect_uses_the_capped_delays(monkeypatch) -> None:
    connect = MagicMock()
    monkeypatch.setattr(compute_client.websockets, "connect", connect)
    client = ComputeClient.__new__(ComputeClient)
    client.compute_app_uri = "ws://backend"
    client.logging_extra = {}
    client.connect()
    assert connect.call_args.kwargs["reconnect_delays"] is reconnect_delays
