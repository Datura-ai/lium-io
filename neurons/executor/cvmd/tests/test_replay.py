"""Replay protection: rejection, restart durability, write ordering, and availability.

The availability tests are as load-bearing as the rejection tests. A replay gate that 401s
concurrent requests and client retries reads as an intermittent auth bug in production, and the
pressure is then to loosen it — so the property is pinned here rather than left to judgement.
"""

import time

import pytest
from conftest import sign_headers, signed_request
from cvmd.app import create_app
from cvmd.auth.replay import (
    FLOOR_FILENAME,
    NANOSECONDS_PER_SECOND,
    NONCES_FILENAME,
    ReplayRejected,
    ReplayStore,
)
from fastapi.testclient import TestClient

STATE_PATH = "/v1/state"


class TestRejection:
    def test_same_nonce_twice_is_rejected(self, client, validator_key):
        nonce = "a" * 32
        first = signed_request(client, validator_key, "GET", STATE_PATH, nonce=nonce)
        second = signed_request(client, validator_key, "GET", STATE_PATH, nonce=nonce)

        assert first.status_code == 200
        assert second.status_code == 401

    def test_replaying_the_exact_request_is_rejected(self, client, validator_key):
        """Byte-for-byte capture and resend — the whole point of the control."""
        headers = sign_headers(validator_key, method="GET", request_target=STATE_PATH)

        assert client.get(STATE_PATH, headers=headers).status_code == 200
        assert client.get(STATE_PATH, headers=headers).status_code == 401

    def test_stale_timestamp_is_rejected(self, client, validator_key):
        stale = time.time_ns() - 3600 * 1_000_000_000
        response = signed_request(client, validator_key, "GET", STATE_PATH, timestamp_ns=stale)
        assert response.status_code == 401

    def test_far_future_timestamp_is_rejected(self, client, validator_key):
        """The window is two-sided: a future timestamp would otherwise park a usable request."""
        future = time.time_ns() + 3600 * 1_000_000_000
        response = signed_request(client, validator_key, "GET", STATE_PATH, timestamp_ns=future)
        assert response.status_code == 401


class TestAvailability:
    """What strict monotonic timestamps would have broken."""

    def test_out_of_order_timestamps_are_both_accepted(self, client, platform_key):
        """The platform key interleaves a /v1/state poll with a DELETE; arrival order is not
        timestamp order. Under monotonicity the second to arrive would 401.
        """
        now = time.time_ns()
        later = signed_request(client, platform_key, "GET", STATE_PATH, timestamp_ns=now)
        earlier = signed_request(client, platform_key, "GET", STATE_PATH, timestamp_ns=now - 1000)

        assert later.status_code == 200
        assert earlier.status_code == 200

    def test_retry_with_same_timestamp_and_fresh_nonce_is_accepted(self, client, platform_key):
        """A client that retries reuses its timestamp. That must not look like a replay."""
        timestamp = time.time_ns()
        first = signed_request(
            client, platform_key, "GET", STATE_PATH, timestamp_ns=timestamp, nonce="b" * 32
        )
        retry = signed_request(
            client, platform_key, "GET", STATE_PATH, timestamp_ns=timestamp, nonce="c" * 32
        )

        assert first.status_code == 200
        assert retry.status_code == 200

    def test_many_requests_in_the_same_nanosecond_are_all_accepted(self, client, platform_key):
        timestamp = time.time_ns()
        for index in range(5):
            response = signed_request(
                client,
                platform_key,
                "GET",
                STATE_PATH,
                timestamp_ns=timestamp,
                nonce=f"{index:032x}",
            )
            assert response.status_code == 200, f"request {index} was rejected"


class TestRestartDurability:
    def _restart(self, config) -> TestClient:
        """A new app on the same state dir — what systemd does after a crash."""
        return TestClient(create_app(config), raise_server_exceptions=False)

    def test_replay_after_restart_is_rejected(self, client, config, validator_key):
        headers = sign_headers(validator_key, method="GET", request_target=STATE_PATH)
        assert client.get(STATE_PATH, headers=headers).status_code == 200

        with self._restart(config) as restarted:
            assert restarted.get(STATE_PATH, headers=headers).status_code == 401

    def test_startup_floor_rejects_older_requests_after_restart(self, client, config, platform_key):
        """Even with a fresh nonce: everything at or below the previous life's high-water mark."""
        accepted_at = time.time_ns()
        assert (
            signed_request(
                client, platform_key, "GET", STATE_PATH, timestamp_ns=accepted_at
            ).status_code
            == 200
        )

        with self._restart(config) as restarted:
            response = signed_request(
                restarted, platform_key, "GET", STATE_PATH, timestamp_ns=accepted_at - 1
            )
            assert response.status_code == 401

    def test_floor_survives_a_lost_nonce_file(self, client, config, state_dir, validator_key):
        """The two records are independent: losing the window file leaves the floor standing."""
        accepted_at = time.time_ns()
        headers = sign_headers(
            validator_key, method="GET", request_target=STATE_PATH, timestamp_ns=accepted_at
        )
        assert client.get(STATE_PATH, headers=headers).status_code == 200

        (state_dir / NONCES_FILENAME).unlink()
        assert (state_dir / FLOOR_FILENAME).exists()

        with self._restart(config) as restarted:
            assert restarted.get(STATE_PATH, headers=headers).status_code == 401

    def test_corrupt_nonce_file_does_not_prevent_startup(self, config, state_dir, validator_key):
        """A corrupt window file is recoverable state, not an outage — the floor still guards."""
        (state_dir / NONCES_FILENAME).write_text("{ this is not json")

        with self._restart(config) as restarted:
            assert signed_request(restarted, validator_key, "GET", STATE_PATH).status_code == 200

    def test_fresh_request_after_restart_is_accepted(self, client, config, validator_key):
        """The floor must not wedge the daemon shut after a restart."""
        assert signed_request(client, validator_key, "GET", STATE_PATH).status_code == 200

        with self._restart(config) as restarted:
            assert signed_request(restarted, validator_key, "GET", STATE_PATH).status_code == 200

    def test_an_in_skew_future_timestamp_holds_the_floor_ahead_of_wall_clock(
        self, client, config, validator_key
    ):
        """A future-dated request refuses every caller after a restart until now catches up.

        The skew window is two-sided, so a timestamp up to skew_seconds ahead is valid and, once
        accepted, becomes the persisted high-water mark. The next start reads it as the floor —
        which is now in the future — and refuses present-time requests until wall-clock passes it.

        Bounded by skew_seconds and self-healing, but real: found on hardware during the DAH-2575
        acceptance run, where a request 5s ahead made the daemon refuse everything for 5s after a
        restart. Pinned so that changing it is a decision rather than an accident. See the
        module docstring in auth/replay.py for why the floor cannot simply be clamped to `now`.
        """
        ahead_ns = time.time_ns() + 30 * NANOSECONDS_PER_SECOND
        assert (
            signed_request(
                client, validator_key, "GET", STATE_PATH, timestamp_ns=ahead_ns
            ).status_code
            == 200
        )

        with self._restart(config) as restarted:
            # Present-time sits below the floor the previous life recorded.
            assert signed_request(restarted, validator_key, "GET", STATE_PATH).status_code == 401
            # Above it still passes, so this is the floor refusing — not a wedged daemon.
            assert (
                signed_request(
                    restarted, validator_key, "GET", STATE_PATH, timestamp_ns=ahead_ns + 1
                ).status_code
                == 200
            )


class _ExplodingStore:
    """Stands in for the state store so the handler dies after auth has already passed."""

    @property
    def document(self):
        raise RuntimeError("daemon died mid-request")


class TestWriteOrdering:
    """The nonce is recorded and fsynced before dispatch, never after."""

    def test_crash_during_dispatch_still_blocks_the_replay(
        self, client, config, app, validator_key
    ):
        """Simulates a daemon that dies mid-request.

        Auth passes, then the handler raises, so the request never completes. If the nonce were
        recorded after dispatch, nothing would be written and a restarted daemon would accept the
        same captured request again. This test fails if the ordering is ever inverted.
        """
        headers = sign_headers(validator_key, method="GET", request_target=STATE_PATH)
        app.state.store = _ExplodingStore()

        assert client.get(STATE_PATH, headers=headers).status_code == 500

        with TestClient(create_app(config), raise_server_exceptions=False) as restarted:
            assert restarted.get(STATE_PATH, headers=headers).status_code == 401

    async def test_record_is_durable_before_check_returns(self, state_dir):
        """At the store level: the record is on disk when check_and_record returns, not later."""
        store = ReplayStore(state_dir, skew_seconds=60)
        timestamp = time.time_ns()
        await store.check_and_record("hotkey", "d" * 32, timestamp)

        # A second store reading the same directory is what a restarted daemon sees.
        reloaded = ReplayStore(state_dir, skew_seconds=60)
        with pytest.raises(ReplayRejected):
            await reloaded.check_and_record("hotkey", "d" * 32, timestamp)


class TestStoreUnit:
    async def test_startup_floor_does_not_advance_within_a_lifetime(self, state_dir):
        """The distinction between a startup floor and a monotonic timestamp, asserted directly."""
        store = ReplayStore(state_dir, skew_seconds=60)
        assert store.startup_floor_ns == 0

        await store.check_and_record("hotkey", "e" * 32, time.time_ns())
        assert store.startup_floor_ns == 0, "the floor advanced mid-lifetime — that is monotonicity"

    async def test_window_is_pruned(self, state_dir):
        """The set holds a window of traffic, not a log of it."""
        store = ReplayStore(state_dir, skew_seconds=60)
        now = time.time_ns()

        for index in range(3):
            await store.check_and_record("hotkey", f"{index:032x}", now)

        assert len(store._seen) == 3
        store._prune(now + 120 * 1_000_000_000)
        assert store._seen == {}
