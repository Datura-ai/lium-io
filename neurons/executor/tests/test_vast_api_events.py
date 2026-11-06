import json
import uuid
from unittest.mock import Mock

import requests
from vast_api.config import VastSettings
from vast_api.events import (
    EVENTS_CONFIG_FILE,
    EVENTS_STATE_FILE,
    ContractEventsPoller,
)

CONFIG = {
    "events_token": "tok-test",
    "events_url": "https://backend.example/vast/events",
    "executor_id": "ec8c2436-1039-45aa-a508-bcbe865fbcd8",
}


class FakeStateHost:
    """In-memory stand-in for the HostOps state-file pair (nsenter-backed in prod)."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = files or {}

    def read_state_file(self, name: str) -> str | None:
        return self.files.get(name)

    def write_state_file(self, name: str, content: str) -> None:
        self.files[name] = content


def _poller(files: dict[str, str] | None = None, containers: list[str] | None = None):
    # enrolled machine (vast-uns present) with the events config persisted,
    # unless the caller overrides the state files
    host = FakeStateHost(
        files if files is not None else {
            EVENTS_CONFIG_FILE: json.dumps(CONFIG),
            "numeric_machine_id": "147063\n",
        }
    )
    docker_ops = Mock()
    docker_ops.get_vast_uns.return_value = Mock()
    docker_ops.vast_rental_containers.return_value = containers or []
    return ContractEventsPoller(VastSettings(), host, docker_ops), host, docker_ops


def _response(status_code: int, text: str = "") -> Mock:
    return Mock(status_code=status_code, text=text)


def _state(host: FakeStateHost) -> dict:
    return json.loads(host.files[EVENTS_STATE_FILE])


# --- edge detection ---


def test_new_container_emits_start_event(monkeypatch):
    poller, host, _ = _poller(containers=["C.47368067"])
    post = Mock(return_value=_response(200))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    assert post.call_count == 1
    assert post.call_args.args == (CONFIG["events_url"],)
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer tok-test"}
    event = post.call_args.kwargs["json"]
    assert uuid.UUID(event["event_id"])  # a real uuid4, minted once
    assert event["occurred_at"].endswith("Z")
    assert event["executor_id"] == "ec8c2436-1039-45aa-a508-bcbe865fbcd8"
    assert event["vast_machine_id"] == 147063
    assert event["contract_id"] == "C.47368067"
    assert event["event"] == "start"
    assert _state(host) == {"last_seen": ["C.47368067"], "pending": []}


def test_vanished_container_emits_end_event(monkeypatch):
    poller, host, _ = _poller(containers=[])
    host.files[EVENTS_STATE_FILE] = json.dumps({"last_seen": ["C.111"], "pending": []})
    post = Mock(return_value=_response(200))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    event = post.call_args.kwargs["json"]
    assert event["contract_id"] == "C.111"
    assert event["event"] == "end"
    assert _state(host) == {"last_seen": [], "pending": []}


def test_unchanged_set_emits_nothing(monkeypatch):
    poller, host, _ = _poller(containers=["C.111"])
    unchanged = json.dumps({"last_seen": ["C.111"], "pending": []})
    host.files[EVENTS_STATE_FILE] = unchanged
    post = Mock()
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    post.assert_not_called()
    assert host.files[EVENTS_STATE_FILE] == unchanged


def test_missing_machine_id_is_null_in_event(monkeypatch):
    poller, _, _ = _poller(
        files={EVENTS_CONFIG_FILE: json.dumps(CONFIG)}, containers=["C.111"]
    )
    post = Mock(return_value=_response(200))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    assert post.call_args.kwargs["json"]["vast_machine_id"] is None


# --- spool / delivery ---


def test_failed_delivery_spools_event_and_retries_with_same_id(monkeypatch):
    poller, host, _ = _poller(containers=["C.111"])
    post = Mock(side_effect=requests.ConnectionError("backend down"))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    # last-seen already advanced (no re-emit on restart), event held in the spool
    state = _state(host)
    assert state["last_seen"] == ["C.111"]
    assert len(state["pending"]) == 1
    event_id = state["pending"][0]["event_id"]

    post_ok = Mock(return_value=_response(200))
    monkeypatch.setattr("vast_api.events.requests.post", post_ok)
    poller.poll_once()

    # retried with the SAME event_id (backend dedup key), then drained
    assert post_ok.call_args.kwargs["json"]["event_id"] == event_id
    assert _state(host)["pending"] == []


def test_5xx_keeps_event_in_spool(monkeypatch):
    poller, host, _ = _poller(containers=["C.111"])
    post = Mock(return_value=_response(503, "unavailable"))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    assert len(_state(host)["pending"]) == 1


def test_429_keeps_event_in_spool(monkeypatch):
    poller, host, _ = _poller(containers=["C.111"])
    post = Mock(return_value=_response(429, "slow down"))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    # rate limiting is temporary: the edge must survive until the backend takes it
    assert len(_state(host)["pending"]) == 1


def test_4xx_drops_event(monkeypatch):
    poller, host, _ = _poller(containers=["C.111"])
    post = Mock(return_value=_response(401, "bad token"))
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    # misconfiguration must not loop forever: dropped, last-seen still advanced
    assert _state(host) == {"last_seen": ["C.111"], "pending": []}


# --- skip conditions ---


def test_skips_when_vast_uns_absent(monkeypatch):
    poller, _, docker_ops = _poller()
    docker_ops.get_vast_uns.return_value = None
    post = Mock()
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    docker_ops.vast_rental_containers.assert_not_called()
    post.assert_not_called()


def test_skips_without_events_config(monkeypatch):
    poller, _, docker_ops = _poller(files={}, containers=["C.111"])
    post = Mock()
    monkeypatch.setattr("vast_api.events.requests.post", post)

    poller.poll_once()

    docker_ops.vast_rental_containers.assert_not_called()
    post.assert_not_called()
