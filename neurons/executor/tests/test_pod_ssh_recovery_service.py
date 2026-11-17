"""DAH-6379: the executor re-runs a restarted rental pod's sshd bootstrap, leaving encrypted pods to the validator."""

from unittest.mock import MagicMock, call

from services.pod_ssh_recovery_service import RESTART_JUPYTER, RESTART_SSHD, handle_event


def _start(name="pod_1", cid="abc", action="start"):
    return {"Type": "container", "Action": action, "id": cid, "Actor": {"ID": cid, "Attributes": {"name": name}}}


def test_a_restarted_pod_gets_its_sshd_bootstrap_and_jupyter_rerun_as_root():
    client = MagicMock()
    client.containers.get.return_value.attrs = {"Mounts": [{"Destination": "/workspace"}]}

    assert handle_event(_start(), client) is True

    client.containers.get.assert_called_once_with("abc")
    assert client.containers.get.return_value.exec_run.call_args_list == [
        call(RESTART_SSHD, detach=True, user="0"),
        call(RESTART_JUPYTER, detach=True, user="0"),
    ]
    assert "[ -f /tmp/lium-ssh-bootstrap.sh ] || exit 0" in RESTART_SSHD[2]
    assert "[ -f /tmp/lium-jupyter-restart.sh ] || exit 0" in RESTART_JUPYTER[2]


def test_an_encrypted_pod_is_left_to_the_validators_mount_then_ssh_recovery():
    client = MagicMock()
    client.containers.get.return_value.attrs = {"Mounts": [{"Destination": "/lium-cipher"}]}

    handle_event(_start(), client)

    client.containers.get.return_value.exec_run.assert_not_called()


def test_only_rental_pod_starts_run_the_sshd_bootstrap():
    client = MagicMock()

    assert handle_event(_start(name="filler_1"), client) is False
    assert handle_event(_start(action="die"), client) is False

    client.containers.get.assert_not_called()


def test_a_failed_sshd_rerun_does_not_raise():
    client = MagicMock()
    client.containers.get.side_effect = RuntimeError("gone")

    assert handle_event(_start(), client) is True


def test_a_reconnect_resumes_from_the_last_event_and_skips_the_replayed_one(monkeypatch):
    import services.pod_ssh_recovery_service as svc

    first = {**_start(cid="a"), "time": 100, "timeNano": 100_000_000_001}
    missed = {**_start(cid="b"), "time": 101, "timeNano": 101_000_000_002}
    sinces, handled = [], []

    def events(decode, since, filters):
        sinces.append(since)
        if len(sinces) == 1:
            yield first
            raise ConnectionError("stream dropped")
        yield first  # the second `since` covers is replayed
        yield missed
        raise KeyboardInterrupt  # ends the test's watcher

    client = MagicMock()
    client.events.side_effect = events
    monkeypatch.setattr(svc.docker, "from_env", lambda: client)
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)
    monkeypatch.setattr(svc.time, "time", lambda: 90.5)  # the watcher's start: its first cursor
    monkeypatch.setattr(svc, "handle_event", lambda e, c: handled.append(e["id"]))

    try:
        svc._watch_forever()
    except KeyboardInterrupt:
        pass

    assert sinces == [90, 100]
    assert handled == ["a", "b"]
