"""DAH-6379: the executor re-runs a restarted rental pod's sshd bootstrap, leaving encrypted pods to the validator."""

import re
from unittest.mock import MagicMock, call

from services.pod_ssh_recovery_service import RESTART_JUPYTER, RESTART_SSHD, handle_event, sweep_running_pods


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


def test_a_live_jupyter_is_left_alone_and_the_guard_never_matches_its_own_command():
    pattern = re.search(r"grep -qs '([^']+)'", RESTART_JUPYTER[2]).group(1)

    assert re.search(pattern, "/opt/jupyter-env/bin/jupyter\0lab\0--port=8888")
    assert not re.search(pattern, RESTART_JUPYTER[2])
    assert RESTART_JUPYTER[2].index("&& exit 0") < RESTART_JUPYTER[2].index("exec sh")


def test_the_startup_sweep_reruns_the_bootstrap_on_pods_started_before_the_watcher():
    pod, filler = MagicMock(id="p1"), MagicMock(id="f1")
    pod.name, filler.name = "pod_1", "filler_pod_1"
    client = MagicMock()
    client.containers.list.return_value = [pod, filler]
    client.containers.get.return_value.attrs = {"Mounts": []}

    sweep_running_pods(client)

    client.containers.list.assert_called_once_with(filters={"name": "pod_", "status": "running"})
    client.containers.get.assert_called_once_with("p1")


def test_the_watcher_sweeps_once_not_on_every_reconnect(monkeypatch):
    import services.pod_ssh_recovery_service as svc

    calls, swept = [], []

    def events(decode, since, filters):
        calls.append(since)
        if len(calls) == 1:
            raise ConnectionError("stream dropped")
        raise KeyboardInterrupt

    client = MagicMock()
    client.events.side_effect = events
    monkeypatch.setattr(svc.docker, "from_env", lambda: client)
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)
    monkeypatch.setattr(svc, "sweep_running_pods", lambda c: swept.append(c))

    try:
        svc._watch_forever()
    except KeyboardInterrupt:
        pass

    assert len(calls) == 2 and swept == [client]
