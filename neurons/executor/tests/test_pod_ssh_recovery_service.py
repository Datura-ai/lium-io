"""DAH-6379: the executor re-runs a restarted rental pod's sshd bootstrap, leaving encrypted pods to the validator."""

from unittest.mock import MagicMock

from services.pod_ssh_recovery_service import RESTART_SSHD, handle_event


def _start(name="pod_1", cid="abc", action="start"):
    return {"Type": "container", "Action": action, "id": cid, "Actor": {"ID": cid, "Attributes": {"name": name}}}


def test_a_restarted_pod_gets_its_sshd_bootstrap_rerun_as_root():
    client = MagicMock()
    client.containers.get.return_value.attrs = {"Mounts": [{"Destination": "/workspace"}]}

    assert handle_event(_start(), client) is True

    client.containers.get.assert_called_once_with("abc")
    client.containers.get.return_value.exec_run.assert_called_once_with(RESTART_SSHD, detach=True, user="0")
    assert "[ -f /tmp/lium-ssh-bootstrap.sh ] || exit 0" in RESTART_SSHD[2]


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
