"""DAH-6379: the executor sets memory.oom.group on every rental pod start, for both docker cgroup drivers."""

from unittest.mock import MagicMock

from services.pod_oom_group_service import RESTART_SSHD, handle_event, set_oom_group


def _dirs(tmp_path):
    return (str(tmp_path / "system.slice/docker-{id}.scope"), str(tmp_path / "docker/{id}"))


def _scope(tmp_path, rel, value="0"):
    d = tmp_path / rel
    d.mkdir(parents=True)
    (d / "memory.oom.group").write_text(value + "\n")
    return d / "memory.oom.group"


def _start(name="pod_1", cid="abc", action="start"):
    return {"Type": "container", "Action": action, "id": cid, "Actor": {"ID": cid, "Attributes": {"name": name}}}


def test_a_restarted_pod_gets_oom_group_back_on_the_cgroupfs_layout(tmp_path):
    f = _scope(tmp_path, "docker/abc")

    assert handle_event(_start(), _dirs(tmp_path)) == "set"
    assert f.read_text().strip() == "1"


def test_the_systemd_scope_layout_is_set_too(tmp_path):
    f = _scope(tmp_path, "system.slice/docker-abc.scope")

    assert set_oom_group("abc", _dirs(tmp_path)) == "set"
    assert f.read_text().strip() == "1"


def test_an_already_set_pod_is_left_alone(tmp_path):
    _scope(tmp_path, "docker/abc", "1")

    assert set_oom_group("abc", _dirs(tmp_path)) == "already"


def test_a_pod_without_a_cgroup_v2_scope_is_reported(tmp_path):
    assert set_oom_group("abc", _dirs(tmp_path)) == "no_cgroup_v2_scope"


def test_only_rental_pod_starts_are_touched(tmp_path):
    f = _scope(tmp_path, "docker/abc")

    assert handle_event(_start(name="filler_1"), _dirs(tmp_path)) is None
    assert handle_event(_start(action="die"), _dirs(tmp_path)) is None
    assert f.read_text().strip() == "0"


def test_a_restarted_pod_gets_its_sshd_bootstrap_rerun_as_root(tmp_path):
    client = MagicMock()
    client.containers.get.return_value.attrs = {"Mounts": [{"Destination": "/workspace"}]}

    handle_event(_start(), _dirs(tmp_path), client=client)

    client.containers.get.assert_called_once_with("abc")
    client.containers.get.return_value.exec_run.assert_called_once_with(RESTART_SSHD, detach=True, user="0")
    assert "[ -f /tmp/lium-ssh-bootstrap.sh ] || exit 0" in RESTART_SSHD[2]


def test_an_encrypted_pod_is_left_to_the_validators_mount_then_ssh_recovery(tmp_path):
    client = MagicMock()
    client.containers.get.return_value.attrs = {"Mounts": [{"Destination": "/lium-cipher"}]}

    handle_event(_start(), _dirs(tmp_path), client=client)

    client.containers.get.return_value.exec_run.assert_not_called()


def test_a_filler_start_runs_no_sshd_bootstrap(tmp_path):
    client = MagicMock()

    handle_event(_start(name="filler_1"), _dirs(tmp_path), client=client)

    client.containers.get.assert_not_called()


def test_a_failed_sshd_rerun_does_not_raise(tmp_path):
    client = MagicMock()
    client.containers.get.side_effect = RuntimeError("gone")
    _scope(tmp_path, "docker/abc")

    assert handle_event(_start(), _dirs(tmp_path), client=client) == "set"
