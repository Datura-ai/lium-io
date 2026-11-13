"""DAH-6379: the executor sets memory.oom.group on every rental pod start, for both docker cgroup drivers."""

from services.pod_oom_group_service import handle_event, set_oom_group


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
