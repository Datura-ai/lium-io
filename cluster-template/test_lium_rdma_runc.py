"""DAH-2620 — the nested-container runtime wrapper injects the fabric and nothing more."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("lium-rdma-runc.py")
spec = importlib.util.spec_from_file_location("lium_rdma_runc", MODULE_PATH)
lium_rdma_runc = importlib.util.module_from_spec(spec)
sys.modules["lium_rdma_runc"] = lium_rdma_runc
spec.loader.exec_module(lium_rdma_runc)


@pytest.fixture
def fake_verbs(tmp_path, monkeypatch) -> Path:
    """A /dev/infiniband holding the safe nodes plus the two that must never be forwarded."""
    verbs_dir = tmp_path / "infiniband"
    verbs_dir.mkdir()
    for name in ("uverbs0", "uverbs1", "rdma_cm", "issm0", "umad0"):
        (verbs_dir / name).write_bytes(b"")
    monkeypatch.setattr(lium_rdma_runc, "VERBS_DIR", str(verbs_dir))
    return verbs_dir


def _spec() -> dict:
    return {
        "process": {
            "capabilities": {"bounding": ["CAP_NET_ADMIN"], "effective": ["CAP_NET_ADMIN"]},
            "rlimits": [{"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024}],
        }
    }


def test_the_verbs_devices_are_added(fake_verbs) -> None:
    # Act
    injected: dict = lium_rdma_runc.inject(_spec())

    # Assert
    paths: list[str] = [device["path"] for device in injected["linux"]["devices"]]
    assert [Path(path).name for path in paths] == ["rdma_cm", "uverbs0", "uverbs1"]


def test_the_subnet_manager_devices_are_never_added(fake_verbs) -> None:
    # Act
    injected: dict = lium_rdma_runc.inject(_spec())

    # Assert
    names: list[str] = [Path(device["path"]).name for device in injected["linux"]["devices"]]
    assert "issm0" not in names
    assert "umad0" not in names


def test_every_added_device_is_allowed_by_the_cgroup(fake_verbs) -> None:
    # Act
    injected: dict = lium_rdma_runc.inject(_spec())

    # Assert
    devices = injected["linux"]["devices"]
    allowed = injected["linux"]["resources"]["devices"]
    assert len(allowed) == len(devices)
    assert {(rule["major"], rule["minor"]) for rule in allowed} == {
        (device["major"], device["minor"]) for device in devices
    }


def test_memory_pinning_is_granted(fake_verbs) -> None:
    # Act
    process: dict = lium_rdma_runc.inject(_spec())["process"]

    # Assert
    assert "CAP_IPC_LOCK" in process["capabilities"]["bounding"]
    assert "CAP_IPC_LOCK" in process["capabilities"]["effective"]
    memlock = [limit for limit in process["rlimits"] if limit["type"] == "RLIMIT_MEMLOCK"]
    assert memlock == [
        {"type": "RLIMIT_MEMLOCK", "hard": lium_rdma_runc.RLIMIT_INFINITY, "soft": lium_rdma_runc.RLIMIT_INFINITY}
    ]
    # the limits the container already carried are untouched
    assert any(limit["type"] == "RLIMIT_NOFILE" for limit in process["rlimits"])


def test_a_host_without_a_fabric_changes_no_devices(tmp_path, monkeypatch) -> None:
    # Arrange — the same image also runs on nodes that have no InfiniBand at all
    monkeypatch.setattr(lium_rdma_runc, "VERBS_DIR", str(tmp_path / "absent"))

    # Act
    injected: dict = lium_rdma_runc.inject(_spec())

    # Assert
    assert injected["linux"]["devices"] == []


def test_running_twice_does_not_duplicate_a_device(fake_verbs) -> None:
    # Arrange — docker can retry a create against the same bundle
    once: dict = lium_rdma_runc.inject(_spec())

    # Act
    twice: dict = lium_rdma_runc.inject(json.loads(json.dumps(once)))

    # Assert
    assert len(twice["linux"]["devices"]) == len(once["linux"]["devices"])
    assert twice["process"]["capabilities"]["bounding"].count("CAP_IPC_LOCK") == 1


def test_the_bundle_flag_is_read_in_both_forms() -> None:
    assert lium_rdma_runc._bundle_path(["create", "--bundle", "/run/x", "id"]) == "/run/x"
    assert lium_rdma_runc._bundle_path(["create", "-b", "/run/y", "id"]) == "/run/y"
    assert lium_rdma_runc._bundle_path(["state", "id"]) is None
