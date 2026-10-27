"""The low-disk image prune measures every filesystem the image store can sit on."""

from collections import namedtuple
from unittest.mock import MagicMock

from services import cache_template_service as cts

Usage = namedtuple("Usage", "free percent")
ROOMY = Usage(free=500 * 1024**3, percent=10)
FULL = Usage(free=1024**3, percent=99)
SNAPSHOTTER = [["driver-type", "io.containerd.snapshotter.v1"]]


def _client(driver_status):
    client = MagicMock()
    client.info.return_value = {"DockerRootDir": "/data/docker", "DriverStatus": driver_status}
    client.containers.list.return_value = []
    image = MagicMock(id="sha256:old", tags=["old:1"])
    client.images.list.return_value = [image]
    return client


def _run(monkeypatch, tmp_path, driver_status, config=None, full_dir="var/lib/containerd"):
    for d in ("data/docker", "var/lib/containerd", "nvme/containerd"):
        (tmp_path / d).mkdir(parents=True)
    if config is not None:
        (tmp_path / "etc/containerd").mkdir(parents=True)
        (tmp_path / "etc/containerd/config.toml").write_text(config)
    full = str(tmp_path / full_dir)
    monkeypatch.setattr(cts, "HOST_ROOT", str(tmp_path))
    monkeypatch.setattr(cts.psutil, "disk_usage", lambda p: FULL if p == full else ROOMY)
    monkeypatch.setattr(cts, "rental_activity", lambda c: None)
    monkeypatch.setattr(cts, "_remove_ref", lambda c, ref: True)
    return cts._prune_unused_images_when_disk_low(_client(driver_status), set())


def test_full_containerd_store_prunes_though_data_root_is_roomy(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER) == 1


def test_containerd_root_moved_in_config_is_measured(monkeypatch, tmp_path):
    config = 'version = 3\n# root = "/ignored"\nroot = "/nvme/containerd"\n[plugins]\nroot = "/not/top"\n'
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER, config, full_dir="nvme/containerd") == 1


def test_containerd_root_not_measured_without_the_containerd_store(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, [["Backing Filesystem", "extfs"]]) == 0
