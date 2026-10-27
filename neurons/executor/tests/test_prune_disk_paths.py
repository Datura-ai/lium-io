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


def _run(monkeypatch, tmp_path, driver_status, config=None, full_dir="var/lib/containerd", files=None, remove_ok=True):
    for d in ("data/docker", "var/lib/containerd", "nvme/containerd", "etc/containerd"):
        (tmp_path / d).mkdir(parents=True)
    if config is not None:
        (tmp_path / "etc/containerd/config.toml").write_text(config)
    for name, text in (files or {}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    full = str(tmp_path / full_dir)
    monkeypatch.setattr(cts, "HOST_ROOT", str(tmp_path))
    monkeypatch.setattr(cts.psutil, "disk_usage", lambda p: FULL if p == full else ROOMY)
    monkeypatch.setattr(cts, "rental_activity", lambda c: None)
    monkeypatch.setattr(cts, "_remove_ref", lambda c, ref: remove_ok)
    return cts._prune_unused_images_when_disk_low(_client(driver_status), set())


def test_full_containerd_store_prunes_though_data_root_is_roomy(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER) == 1


def test_containerd_root_moved_in_config_is_measured(monkeypatch, tmp_path):
    config = 'version = 3\n# root = "/ignored"\nroot = "/nvme/containerd"\n[plugins]\nroot = "/not/top"\n'
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER, config, full_dir="nvme/containerd") == 1


def test_containerd_root_not_measured_without_the_containerd_store(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, [["Backing Filesystem", "extfs"]]) == 0


def test_containerd_root_set_in_a_glob_import_is_measured(monkeypatch, tmp_path):
    config = 'version = 3\nimports = [\n  "/etc/containerd/conf.d/*.toml",\n]\n[plugins]\n'
    files = {"etc/containerd/conf.d/10-root.toml": 'root = "/nvme/containerd"\n'}
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER, config, "nvme/containerd", files) == 1


def test_containerd_root_set_in_a_relative_nested_import_is_measured(monkeypatch, tmp_path):
    config = 'imports = ["extra.toml"]\n'
    files = {
        "etc/containerd/extra.toml": "imports = ['/etc/containerd/more.toml']\n",
        "etc/containerd/more.toml": 'root = "/nvme/containerd"\n',
    }
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER, config, "nvme/containerd", files) == 1


def test_full_disk_with_nothing_removable_is_reported(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(cts.logger, "warning", warnings.append)
    assert _run(monkeypatch, tmp_path, SNAPSHOTTER, remove_ok=False) == 0
    assert len(warnings) == 1 and "var/lib/containerd" in warnings[0] and "stays full" in warnings[0]
