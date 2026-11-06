"""The scrape reports what filled the disk: image layers, container writable layers and volumes.

Docker's own /system/df only accounts for the `local` volume driver and is blind to the vloopback
volumes fillers and pods actually run on, so the volume figure is assembled from both sources.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the disk
helpers are compiled out of the source and executed on their own, the same reason
test_scrape_encryption_key_order.py parses rather than imports.
"""

import ast
import glob
import http.client
import io
import json
import os
import re
import shutil
import socket
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

DISK_HELPERS = {
    "DOCKER_SOCKET_PATH",
    "DOCKER_API_TIMEOUT_SECONDS",
    "VLOOPBACK_DRIVER_PREFIX",
    "VLOOPBACK_HOST_DATA_DIR_NAMES",
    "UnixSocketHTTPConnection",
    "docker_api_get",
    "get_vloopback_volume_bytes",
    "mounted_fs_used_bytes",
    "host_mount_points",
    "get_dmcrypt_volume_bytes",
    "get_container_log_bytes",
    "get_host_disk_usage",
    "get_docker_disk_usage",
    "cache_volume_bytes",
    "ENGY_CACHE_VOLUME_PREFIX",
    "FILLER_VOLUME_LABEL_KEY",
    "FILLER_VOLUME_LABEL_VALUE",
    "FILLER_CONTAINER_PREFIX",
    "release_filler_cache",
    "disk_root_dir",
    "containerd_store_root",
    "host_realpath",
    "CONTAINERD_ROOT_UNKNOWN",
    "on_one_filesystem",
    "hard_disk_reading",
    "report_tighter_store",
    "get_content_store_usage",
    "get_docker_root_disk_type",
    "HOST_MOUNTS_PATH",
}


_CTRD_ADDRESS = "/run/containerd/containerd.sock"
_OVERLAYFS = "io.containerd.snapshotter.v1.overlayfs"
_CONTENT_PLUGIN = "io.containerd.content.v1.content"


def _ctr(exports: dict):
    """`ctr plugins ls -d` of the running containerd on Docker's address, whose plugins export these roots."""

    def run_cmd(cmd):
        if f"ctr --address {_CTRD_ADDRESS} plugins ls -d" not in cmd:
            raise RuntimeError(f"not on a host: {cmd}")
        blocks = ["\tType:          io.containerd.grpc.v1\n\tID:            cri\n\tRequires:\n\t               io.containerd.event.v1\n"]
        for plugin, root in exports.items():
            kind, name = plugin.rsplit(".", 1)
            blocks.append(f"\tType:          {kind}\n\tID:            {name}\n\tPlatforms:     linux/amd64\n"
                          f"\tExports:       \n\t               root      {root}\n")
        return "\n".join(blocks)

    return run_cmd


def _running_root(root: str) -> dict:
    return {_OVERLAYFS: f"{root}/{_OVERLAYFS}", _CONTENT_PLUGIN: f"{root}/{_CONTENT_PLUGIN}"}


@pytest.fixture
def scrape() -> dict:
    """The disk helpers, executed in a namespace of their own."""
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        DISK_HELPERS,
        {"glob": glob, "http": http, "json": json, "os": os, "re": re, "shutil": shutil, "socket": socket,
         "run_cmd": _ctr(_running_root("/var/lib/containerd"))},
    )
    return namespace


def _stub_docker_api(scrape: dict, responses: dict) -> None:
    scrape["docker_api_get"] = lambda path: responses[path]


def test_docker_disk_usage_splits_df_by_kind(scrape: dict) -> None:
    # Arrange
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 45079943748,
                "Containers": [{"SizeRw": 282628096}, {"SizeRw": 1024}],
                "Volumes": [{"UsageData": {"Size": 5454000000}}],
                # a custom build leaves layers here; docker reports them apart from LayersSize
                "BuildCache": [{"Size": 2000000000}, {"Size": -1}],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert
    assert usage == {
        "hard_disk_images": (45079943748 + 2000000000) // 1024,
        "hard_disk_containers": (282628096 + 1024) // 1024,
        "hard_disk_volumes": 5454000000 // 1024,
        "filler_released_kb": 0,
        "engy_released_kb": 0,
        "hard_disk_engy_cache_used_gb": 0,
        "hard_disk_by_container": {},
        "hard_disk_by_volume": {},
    }


def test_docker_disk_usage_reports_each_pods_share(scrape: dict, monkeypatch) -> None:
    # Arrange — DAH-5059: a pod with a named vloopback workspace, a dind pod whose anonymous
    # /var/lib/docker volume has no name the backend knows, and a stopped container with no size.
    dind_volume = "ab" * 32
    blocks_by_path = {"/proc/1/root/var/lib/docker/vloopback-v2/volume_pod1": 2048}
    monkeypatch.setitem(scrape, "glob", type("_Glob", (), {"glob": staticmethod(lambda pattern: [])}))
    monkeypatch.setitem(
        scrape,
        "os",
        type(
            "_Os",
            (),
            {
                "path": type("_Path", (), {"join": staticmethod(os.path.join), "isdir": staticmethod(lambda path: True)}),
                "stat": staticmethod(lambda path: type("_Stat", (), {"st_blocks": blocks_by_path[path]})),
            },
        ),
    )
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 0,
                "Containers": [
                    {
                        "Names": ["/container_pod1"],
                        "SizeRw": 4096,
                        "Mounts": [{"Type": "volume", "Name": "volume_pod1", "Destination": "/root"}],
                    },
                    {
                        "Names": ["/container_dind"],
                        "SizeRw": 1024,
                        "Mounts": [
                            {"Type": "volume", "Name": dind_volume, "Destination": "/var/lib/docker"},
                            {"Type": "bind", "Source": "/tmp", "Destination": "/tmp"},
                        ],
                    },
                    {"Names": ["/stopped"]},
                ],
                "Volumes": [
                    {"Name": "volume_pod1", "UsageData": {"Size": -1}},
                    {"Name": dind_volume, "UsageData": {"Size": 10240}},
                ],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": [{"Name": "volume_pod1", "Driver": "vloopback-v2"}]},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert — kB; the vloopback volume by its real blocks, the dind volume under its container
    assert usage["hard_disk_by_container"] == {"container_pod1": 4, "container_dind": 11, "stopped": 0}
    assert usage["hard_disk_by_volume"] == {"volume_pod1": 2048 * 512 // 1024, dind_volume: 10}
    assert usage["hard_disk_containers"] == 5
    assert usage["hard_disk_volumes"] == (10240 + 2048 * 512) // 1024


GIB = 1024**3
POD_ID = "0f5e6a3c-1b2d-4e5f-8a9b-0c1d2e3f4a5b"
DMCRYPT = {"Driver": "local", "Options": {"type": "ext4", "device": f"/dev/mapper/lium-{POD_ID}"}}


@pytest.mark.parametrize(
    "volume, df_size, backing_bytes, mounted_used, by_volume, total",
    [
        # plain local volume: /system/df walks its files, nothing else to read
        ({"Driver": "local"}, 3 * GIB, None, None, 3 * GIB, 3 * GIB),
        # preallocated vloopback (partial rental): the file holds 600 GiB, its filesystem 1 GiB
        ({"Driver": "vloopback:v2"}, -1, 600 * GIB, 1 * GIB, 1 * GIB, 600 * GIB),
        # sparse vloopback after writing 50 GiB and deleting 48: the file keeps the blocks
        ({"Driver": "vloopback:v2"}, -1, 50 * GIB, 2 * GIB, 2 * GIB, 50 * GIB),
        # a stopped pod's volume is not mounted: its file is all there is to read
        ({"Driver": "vloopback:v2"}, -1, 600 * GIB, None, 600 * GIB, 600 * GIB),
        # dm-crypt (lium-io#1986), preallocated, sparse and unmounted: /system/df leaves it out
        (DMCRYPT, None, 600 * GIB, 1 * GIB, 1 * GIB, 600 * GIB),
        (DMCRYPT, None, 50 * GIB, 2 * GIB, 2 * GIB, 50 * GIB),
        (DMCRYPT, None, 600 * GIB, None, 600 * GIB, 600 * GIB),
    ],
    ids=[
        "local",
        "vloopback-prealloc-parent-absent",
        "vloopback-sparse-deleted",
        "vloopback-unmounted",
        "dmcrypt-prealloc-parent-absent",
        "dmcrypt-sparse",
        "dmcrypt-unmounted",
    ],
)
def test_by_volume_reports_what_the_pods_filesystem_holds(
    scrape: dict, monkeypatch, volume, df_size, backing_bytes, mounted_used, by_volume, total
) -> None:
    # Arrange — review on #1989: a pod's share is its mounted filesystem's used bytes; the disk
    # total stays what the backing file takes
    root = "/proc/1/root/var/lib/docker"
    dmcrypt = "Options" in volume
    mountpoint = "/var/lib/docker/volumes/volume_pod/_data" if dmcrypt else "/mnt/volume_pod"
    host_mount = mountpoint if dmcrypt else "/var/lib/docker/plugins/abc/propagated-mount/volume_pod"
    backing = f"{root}/lium-crypt/{POD_ID}.img" if dmcrypt else f"{root}/vloopback-v2/volume_pod"

    def stat(path):
        if path != backing or backing_bytes is None:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_blocks=backing_bytes // 512)

    def statvfs(path):
        assert path == f"/proc/1/root{host_mount}"
        return SimpleNamespace(f_blocks=1000 * GIB // 4096, f_bfree=(1000 * GIB - mounted_used) // 4096, f_frsize=4096)

    path = SimpleNamespace(
        join=os.path.join,
        isdir=lambda candidate: candidate == f"{root}/vloopback-v2",
        # F-2: the host-side parent is absent in the executor container, so py3.11's ismount (realpath
        # of the parent through /proc/1/root) says False for every mount; the host's mountinfo decides
        ismount=lambda candidate: False,
    )
    monkeypatch.setitem(scrape, "os", SimpleNamespace(path=path, stat=stat, statvfs=statvfs))
    glob_hit = [f"/proc/1/root{host_mount}"]
    monkeypatch.setitem(scrape, "glob", SimpleNamespace(glob=lambda pattern: glob_hit if "propagated-mount" in pattern else []))
    mountinfo = [
        "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n",
        # an escaped neighbour (a space in its path) must not hide or fake the pod's mount
        "30 22 7:3 / /mnt/my\\040disk rw shared:9 - ext4 /dev/loop3 rw\n",
    ] + ([f"31 22 7:4 / {host_mount} rw,relatime shared:10 - ext4 /dev/loop4 rw\n"] if mounted_used is not None else [])
    monkeypatch.setitem(scrape, "open", lambda path: io.StringIO("".join(mountinfo)) if path == "/proc/1/mountinfo" else None)
    df_volumes = [] if df_size is None else [{"Name": "volume_pod", "UsageData": {"Size": df_size}}]
    _stub_docker_api(
        scrape,
        {
            "/system/df": {"LayersSize": 0, "Containers": [], "Volumes": df_volumes},
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": [{"Name": "volume_pod", "Mountpoint": mountpoint, **volume}]},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert
    assert usage["hard_disk_by_volume"] == {"volume_pod": by_volume // 1024}
    assert usage["hard_disk_volumes"] == total // 1024


def test_docker_disk_usage_ignores_unknown_sizes(scrape: dict) -> None:
    # Arrange — docker reports -1 for a size it has not computed
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 0,
                "Containers": [{"SizeRw": -1}, {}],
                "Volumes": [{"UsageData": {"Size": -1}}, {"UsageData": None}],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert
    assert usage == {
        "hard_disk_images": 0,
        "hard_disk_containers": 0,
        "hard_disk_volumes": 0,
        "filler_released_kb": 0,
        "engy_released_kb": 0,
        "hard_disk_engy_cache_used_gb": 0,
        "hard_disk_by_container": {},
        "hard_disk_by_volume": {},
    }


def test_vloopback_volumes_are_added_to_the_local_driver_total(scrape: dict, monkeypatch) -> None:
    # Arrange — a preallocated volume holding its whole declared size, and a sparse one holding
    # almost nothing; only the blocks each actually occupies count.
    blocks_by_path = {
        "/proc/1/root/var/lib/docker/plugins/abc/rootfs/var/lib/docker/loopback/volume_full": 1953136,
        "/proc/1/root/var/lib/docker/plugins/abc/rootfs/var/lib/docker/loopback/volume_sparse": 7488,
    }
    monkeypatch.setitem(
        scrape,
        "glob",
        # the container-log walk globs the same root; only the volume pattern has matches here
        type("_Glob", (), {"glob": staticmethod(lambda pattern: [] if pattern.endswith("*.log*") else sorted({path.rsplit("/", 1)[0] for path in blocks_by_path}))}),
    )
    monkeypatch.setitem(
        scrape,
        "os",
        type(
            "_Os",
            (),
            {
                "path": os.path,
                "stat": staticmethod(lambda path: type("_Stat", (), {"st_blocks": blocks_by_path[path]})),
            },
        ),
    )
    _stub_docker_api(
        scrape,
        {
            "/system/df": {"LayersSize": 0, "Containers": [], "Volumes": [{"UsageData": {"Size": 1024}}]},
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {
                "Volumes": [
                    {"Name": "volume_full", "Driver": "vloopback"},
                    {"Name": "volume_sparse", "Driver": "vloopback"},
                    {"Name": "some_local_volume", "Driver": "local"},
                ]
            },
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert
    assert usage["hard_disk_volumes"] == (1024 + (1953136 + 7488) * 512) // 1024


def test_vloopback_volume_missing_its_backing_file_is_skipped(scrape: dict, monkeypatch) -> None:
    # Arrange — a volume the plugin has not materialised must not abort the whole breakdown
    _stub_docker_api(scrape, {"/volumes": {"Volumes": [{"Name": "volume_gone", "Driver": "vloopback"}]}})
    monkeypatch.setitem(
        scrape,
        "glob",
        type("_Glob", (), {"glob": staticmethod(lambda pattern: ["/loopback"])}),
    )
    monkeypatch.setitem(
        scrape,
        "os",
        type(
            "_Os",
            (),
            {
                "path": os.path,
                "stat": staticmethod(lambda path: (_ for _ in ()).throw(FileNotFoundError(path))),
            },
        ),
    )

    # Act
    total = scrape["get_vloopback_volume_bytes"]("/var/lib/docker")

    # Assert
    assert total == 0


def test_missing_plugin_data_dir_raises_instead_of_reporting_zero(scrape: dict, monkeypatch) -> None:
    # Arrange — the glob finds no plugin rootfs at all, so every volume would be counted as 0.
    # That is a structural miss, not a single unmaterialised volume: reporting 0 TB of volumes
    # reads as "the disk is free", so it has to surface as an error key instead.
    _stub_docker_api(scrape, {"/volumes": {"Volumes": [{"Name": "volume_full", "Driver": "vloopback"}]}})
    monkeypatch.setitem(
        scrape,
        "glob",
        type("_Glob", (), {"glob": staticmethod(lambda pattern: [])}),
    )

    # Act / Assert
    with pytest.raises(RuntimeError):
        scrape["get_vloopback_volume_bytes"]("/var/lib/docker")


@pytest.mark.parametrize("old_blocks, total_blocks", [(1953136, 1953136 + 7488), (None, 7488)])
def test_vloopback_volume_bytes_counts_the_v2_and_v3_host_data_dirs_beside_the_old_plugin_rootfs(
    scrape: dict, monkeypatch, old_blocks, total_blocks
) -> None:
    # the old plugin's files sit in its rootfs (found by the glob), v2's in a host dir under DockerRootDir;
    # a node with v2 only is a fresh one, not a structural miss
    old_dir = "/proc/1/root/var/lib/docker/plugins/abc/rootfs/var/lib/docker/loopback"
    v2_dir = "/proc/1/root/var/lib/docker/vloopback-v2"
    v3_dir = "/proc/1/root/var/lib/docker/vloopback-v3"
    blocks = {f"{v2_dir}/volume_v2": 7488, f"{v3_dir}/volume_v3": 96} | (
        {f"{old_dir}/volume_old": old_blocks} if old_blocks else {}
    )

    def stat(path):
        if path not in blocks:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_blocks=blocks[path])

    monkeypatch.setitem(scrape, "glob", SimpleNamespace(glob=lambda pattern: [old_dir] if old_blocks else []))
    path = SimpleNamespace(join=os.path.join, isdir=lambda candidate: candidate in (v2_dir, v3_dir))
    monkeypatch.setitem(scrape, "os", SimpleNamespace(path=path, stat=stat))
    volumes = [
        {"Name": "volume_old", "Driver": "vloopback:latest"},
        {"Name": "volume_v2", "Driver": "vloopback:v2"},
        {"Name": "volume_v3", "Driver": "vloopback:v3"},
    ]
    _stub_docker_api(scrape, {"/volumes": {"Volumes": volumes}})

    assert scrape["get_vloopback_volume_bytes"]("/var/lib/docker") == (total_blocks + 96) * 512


def _obfuscation_tables() -> tuple[set[str], set[str]]:
    """The two independent registries a scrape key has to appear in.

    ORIGINAL_KEYS renames the key back to its final shape on the validator side;
    generate_key_mappings() is what actually obfuscates it in the shipped binary. A key missing
    from the first arrives under its raw scrape name and is then stripped by the backend model;
    missing from the second it ships in the clear.
    """
    service = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())

    original_keys: set[str] = set()
    mapped_keys: set[str] = set()
    for node in ast.walk(service):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ORIGINAL_KEYS":
            original_keys = set(ast.literal_eval(node.value))
        if isinstance(node, ast.FunctionDef) and node.name == "generate_key_mappings":
            mapped_keys = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }

    assert original_keys, "ORIGINAL_KEYS not found"
    assert mapped_keys, "generate_key_mappings not found"
    return original_keys, mapped_keys


def test_every_hard_disk_key_is_registered_in_both_obfuscation_tables() -> None:
    # Arrange
    scrape_source = ast.parse((SRC / "miner_jobs" / "machine_scrape.py").read_text())
    emitted = {
        node.value
        for node in ast.walk(scrape_source)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("hard_disk_")
    }
    original_keys, mapped_keys = _obfuscation_tables()

    # Act / Assert
    assert emitted, "the scrape emits no hard_disk_* keys — the parse is looking in the wrong place"
    assert emitted <= original_keys, f"missing from ORIGINAL_KEYS: {sorted(emitted - original_keys)}"
    assert emitted <= mapped_keys, f"missing from generate_key_mappings: {sorted(emitted - mapped_keys)}"


def test_docker_api_rejects_a_non_ok_response(scrape: dict, monkeypatch) -> None:
    # Arrange
    class _Response:
        status = 500

        def read(self):
            return b"boom"

    class _Conn:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    monkeypatch.setitem(scrape, "UnixSocketHTTPConnection", _Conn)

    # Act / Assert
    with pytest.raises(RuntimeError, match="HTTP 500"):
        scrape["docker_api_get"]("/system/df")


def test_container_json_logs_count_as_docker_not_as_provider_data(scrape: dict[str, object], monkeypatch) -> None:
    # Arrange — review ask (PR #1245): `SizeRw` is the writable layer only, json logs sit outside
    # it, and nothing rotates them on an executor. Unsubtracted, a chatty renter pod reads as the
    # provider's own data and the DAH-2734 gate zeroes an honest machine.
    log_blocks = {
        "/proc/1/root/var/lib/docker/containers/abc/abc-json.log": 41943040,
        # a host with log-opts keeps rotated copies; they hold real bytes too
        "/proc/1/root/var/lib/docker/containers/abc/abc-json.log.1": 20971520,
    }
    monkeypatch.setitem(
        scrape,
        "glob",
        type("_Glob", (), {"glob": staticmethod(lambda pattern: list(blocks_for(pattern)))}),
    )

    def blocks_for(pattern):
        return log_blocks if pattern.endswith("*.log*") else []

    monkeypatch.setitem(
        scrape,
        "os",
        type(
            "_Os",
            (),
            {
                "path": os.path,
                "stat": staticmethod(lambda path: type("_Stat", (), {"st_blocks": log_blocks[path]})),
            },
        ),
    )
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 0,
                "Containers": [{"SizeRw": 1024}],
                "Volumes": [],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert
    assert usage["hard_disk_containers"] == (1024 + (41943040 + 20971520) * 512) // 1024


def test_a_host_without_json_logs_reports_zero(scrape: dict[str, object], monkeypatch) -> None:
    # Arrange — the journald log driver writes no json log at all
    monkeypatch.setitem(scrape, "glob", type("_Glob", (), {"glob": staticmethod(lambda pattern: [])}))

    # Act / Assert
    assert scrape["get_container_log_bytes"]("/var/lib/docker") == 0


def _stub_filesystem(scrape: dict, monkeypatch, *, host_dirs: set[str], usage_by_path: dict, links: dict | None = None) -> list[str]:
    """isdir answers from `host_dirs`, readlink from `links`; disk_usage answers from `usage_by_path` and records the
    path asked."""
    asked: list[str] = []

    def readlink(path):
        if path not in (links or {}):
            raise OSError(22, "not a link", path)
        return links[path]

    def disk_usage(path):
        asked.append(path)
        return usage_by_path[path]

    monkeypatch.setitem(
        scrape,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(
                isdir=lambda path: path in host_dirs, join=os.path.join, dirname=os.path.dirname, normpath=os.path.normpath
            ),
            readlink=readlink,
        ),
    )
    monkeypatch.setitem(scrape, "shutil", type("_Shutil", (), {"disk_usage": staticmethod(disk_usage)}))
    return asked


_Usage = namedtuple("_Usage", "total used free")
_ROOT_PARTITION = _Usage(2_112_647_088 * 1024, 28_045_744 * 1024, 1_977_210_780 * 1024)
_DOCKER_PARTITION = _Usage(5_368_709_120 * 1024, 139_377_768 * 1024, 5_229_331_352 * 1024)


def test_disk_usage_is_measured_on_dockers_data_root(scrape: dict, monkeypatch) -> None:
    # Arrange — ticket-0286: a 2 TB root partition and /var/lib/docker on its own 5 TB partition.
    # The scrape runs in the executor container, so `/` is the container overlay; the host's
    # docker root is only reachable through PID 1's root.
    _stub_docker_api(scrape, {"/info": {"DockerRootDir": "/var/lib/docker"}})
    asked = _stub_filesystem(
        scrape,
        monkeypatch,
        host_dirs={"/proc/1/root/var/lib/docker"},
        usage_by_path={"/proc/1/root/var/lib/docker": _DOCKER_PARTITION, "/": _ROOT_PARTITION},
    )

    # Act
    usage = scrape["get_host_disk_usage"]()

    # Assert — the partition that holds the containers, not the one that holds the OS
    assert usage == _DOCKER_PARTITION
    assert asked == ["/proc/1/root/var/lib/docker"]


def test_disk_usage_follows_a_custom_docker_data_root(scrape: dict, monkeypatch) -> None:
    _stub_docker_api(scrape, {"/info": {"DockerRootDir": "/mnt/nvme/docker"}})
    asked = _stub_filesystem(
        scrape,
        monkeypatch,
        host_dirs={"/proc/1/root/mnt/nvme/docker", "/proc/1/root/var/lib/docker"},
        usage_by_path={"/proc/1/root/mnt/nvme/docker": _DOCKER_PARTITION, "/proc/1/root/var/lib/docker": _ROOT_PARTITION},
    )

    assert scrape["get_host_disk_usage"]() == _DOCKER_PARTITION
    assert asked == ["/proc/1/root/mnt/nvme/docker"]


_SNAPSHOTTER_INFO = {"DockerRootDir": "/data/docker", "DriverStatus": [["driver-type", "io.containerd.snapshotter.v1"]],
                     "Containerd": {"Address": _CTRD_ADDRESS}}
_CTRD = "/proc/1/root/var/lib/containerd"
_SNAPSHOTS = f"{_CTRD}/{_OVERLAYFS}"
_CONTENT = f"{_CTRD}/{_CONTENT_PLUGIN}"


def test_disk_usage_is_measured_where_the_containerd_image_store_keeps_containers(scrape: dict, monkeypatch) -> None:
    # webg8845: data-root on a big disk, but Docker's containerd image store keeps the containers on
    # the root disk under /var/lib/containerd; the data-root is not where rentals land
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    asked = _stub_filesystem(
        scrape,
        monkeypatch,
        host_dirs={"/proc/1/root/data/docker", _SNAPSHOTS},
        usage_by_path={"/proc/1/root/data/docker": _DOCKER_PARTITION, _SNAPSHOTS: _ROOT_PARTITION},
    )

    assert scrape["get_host_disk_usage"]() == _ROOT_PARTITION
    assert asked == [_SNAPSHOTS]


@pytest.mark.parametrize(
    "config_root, running_root, expected",
    [
        ("/var/lib/containerd", "/var/lib/containerd", _ROOT_PARTITION),  # unchanged configuration
        # root moved to the 14 TiB disk in the config, containerd not restarted: it still writes to the root disk
        ("/data/containerd", "/var/lib/containerd", _ROOT_PARTITION),
        ("/data/containerd", "/data/containerd", _DOCKER_PARTITION),  # restarted onto it
    ],
)
def test_the_running_containerds_root_is_measured_not_its_config(scrape: dict, monkeypatch, config_root, running_root, expected) -> None:
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    scrape["open"] = lambda path: io.StringIO(f'version = 2\nroot = "{config_root}"\n')
    scrape["run_cmd"] = _ctr(_running_root(running_root))
    paths = {f"/proc/1/root/var/lib/containerd/{_OVERLAYFS}": _ROOT_PARTITION, f"/proc/1/root/data/containerd/{_OVERLAYFS}": _DOCKER_PARTITION}
    _stub_filesystem(scrape, monkeypatch, host_dirs=set(paths), usage_by_path=paths)

    assert scrape["get_host_disk_usage"]() == expected


def _ctr_fails(cmd):
    raise RuntimeError("chroot: failed to run command 'ctr': No such file or directory")


@pytest.mark.parametrize(
    "address, run_cmd",
    [
        (_CTRD_ADDRESS, _ctr_fails),  # the running daemon cannot be asked: never fall back to its config files
        (None, _ctr(_running_root("/var/lib/containerd"))),  # Docker names no containerd
        ("/run/x.sock; rm -rf /", _ctr(_running_root("/var/lib/containerd"))),  # never handed to a shell
        (_CTRD_ADDRESS, _ctr(_running_root("/gone/containerd"))),  # the reported root is not on the host
        (_CTRD_ADDRESS, _ctr({_CONTENT_PLUGIN: f"/var/lib/containerd/{_CONTENT_PLUGIN}"})),  # no snapshotter root reported
    ],
)
def test_an_undeterminable_containerd_root_is_not_measured(scrape: dict, monkeypatch, address, run_cmd) -> None:
    _stub_docker_api(scrape, {"/info": {**_SNAPSHOTTER_INFO, "Containerd": {"Address": address}}})
    scrape["run_cmd"] = run_cmd
    asked = _stub_filesystem(scrape, monkeypatch, host_dirs={_SNAPSHOTS, _CONTENT}, usage_by_path={})

    with pytest.raises(RuntimeError, match=scrape["CONTAINERD_ROOT_UNKNOWN"]):
        scrape["get_host_disk_usage"]()
    assert asked == []


_GiB = 1024**3
_SMALL = _Usage(400 * _GiB, 350 * _GiB, 50 * _GiB)
_BIG = _Usage(14 * 1024 * _GiB, 1024 * _GiB, 13 * 1024 * _GiB)


@pytest.mark.parametrize(
    "store, data_root, same_device, expected_total_gib, expected_free_gib",
    [
        # one filesystem: the filler run volumes under data-root free room there
        (_SMALL, _SMALL, True, 400, 150),
        # store on 14 TiB, data-root (rental volumes, filler run volumes) with 50 GiB free: data-root binds
        (_BIG, _SMALL, False, 400, 150),
        # store with 50 GiB free, data-root on 14 TiB: the store binds, and the run volumes free nothing there
        (_SMALL, _BIG, False, 400, 50),
        # store undersized but with more free room than data-root: the store's size, data-root's free room
        (_Usage(400 * _GiB, 50 * _GiB, 350 * _GiB), _Usage(14 * 1024 * _GiB, 14 * 1024 * _GiB - 50 * _GiB, 50 * _GiB),
         False, 400, 150),
    ],
)
def test_a_renter_gets_the_room_both_the_store_and_data_root_have(
    scrape: dict, monkeypatch, store, data_root, same_device, expected_total_gib, expected_free_gib
) -> None:
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    paths = {_SNAPSHOTS: store, "/proc/1/root/data/docker": data_root}
    _stub_filesystem(scrape, monkeypatch, host_dirs=set(paths), usage_by_path=paths)
    devices = {_SNAPSHOTS: 1, "/proc/1/root/data/docker": 1 if same_device else 2}
    scrape["os"].stat = lambda path: SimpleNamespace(st_dev=devices[path])

    store_usage, data_root_usage = scrape["get_host_disk_usage"](True)
    hard_disk = {**scrape["hard_disk_reading"](data_root_usage or store_usage), "filler_released_kb": 100 * 1024**2}
    scrape["release_filler_cache"](hard_disk)
    scrape["report_tighter_store"](hard_disk, store_usage if data_root_usage else None)

    assert (data_root_usage is None) == same_device
    assert (hard_disk["hard_disk_total"], hard_disk["hard_disk_free"]) == (
        expected_total_gib * 1024**2,
        expected_free_gib * 1024**2,
    )


@pytest.mark.parametrize(
    "store, data_root",
    [
        # only the containerd store (the bigger filesystem) is over 90% used
        (_Usage(14 * 1024 * _GiB, 13 * 1024 * _GiB, 1024 * _GiB), _Usage(400 * _GiB, 40 * _GiB, 360 * _GiB)),
        # only data-root (the bigger filesystem) is over 90% used
        (_Usage(400 * _GiB, 40 * _GiB, 360 * _GiB), _Usage(14 * 1024 * _GiB, 13 * 1024 * _GiB, 1024 * _GiB)),
    ],
)
def test_the_fuller_of_the_store_and_data_root_is_the_reported_utilization(scrape: dict, store, data_root) -> None:
    # F-48: an idle node must not pass the 90% utilization limit while either filesystem a rental needs is over it
    hard_disk = scrape["hard_disk_reading"](data_root)

    scrape["report_tighter_store"](hard_disk, store)

    assert hard_disk["hard_disk_utilization"] == pytest.approx(13 / 14 * 100)
    assert hard_disk["hard_disk_total"] == 400 * 1024**2


_FULL = _Usage(400 * _GiB, 400 * _GiB, 0)


@pytest.mark.parametrize(
    "full, expected_total_gib",
    [
        # pulled layers land in the content store: a full one fails a pull, but its size is not the disk gate's
        (_CONTENT, 14 * 1024),
        (_SNAPSHOTS, 400),
        ("/proc/1/root/data/docker", 400),
    ],
)
def test_a_full_filesystem_leaves_no_free_room_whichever_it_is(scrape: dict, monkeypatch, full, expected_total_gib) -> None:
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    paths = {_SNAPSHOTS: _BIG, _CONTENT: _BIG, "/proc/1/root/data/docker": _BIG, full: _FULL}
    _stub_filesystem(scrape, monkeypatch, host_dirs={_CTRD, *paths}, usage_by_path=paths)
    scrape["os"].stat = lambda path: SimpleNamespace(st_dev=hash(path))

    store_usage, data_root_usage = scrape["get_host_disk_usage"](True)
    hard_disk = scrape["hard_disk_reading"](data_root_usage)
    scrape["report_tighter_store"](hard_disk, store_usage)
    scrape["report_tighter_store"](hard_disk, scrape["get_content_store_usage"](), False)

    assert hard_disk["hard_disk_free"] == 0
    assert hard_disk["hard_disk_utilization"] == 100
    assert hard_disk["hard_disk_total"] == expected_total_gib * 1024**2


def test_a_content_store_on_the_snapshots_filesystem_is_not_read_twice(scrape: dict, monkeypatch) -> None:
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    asked = _stub_filesystem(scrape, monkeypatch, host_dirs={_CTRD, _SNAPSHOTS, _CONTENT}, usage_by_path={})
    scrape["os"].stat = lambda path: SimpleNamespace(st_dev=1)

    assert scrape["get_content_store_usage"]() is None
    assert asked == []


@pytest.mark.parametrize(
    "snapshots_device, data_root_device, expected",
    [("nvme0n1p1", "sdb1", "hdd"), ("sdb1", "nvme0n1p1", "nvme")],
)
def test_disk_type_is_data_roots_where_rental_volumes_live(scrape: dict, monkeypatch, snapshots_device, data_root_device, expected) -> None:
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    _stub_filesystem(scrape, monkeypatch, host_dirs={_CTRD, _SNAPSHOTS, "/proc/1/root/data/docker"}, usage_by_path={})
    devices = {"/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs": snapshots_device, "/data/docker": data_root_device}
    scrape["block_device_holding"] = lambda mounts_text, path: devices[path]
    scrape["disk_type_of"] = {"nvme0n1p1": "nvme", "sdb1": "hdd"}.get

    assert scrape["get_docker_root_disk_type"]() == expected


def test_disk_usage_falls_back_to_root_without_a_reachable_docker_root(scrape: dict, monkeypatch) -> None:
    # outside the executor container (no pid: host) the host path does not exist; `/` is what there is
    _stub_docker_api(scrape, {"/info": {"DockerRootDir": "/var/lib/docker"}})
    asked = _stub_filesystem(scrape, monkeypatch, host_dirs=set(), usage_by_path={"/": _ROOT_PARTITION})

    assert scrape["get_host_disk_usage"]() == _ROOT_PARTITION
    assert asked == ["/"]


def test_disk_usage_survives_a_silent_docker_socket(scrape: dict, monkeypatch) -> None:
    # the docker socket is the fragile half (hard_disk_docker_scrape_error); total/used/free must
    # still be reported, from the default data root when the host has it
    def _no_docker(path):
        raise RuntimeError("docker api: connection refused")

    scrape["docker_api_get"] = _no_docker
    asked = _stub_filesystem(
        scrape,
        monkeypatch,
        host_dirs={"/proc/1/root/var/lib/docker"},
        usage_by_path={"/proc/1/root/var/lib/docker": _DOCKER_PARTITION},
    )

    assert scrape["get_host_disk_usage"]() == _DOCKER_PARTITION
    assert asked == ["/proc/1/root/var/lib/docker"]


def test_the_engy_cache_is_reported_as_free_disk_and_the_dolphin_cache_stays_used(scrape: dict) -> None:
    # Arrange — an engy checkpoint and a dolphin cache next to a renter's volume. DAH-5260: a customer create
    # removes the Engy cache when the renter needs its space, so an idle node lists it as free.
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 0,
                "Containers": [],
                "Volumes": [
                    {"Name": "engy_cache_hf_kimi-k3-0123abcd", "UsageData": {"Size": 1500 * 1024**3}},
                    {"Name": "dphn_cache_hf_x", "UsageData": {"Size": 40 * 1024**3}},
                    {"Name": "volume_renter", "UsageData": {"Size": 10 * 1024**3}},
                ],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )
    hard_disk = {
        "hard_disk_total": 4000 * 1024**2,
        "hard_disk_used": 1600 * 1024**2,
        "hard_disk_free": 2400 * 1024**2,
        "hard_disk_utilization": 40.0,
    }

    # Act
    hard_disk.update(scrape["get_docker_disk_usage"]())
    scrape["release_filler_cache"](hard_disk)

    # Assert — the engy cache moved from used to free and out of the volumes breakdown
    assert hard_disk == {
        "hard_disk_total": 4000 * 1024**2,
        "hard_disk_used": 100 * 1024**2,
        "hard_disk_free": 3900 * 1024**2,
        "hard_disk_utilization": 2.5,
        "hard_disk_images": 0,
        "hard_disk_containers": 0,
        "hard_disk_volumes": 50 * 1024**2,
        # DAH-5045: the engy checkpoint's real used size, without the dolphin cache or the renter's volume
        "hard_disk_engy_cache_used_gb": 1500 * 1024**3 // 10**9,
        "hard_disk_engy_cache_released_gb": 1500 * 1024**3 // 10**9,
        "hard_disk_by_container": {},
        "hard_disk_by_volume": {
            "engy_cache_hf_kimi-k3-0123abcd": 1500 * 1024**2,
            "dphn_cache_hf_x": 40 * 1024**2,
            "volume_renter": 10 * 1024**2,
        },
    }


def test_an_engy_cache_a_non_filler_container_mounts_stays_used(scrape: dict) -> None:
    # Arrange — an engy_cache_* volume mounted by a container that is not a filler: a customer create cannot remove it
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 0,
                "Containers": [{"Names": ["/provider_app"], "Mounts": [{"Type": "volume", "Name": "engy_cache_x"}]}],
                "Volumes": [{"Name": "engy_cache_x", "UsageData": {"Size": 1500 * 1024**3}}],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert — not released, still a used volume, and the checkpoint progress still reads it
    assert usage["filler_released_kb"] == 0
    assert usage["hard_disk_volumes"] == 1500 * 1024**2
    assert usage["hard_disk_engy_cache_used_gb"] == 1500 * 1024**3 // 10**9


@pytest.mark.parametrize(
    "held, store_free_gib, expected_free_gib, expected_released_gib",
    [
        # shared filesystem (no separate store): a reclaimable cache is all credited, a held one none of it
        (False, None, 2000, 1500),
        (True, None, 500, 0),
        # a separate store with less room hides part of the credit: only what shows in free was released
        (False, 1000, 1000, 500),
        (True, 1000, 500, 0),
        # a separate store with more room hides nothing
        (False, 3000, 2000, 1500),
    ],
)
def test_the_engy_cache_credited_to_free_is_reported(
    scrape: dict, held, store_free_gib, expected_free_gib, expected_released_gib
) -> None:
    # Arrange — 1500 GiB Engy cache on a 4000 GiB data-root with 500 GiB free before the release; the backend
    # subtracts hard_disk_engy_cache_released_gb (not the used size) for a job that keeps the cache
    mounts = [{"Type": "volume", "Name": "engy_cache_x"}]
    _stub_docker_api(
        scrape,
        {
            "/system/df": {
                "LayersSize": 0,
                "Containers": [{"Names": ["/provider_app" if held else f"/{scrape['FILLER_CONTAINER_PREFIX']}x"], "Mounts": mounts}],
                "Volumes": [{"Name": "engy_cache_x", "UsageData": {"Size": 1500 * _GiB}}],
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )
    hard_disk = {"hard_disk_total": 4000 * 1024**2, "hard_disk_used": 3500 * 1024**2,
                 "hard_disk_free": 500 * 1024**2, "hard_disk_utilization": 87.5}
    store = None
    if store_free_gib is not None:
        store = _Usage(8000 * _GiB, (8000 - store_free_gib) * _GiB, store_free_gib * _GiB)

    # Act
    hard_disk.update(scrape["get_docker_disk_usage"]())
    scrape["release_filler_cache"](hard_disk)
    scrape["report_tighter_store"](hard_disk, store)

    # Assert
    assert hard_disk["hard_disk_free"] == expected_free_gib * 1024**2
    assert hard_disk["hard_disk_engy_cache_released_gb"] == expected_released_gib * _GiB // 10**9
    assert hard_disk["hard_disk_engy_cache_used_gb"] == 1500 * _GiB // 10**9


def test_a_fillers_own_run_volume_is_reported_as_free_disk(scrape: dict, monkeypatch) -> None:
    # Arrange — an idle node running our filler: its preallocated 1900 GB run volume (labelled by the
    # filler create) and a renter-style volume without the label
    v2_dir = "/proc/1/root/var/lib/docker/vloopback-v2"
    blocks = {f"{v2_dir}/volume_filler": 1900 * 1024**3 // 512, f"{v2_dir}/volume_other": 10 * 1024**3 // 512}
    monkeypatch.setitem(scrape, "glob", SimpleNamespace(glob=lambda pattern: []))
    path = SimpleNamespace(join=os.path.join, isdir=lambda candidate: candidate == v2_dir)
    monkeypatch.setitem(scrape, "os", SimpleNamespace(path=path, stat=lambda p: SimpleNamespace(st_blocks=blocks[p])))
    _stub_docker_api(
        scrape,
        {
            "/system/df": {"LayersSize": 0, "Containers": [], "Volumes": []},
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {
                "Volumes": [
                    {"Name": "volume_filler", "Driver": "vloopback:v2", "Labels": {"io.lium.workload": "filler"}},
                    {"Name": "volume_other", "Driver": "vloopback:v2", "Labels": None},
                ]
            },
            "/containers/json?all=1": [{"Names": ["/filler_pod1"], "Mounts": [{"Name": "volume_filler"}]}],
        },
    )
    hard_disk = {
        "hard_disk_total": 4000 * 1024**2,
        "hard_disk_used": 1950 * 1024**2,
        "hard_disk_free": 2050 * 1024**2,
        "hard_disk_utilization": 48.75,
    }

    # Act
    hard_disk.update(scrape["get_docker_disk_usage"]())
    scrape["release_filler_cache"](hard_disk)

    # Assert — only the filler's volume moved from used to free
    assert hard_disk["hard_disk_free"] == 3950 * 1024**2
    assert hard_disk["hard_disk_used"] == 50 * 1024**2
    assert hard_disk["hard_disk_volumes"] == 10 * 1024**2


def test_a_labelled_volume_a_non_filler_container_mounts_stays_used(scrape: dict, monkeypatch) -> None:
    # Arrange — a provider labels a volume as filler and mounts it into its own container
    v2_dir = "/proc/1/root/var/lib/docker/vloopback-v2"
    monkeypatch.setitem(scrape, "glob", SimpleNamespace(glob=lambda pattern: []))
    path = SimpleNamespace(join=os.path.join, isdir=lambda candidate: candidate == v2_dir)
    monkeypatch.setitem(scrape, "os", SimpleNamespace(path=path, stat=lambda p: SimpleNamespace(st_blocks=8)))
    _stub_docker_api(
        scrape,
        {
            "/volumes": {
                "Volumes": [{"Name": "volume_x", "Driver": "vloopback:v2", "Labels": {"io.lium.workload": "filler"}}]
            },
            "/containers/json?all=1": [
                {"Names": ["/filler_pod1"], "Mounts": [{"Name": "volume_x"}]},
                {"Names": ["/provider_app"], "Mounts": [{"Name": "volume_x"}]},
            ],
        },
    )

    # Act / Assert — not reclaimable, still counted among all volumes
    assert scrape["get_vloopback_volume_bytes"]("/var/lib/docker", True) == 0
    assert scrape["get_vloopback_volume_bytes"]("/var/lib/docker") == 8 * 512


def test_a_labelled_volume_the_rental_cleanup_does_not_sweep_stays_used(scrape: dict, monkeypatch) -> None:
    # Arrange — a labelled, unmounted vloopback volume outside the `volume_*` names the rental cleanup removes
    v2_dir = "/proc/1/root/var/lib/docker/vloopback-v2"
    monkeypatch.setitem(scrape, "glob", SimpleNamespace(glob=lambda pattern: []))
    path = SimpleNamespace(join=os.path.join, isdir=lambda candidate: candidate == v2_dir)
    monkeypatch.setitem(scrape, "os", SimpleNamespace(path=path, stat=lambda p: SimpleNamespace(st_blocks=8)))
    _stub_docker_api(
        scrape,
        {
            "/volumes": {
                "Volumes": [{"Name": "provider_data", "Driver": "vloopback:v2", "Labels": {"io.lium.workload": "filler"}}]
            },
            "/containers/json?all=1": [],
        },
    )

    # Act / Assert — not reclaimable, still counted among all volumes
    assert scrape["get_vloopback_volume_bytes"]("/var/lib/docker", True) == 0
    assert scrape["get_vloopback_volume_bytes"]("/var/lib/docker") == 8 * 512


def test_release_filler_cache_without_a_disk_reading_only_drops_the_key(scrape: dict) -> None:
    # Arrange — the host half of the scrape failed, the docker half did not
    hard_disk = {"hard_disk_volumes": 0, "filler_released_kb": 5, "engy_released_kb": 5}

    # Act
    scrape["release_filler_cache"](hard_disk)

    # Assert — nothing was credited to free
    assert hard_disk == {"hard_disk_volumes": 0, "hard_disk_engy_cache_released_gb": 0}


def test_a_capability_check_is_named_and_removed_even_when_it_times_out() -> None:
    # DAH-5192: `--rm` never fires for a container a killed client created but never started
    removed: list[list[str]] = []

    def timing_out_run(command, **kwargs):
        raise TimeoutError(command)

    scrape = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"CAPABILITY_CHECK_NAME_PREFIX", "run_capability_check"},
        {
            "os": os,
            "subprocess": SimpleNamespace(run=timing_out_run, PIPE=-1),
            "run_storage_probe": lambda command, timeout: removed.append(command),
            "VLOOPBACK_CLEANUP_TIMEOUT_SECONDS": 10,
        },
    )

    with pytest.raises(TimeoutError) as raised:
        scrape["run_capability_check"](["docker", "run", "--rm", "--gpus", "all", "img", "nvidia-smi"])

    command = raised.value.args[0]
    name = command[command.index("--name") + 1]
    assert command[:3] == ["docker", "run", "--rm"] and name.startswith("lium_capability_check_")
    # the exact name the stale cleanup removes by, and nothing looser
    from services.container_cleanup import never_started_check_containers

    assert never_started_check_containers(f"a1 {name}") == ["a1"]
    assert removed == [["docker", "rm", "-f", name]]


@pytest.mark.parametrize(
    "links",
    [
        {"/proc/1/root/var/lib/containerd": "/data/containerd"},  # absolute: through /proc/1/root it would miss
        {"/proc/1/root/var/lib/containerd": "../../data/containerd"},  # relative
        # a link in a parent directory, then an absolute one
        {"/proc/1/root/var": "/srv/var", "/proc/1/root/srv/var/lib/containerd": "/data/containerd"},
    ],
)
def test_a_symlinked_containerd_root_is_followed_on_the_host(scrape: dict, monkeypatch, links) -> None:
    # /var/lib/containerd -> /data/containerd, as the installer on the host sees it; /data/containerd is not in
    # the executor container
    _stub_docker_api(scrape, {"/info": _SNAPSHOTTER_INFO})
    snapshots = f"/proc/1/root/data/containerd/{_OVERLAYFS}"
    asked = _stub_filesystem(
        scrape, monkeypatch, host_dirs={snapshots}, usage_by_path={snapshots: _DOCKER_PARTITION}, links=links,
    )

    assert scrape["get_host_disk_usage"]() == _DOCKER_PARTITION
    assert asked == [snapshots]


def test_host_mount_points_decodes_mountinfo_escapes(scrape: dict, monkeypatch) -> None:
    mountinfo = "22 1 259:2 / / rw - ext4 /dev/root rw\n30 22 7:3 / /mnt/my\\040disk rw - ext4 /dev/loop3 rw\n"
    monkeypatch.setitem(scrape, "open", lambda path: io.StringIO(mountinfo))

    assert scrape["host_mount_points"]() == {"/", "/mnt/my disk"}


@pytest.mark.parametrize(
    "line, mount_point",
    [
        ("31 22 7:4 / /var/lib/docker/volumes/v/_data rw - ext4 /dev/loop4 rw", "/var/lib/docker/volumes/v/_data"),
        ("30 22 7:3 / /mnt/my\\040disk rw - ext4 /dev/loop3 rw", "/mnt/my disk"),
        ("32 22 7:5 / /mnt/tab\\011and\\134slash rw - ext4 /dev/loop5 rw", "/mnt/tab\tand\\slash"),
        ("garbage", None),
    ],
    ids=["plain", "escaped-space", "escaped-tab-backslash", "short-line"],
)
def test_packaged_scrape_reads_escaped_mount_points(monkeypatch, line, mount_point) -> None:
    # F-5: run host_mount_points as it ships, out of obfuscator.py's output. A lambda there kept its
    # argument name while its body was renamed, so an escaped path raised NameError and the whole
    # docker disk breakdown fell into the error key.
    from miner_jobs.obfuscator import obfuscate_code

    tree = ast.parse(obfuscate_code((SRC / "miner_jobs" / "machine_scrape.py").read_text()))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(child, ast.Constant) and child.value == "/proc/1/mountinfo" for child in ast.walk(node))
    )
    mountinfo = f"22 1 259:2 / / rw - ext4 /dev/root rw\n{line}\n"
    namespace = {"re": re, "open": lambda path: io.StringIO(mountinfo)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "obfuscated", "exec"), namespace)

    mount_points = namespace[function.name]()

    assert mount_points == {"/"} | ({mount_point} if mount_point else set())
