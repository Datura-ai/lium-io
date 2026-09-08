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
import json
import os
import shutil
import socket
from collections import namedtuple
from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

DISK_HELPERS = {
    "DOCKER_SOCKET_PATH",
    "DOCKER_API_TIMEOUT_SECONDS",
    "VLOOPBACK_DRIVER_PREFIX",
    "UnixSocketHTTPConnection",
    "docker_api_get",
    "get_vloopback_volume_bytes",
    "get_container_log_bytes",
    "get_host_disk_usage",
    "get_docker_disk_usage",
}


@pytest.fixture
def scrape() -> dict:
    """The disk helpers, executed in a namespace of their own."""
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        DISK_HELPERS,
        {"glob": glob, "http": http, "json": json, "os": os, "shutil": shutil, "socket": socket},
    )


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
    }


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
    assert usage == {"hard_disk_images": 0, "hard_disk_containers": 0, "hard_disk_volumes": 0}


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


def _stub_filesystem(scrape: dict, monkeypatch, *, host_dirs: set[str], usage_by_path: dict) -> list[str]:
    """isdir answers from `host_dirs`; disk_usage answers from `usage_by_path` and records the path asked."""
    asked: list[str] = []

    def disk_usage(path):
        asked.append(path)
        return usage_by_path[path]

    monkeypatch.setitem(
        scrape,
        "os",
        type("_Os", (), {"path": type("_Path", (), {"isdir": staticmethod(lambda path: path in host_dirs)})}),
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
