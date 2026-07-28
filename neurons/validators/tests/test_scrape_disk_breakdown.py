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
import socket
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

DISK_HELPERS = {
    "DOCKER_SOCKET_PATH",
    "DOCKER_API_TIMEOUT_SECONDS",
    "VLOOPBACK_DRIVER_PREFIX",
    "UnixSocketHTTPConnection",
    "docker_api_get",
    "get_vloopback_volume_bytes",
    "get_docker_disk_usage",
}


def _definition_name(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return ""


@pytest.fixture
def scrape() -> dict:
    """The disk helpers, executed in a namespace of their own."""
    tree = ast.parse((SRC / "miner_jobs" / "machine_scrape.py").read_text())
    kept = [node for node in tree.body if _definition_name(node) in DISK_HELPERS]
    assert {_definition_name(node) for node in kept} == DISK_HELPERS, (
        f"machine_scrape.py no longer defines all of {sorted(DISK_HELPERS)} at module level"
    )

    namespace = {"glob": glob, "http": http, "json": json, "os": os, "socket": socket}
    exec(compile(ast.Module(body=kept, type_ignores=[]), "machine_scrape_disk", "exec"), namespace)
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
            },
            "/info": {"DockerRootDir": "/var/lib/docker"},
            "/volumes": {"Volumes": []},
        },
    )

    # Act
    usage = scrape["get_docker_disk_usage"]()

    # Assert
    assert usage == {
        "hard_disk_images": 45079943748 // 1024,
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
        type("_Glob", (), {"glob": staticmethod(lambda pattern: sorted({path.rsplit("/", 1)[0] for path in blocks_by_path}))}),
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
