"""DAH-2571 — get_infiniband_ports() in machine_scrape.py reads the kernel's RDMA view from sysfs.

Fixtures are values measured on production executors on 2026-08-04/05, not invented shapes.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are extracted by ast and executed in their own namespace (same pattern as
test_scrape_ncu_profiling.py).
"""

import ast
import glob
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"

INFINIBAND_HELPERS = {
    "INFINIBAND_SYSFS_PATH",
    "InfinibandObservation",
    "GID_TABLE_ENTRIES_READ",
    "IPV4_MAPPED_GID_PREFIX",
    "read_sysfs_value",
    "InfinibandPort",
    "read_infiniband_port",
    "get_infiniband_ports",
}

# One ConnectX-7 port as 204.9.206.243 (8xB300) reported it: its own subnet manager, sm_lid
# resolving to its own port.
IB_PORT_FILES = {
    "link_layer": "InfiniBand",
    "state": "4: ACTIVE",
    "phys_state": "5: LinkUp",
    "rate": "100 Gb/sec (2X HDR)",
    "lid": "0x4",
    "sm_lid": "0x1",
    "pkeys/0": "0x7fff",
    "gids/0": "fe80:0000:0000:0000:9a03:9bff:fe1d:8a42",
    "gids/1": "fe80:0000:0000:0000:9a03:9bff:fe1d:8a42",
    "gids/2": "0000:0000:0000:0000:0000:ffff:0a7d:016b",
    "gids/3": "0000:0000:0000:0000:0000:ffff:0a7d:016b",
}


# 38.255.28.18 (8xH200), mlx5_0 in Ethernet mode. The IPv4-mapped GID says which segment the port
# is reachable on; mlx5 puts it at indices 2-3, other drivers elsewhere, so it is found by prefix.
ROCE_PORT_FILES = {
    "link_layer": "Ethernet",
    "state": "4: ACTIVE",
    "phys_state": "5: LinkUp",
    "rate": "100 Gb/sec (4X EDR)",
    "lid": "0x0",
    "sm_lid": "0x0",
    "pkeys/0": "0xffff",
    "gids/0": "fe80:0000:0000:0000:c670:bdff:fe7e:66c9",
    "gids/1": "fe80:0000:0000:0000:c670:bdff:fe7e:66c9",
    "gids/2": "0000:0000:0000:0000:0000:ffff:26ff:1c12",
    "gids/3": "0000:0000:0000:0000:0000:ffff:26ff:1c12",
}


@pytest.fixture
def scrape() -> dict[str, Any]:
    """The InfiniBand helpers, executed in a namespace of their own."""
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", INFINIBAND_HELPERS, {"os": os, "glob": glob}
    )


def write_port(sysfs_root: Path, device: str, port: str, files: dict[str, str]) -> None:
    device_path = sysfs_root / device
    device_path.mkdir(parents=True, exist_ok=True)
    (device_path / "node_guid").write_text("9a03:9bff:fe1d:8a42\n")
    (device_path / "sys_image_guid").write_text("ac3a:e203:000f:62f7\n")
    for name, content in files.items():
        file_path = device_path / "ports" / port / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"{content}\n")


def test_infiniband_port_is_reported_with_its_fabric_identity(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange
    write_port(tmp_path, "mlx5_2", "1", IB_PORT_FILES)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]().ports

    # Assert — sm_lid plus the LID is what tells two hosts apart; every prod GID starts fe80::, so
    # the prefix on its own identifies nothing.
    assert [port.as_payload() for port in ports] == [
        {
            "ib_device": "mlx5_2",
            "ib_port": "1",
            "ib_node_guid": "9a03:9bff:fe1d:8a42",
            "ib_sys_image_guid": "ac3a:e203:000f:62f7",
            "ib_link_layer": "InfiniBand",
            "ib_state": "4: ACTIVE",
            "ib_phys_state": "5: LinkUp",
            "ib_rate": "100 Gb/sec (2X HDR)",
            "ib_lid": "0x4",
            "ib_sm_lid": "0x1",
            "ib_pkey": "0x7fff",
            "ib_gids": [
                "fe80:0000:0000:0000:9a03:9bff:fe1d:8a42",
                "fe80:0000:0000:0000:9a03:9bff:fe1d:8a42",
                "0000:0000:0000:0000:0000:ffff:0a7d:016b",
                "0000:0000:0000:0000:0000:ffff:0a7d:016b",
            ],
        }
    ]


@pytest.mark.parametrize(
    ("device", "files", "expected_ipv4_gid"),
    [
        # mlx5 puts the IPv4-mapped GID at 2-3 (38.255.28.18); Intel irdma puts it at 1 and leaves
        # 2-3 zeroed (23.153.44.20). Matching on the index would find nothing on irdma.
        ("mlx5_0", ROCE_PORT_FILES, "0000:0000:0000:0000:0000:ffff:26ff:1c12"),
        (
            "irdma0",
            {
                **ROCE_PORT_FILES,
                "lid": "0x1",
                "gids/1": "0000:0000:0000:0000:0000:ffff:ac10:6621",
                "gids/2": "0000:0000:0000:0000:0000:0000:0000:0000",
                "gids/3": "0000:0000:0000:0000:0000:0000:0000:0000",
            },
            "0000:0000:0000:0000:0000:ffff:ac10:6621",
        ),
    ],
)
def test_ipv4_mapped_gid_is_found_by_prefix_on_every_driver(
    scrape: dict[str, Any], tmp_path: Path, device: str, files: dict[str, str], expected_ipv4_gid: str
) -> None:
    # Arrange — the IPv4-mapped GID is what tells two Ethernet ports they share a segment
    write_port(tmp_path, device, "1", files)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]().ports

    # Assert
    prefix = scrape["IPV4_MAPPED_GID_PREFIX"]
    ipv4_gids = [gid for gid in ports[0].gids if gid.startswith(prefix)]
    assert ports[0].link_layer == "Ethernet"
    # mlx5 lists it twice (RoCE v1 and v2), irdma once — the value is what matters, not the count
    assert set(ipv4_gids) == {expected_ipv4_gid}


def test_link_layer_not_lid_separates_the_two_fabrics(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — Intel irdma (E810) reports lid 0x1 on an Ethernet port, observed on 23.153.44.20
    # and 69.63.236.160. mlx5 reports 0x0 there. Only link_layer is safe to branch on.
    write_port(tmp_path, "irdma0", "1", {**ROCE_PORT_FILES, "lid": "0x1", "sm_lid": "0x0"})
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]().ports

    # Assert
    assert ports[0].link_layer == "Ethernet"
    assert ports[0].lid == "0x1"


def test_every_device_and_port_is_listed(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — a B300 host carries 24 mlx5 devices, ~16 of them disabled. Disabled ports are what
    # a provider would cable, so they must be reported; their `rate` is a driver placeholder.
    write_port(tmp_path, "mlx5_2", "1", IB_PORT_FILES)
    write_port(tmp_path, "mlx5_9", "1", {**IB_PORT_FILES, "state": "1: DOWN", "phys_state": "3: Disabled", "rate": "40 Gb/sec (4X QDR)"})
    write_port(tmp_path, "mlx5_9", "2", {**IB_PORT_FILES, "state": "1: DOWN", "phys_state": "3: Disabled"})
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]().ports

    # Assert
    assert [(port.device, port.port) for port in ports] == [
        ("mlx5_2", "1"),
        ("mlx5_9", "1"),
        ("mlx5_9", "2"),
    ]


def test_missing_sysfs_tree_says_so_instead_of_looking_like_no_hardware(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — either the host has no RDMA at all, or the scrape cannot see the tree from where it
    # runs. Both give an empty list, so the reason has to be reported.
    missing = str(tmp_path / "does-not-exist")
    scrape["INFINIBAND_SYSFS_PATH"] = missing

    # Act
    observation = scrape["get_infiniband_ports"]()

    # Assert
    assert observation.ports == []
    assert observation.scrape_error == f"{missing} does not exist"


def test_empty_sysfs_tree_is_distinguished_from_a_missing_one(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange — the tree exists but the kernel registered no RDMA device
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    observation = scrape["get_infiniband_ports"]()

    # Assert
    assert observation.ports == []
    assert observation.scrape_error == f"{tmp_path} lists no devices"


def test_device_without_ports_is_reported_as_such(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — a device directory with no ports/ subtree
    (tmp_path / "mlx5_0").mkdir(parents=True)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    observation = scrape["get_infiniband_ports"]()

    # Assert
    assert observation.ports == []
    assert observation.scrape_error == "1 device(s) present, none exposing a port"


def test_a_successful_walk_reports_no_error(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange
    write_port(tmp_path, "mlx5_2", "1", IB_PORT_FILES)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act / Assert
    assert scrape["get_infiniband_ports"]().scrape_error == ""


def test_unreadable_attribute_leaves_an_empty_field(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — older drivers do not expose every attribute
    write_port(tmp_path, "mlx5_2", "1", {key: value for key, value in IB_PORT_FILES.items() if key != "sm_lid"})
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]().ports

    # Assert — the port still lands, only the missing attribute is empty
    assert ports[0].sm_lid == ""
    assert ports[0].state == "4: ACTIVE"


def test_infiniband_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = [
        "data_infiniband_ports",
        "data_infiniband_scrape_error",
        "ib_device",
        "ib_port",
        "ib_node_guid",
        "ib_sys_image_guid",
        "ib_link_layer",
        "ib_state",
        "ib_phys_state",
        "ib_rate",
        "ib_lid",
        "ib_sm_lid",
        "ib_pkey",
        "ib_gids",
    ]

    # Act
    original_keys = dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Assert
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys


@lru_cache(maxsize=1)
def _source_text() -> str:
    return (SRC / "miner_jobs" / "machine_scrape.py").read_text()


@lru_cache(maxsize=1)
def _obfuscated_text() -> str:
    """The scrape as it ships: obfuscated, then payload keys swapped for random names.

    Cached because every call re-rolls the random names — two calls disagree on what anything is
    called, which silently breaks any test that pairs a name with a body.
    """
    import contextlib
    import io

    from miner_jobs.obfuscator import obfuscate_code
    from services.file_encrypt_service import FileEncryptService

    with contextlib.redirect_stdout(io.StringIO()):  # the obfuscator narrates to stdout
        obfuscated = obfuscate_code(_source_text())
        key_mapping, _ = FileEncryptService.generate_key_mappings(FileEncryptService.__new__(FileEncryptService))
    for original_key, random_name in key_mapping.items():
        obfuscated = obfuscated.replace(original_key, random_name)
    return obfuscated


def _obfuscated_infiniband_source() -> str:
    """Just the InfiniBand helpers of the obfuscated module, as runnable source."""

    def top_level_name(node: ast.AST) -> str:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            return node.name
        if isinstance(node, ast.Assign):
            return getattr(node.targets[0], "id", "")
        return ""

    source_tree, obfuscated_tree = ast.parse(_source_text()), ast.parse(_obfuscated_text())
    kept = [
        obfuscated_tree.body[index]
        for index, node in enumerate(source_tree.body)
        if top_level_name(node) in INFINIBAND_HELPERS
    ]
    assert len(kept) == len(INFINIBAND_HELPERS), "machine_scrape.py no longer defines all of them at module level"
    return ast.unparse(ast.Module(body=kept, type_ignores=[]))



def _obfuscated_name_of(source_name: str) -> str:
    """What `source_name` is called after obfuscation — node order is preserved, names are not."""
    source_tree, obfuscated_tree = ast.parse(_source_text()), ast.parse(_obfuscated_text())
    for index, node in enumerate(source_tree.body):
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == source_name:
            return obfuscated_tree.body[index].name
    raise AssertionError(f"{source_name} is no longer defined at module level")


def _obfuscated_infiniband_namespace() -> dict[str, Any]:
    """The InfiniBand helpers as they ship, executed.

    The unobfuscated source is not what runs on an executor, and the two disagree: obfuscator.py
    renames `__init__` parameters while leaving keyword names at the call site, so a keyword
    construction raises TypeError only in the packaged scrape. That is what shipped in #1192 and
    returned an empty port list on all 373 prod executors, hosts with 24 mlx5 devices included.
    """
    namespace: dict[str, Any] = {"os": os, "glob": glob}
    exec(_obfuscated_infiniband_source(), namespace)  # noqa: S102
    return namespace


def test_the_obfuscated_scrape_parses_a_port_and_does_not_just_return_empty(tmp_path: Path) -> None:
    # Arrange
    namespace = _obfuscated_infiniband_namespace()
    sysfs_path_name = next(name for name, value in namespace.items() if value == "/sys/class/infiniband")
    read_ports = next(
        value for value in namespace.values() if callable(value) and "RDMA port" in (getattr(value, "__doc__", "") or "")
    )
    write_port(tmp_path, "mlx5_2", "1", IB_PORT_FILES)
    namespace[sysfs_path_name] = str(tmp_path)

    # Act
    observation = read_ports()

    # Assert — an empty list here means the packaged scrape is broken, whatever the source does
    assert observation.scrape_error == ""
    assert len(observation.ports) == 1
    assert observation.ports[0].link_layer == "InfiniBand"
    # payload keys are substituted with random names by then, so only their count is checkable
    assert len(observation.ports[0].as_payload()) == 12


def test_no_locally_defined_name_in_the_scrape_is_called_with_keyword_arguments() -> None:
    """obfuscator.py renames parameters but not the keyword names at call sites, so any keyword
    call to something this file defines raises TypeError once packaged — silently, because every
    probe here swallows its own exception. `InfinibandPort(device=...)` did exactly that on all 373
    prod executors. Builtins and imported callables are fine; only our own names break.
    """
    # Arrange
    tree = ast.parse((SRC / "miner_jobs" / "machine_scrape.py").read_text())
    locally_defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.ClassDef)}

    # Act
    keyword_calls = [
        f"{node.func.id}(...) at line {node.lineno} passes {[keyword.arg for keyword in node.keywords]}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in locally_defined
        and node.keywords
    ]

    # Assert
    assert keyword_calls == [], "call it positionally — the packaged scrape cannot take these keywords"


def test_the_pyinstaller_binary_parses_a_port(tmp_path: Path) -> None:
    """The last transformation before an executor runs it: obfuscate, substitute keys, freeze.

    The AST-level test above catches renaming bugs, but the artifact that ships is a PyInstaller
    one-file build, so this drives that. With the keyword construction that shipped in #1192 the
    binary reproduces the production error verbatim — `__init__() got an unexpected keyword
    argument 'device'` — and returns no ports.
    """
    # Arrange — build the same helpers the AST test executes, pointed at a fake sysfs tree
    namespace_source = _obfuscated_infiniband_source()
    sysfs_root = tmp_path / "sys"
    write_port(sysfs_root, "mlx5_2", "1", IB_PORT_FILES)
    probe_source = namespace_source.replace("'/sys/class/infiniband'", repr(str(sysfs_root)))
    entry_point = _obfuscated_name_of("get_infiniband_ports")
    probe_source += (
        "\nimport json"
        f"\n_observation = {entry_point}()"
        "\nprint(json.dumps({'ports': len(_observation.ports), 'error': _observation.scrape_error}))\n"
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    probe_path = build_dir / "scrape_probe.py"
    probe_path.write_text("import os, glob\n" + probe_source)

    # Act — freeze it exactly like FileEncryptService.make_binary_file does, then run it
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(probe_path), "--onefile", "--noconsole",
         "--log-level=ERROR", "--distpath", str(build_dir), "--workpath", str(build_dir / "w"),
         "--specpath", str(build_dir), "--name", "scrape_probe"],
        check=True, capture_output=True,
    )
    result = subprocess.run([str(build_dir / "scrape_probe")], capture_output=True, text=True, timeout=180)

    # Assert
    assert result.stdout.strip(), f"the frozen binary printed nothing; rc={result.returncode} stderr={result.stderr[:300]}"
    reported = json.loads(result.stdout.strip().splitlines()[-1])
    assert reported["error"] == ""
    assert reported["ports"] == 1


def test_an_unreadable_sysfs_tree_is_not_reported_as_no_devices(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    """glob would swallow the EACCES and answer [], which reads as "this host has no RDMA" — the
    exact silent-nothing this function exists to stop. os.listdir raises and the reason survives.
    """
    # Arrange
    write_port(tmp_path, "mlx5_2", "1", IB_PORT_FILES)
    tmp_path.chmod(0o000)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    try:
        # Act
        observation = scrape["get_infiniband_ports"]()
    finally:
        tmp_path.chmod(0o755)

    # Assert
    assert observation.ports == []
    assert "Permission denied" in observation.scrape_error
