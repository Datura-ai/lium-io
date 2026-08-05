"""DAH-2571 — get_infiniband_ports() in machine_scrape.py reads the kernel's RDMA view from sysfs.

Fixtures are values measured on production executors on 2026-08-04/05, not invented shapes.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are extracted by ast and executed in their own namespace (same pattern as
test_scrape_ncu_profiling.py).
"""

import ast
import glob
import os
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

INFINIBAND_HELPERS = {
    "INFINIBAND_SYSFS_PATH",
    "GID_TABLE_ENTRIES_READ",
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
}

# 38.255.28.18 (8xH200), mlx5_0 in Ethernet mode. GIDs 2-3 are the IPv4-mapped ones — the only
# entries that say which segment the port is reachable on; 0-1 carry the default fe80:: prefix
# every prod port shares.
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
    ports = scrape["get_infiniband_ports"]()

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
            "ib_gids": ["fe80:0000:0000:0000:9a03:9bff:fe1d:8a42"],
        }
    ]


def test_roce_port_keeps_the_ipv4_mapped_gids(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange
    write_port(tmp_path, "mlx5_0", "1", ROCE_PORT_FILES)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]()

    # Assert — GIDs 2-3 encode the port's IPv4 address, which is how two RoCE ports are told they
    # share a segment. Losing them would leave Ethernet ports unpairable.
    assert ports[0].link_layer == "Ethernet"
    assert ports[0].gids[2:] == [
        "0000:0000:0000:0000:0000:ffff:26ff:1c12",
        "0000:0000:0000:0000:0000:ffff:26ff:1c12",
    ]


def test_link_layer_not_lid_separates_the_two_fabrics(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — Intel irdma (E810) reports lid 0x1 on an Ethernet port, observed on 23.153.44.20
    # and 69.63.236.160. mlx5 reports 0x0 there. Only link_layer is safe to branch on.
    write_port(tmp_path, "irdma0", "1", {**ROCE_PORT_FILES, "lid": "0x1", "sm_lid": "0x0"})
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]()

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
    ports = scrape["get_infiniband_ports"]()

    # Assert
    assert [(port.device, port.port) for port in ports] == [
        ("mlx5_2", "1"),
        ("mlx5_9", "1"),
        ("mlx5_9", "2"),
    ]


def test_host_without_rdma_hardware_reports_no_ports(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — the usual case: no /sys/class/infiniband at all
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path / "does-not-exist")

    # Act / Assert — an empty list, not a scrape failure
    assert scrape["get_infiniband_ports"]() == []


def test_unreadable_attribute_leaves_an_empty_field(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — older drivers do not expose every attribute
    write_port(tmp_path, "mlx5_2", "1", {key: value for key, value in IB_PORT_FILES.items() if key != "sm_lid"})
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]()

    # Assert — the port still lands, only the missing attribute is empty
    assert ports[0].sm_lid == ""
    assert ports[0].state == "4: ACTIVE"


def _dict_literal_keys(module: ast.Module, dict_name: str) -> list[str]:
    """Keys of the single dict literal assigned to `dict_name`, in source order."""
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == dict_name
            and isinstance(node.value, ast.Dict)
        ):
            return [key.value for key in node.value.keys]
    raise AssertionError(f"{dict_name} dict literal not found")


def test_infiniband_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = [
        "data_infiniband_ports",
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
    original_keys = _dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = _dict_literal_keys(service_module, "all_keys")

    # Assert
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys
