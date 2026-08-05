"""DAH-2571 — get_infiniband_ports() in machine_scrape.py reads the kernel's RDMA view from sysfs.

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
    "read_sysfs_value",
    "InfinibandPort",
    "read_infiniband_port",
    "get_infiniband_ports",
}

# One ConnectX-7 port as the two probed B300 hosts reported it on 2026-08-04.
ACTIVE_PORT_FILES = {
    "link_layer": "InfiniBand",
    "state": "4: ACTIVE",
    "phys_state": "5: LinkUp",
    "rate": "100 Gb/sec (2X HDR)",
    "lid": "0x4",
    "sm_lid": "0x1",
    "gids/0": "fe80:0000:0000:0000:9a03:9bff:fe1d:8a42",
}


@pytest.fixture
def scrape() -> dict[str, Any]:
    """The InfiniBand helpers, executed in a namespace of their own."""
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", INFINIBAND_HELPERS, {"os": os, "glob": glob}
    )


def write_port(sysfs_root: Path, device: str, port: str, files: dict[str, str]) -> None:
    device_path = sysfs_root / device
    (device_path).mkdir(parents=True, exist_ok=True)
    (device_path / "node_guid").write_text("9a03:9bff:fe1d:8a42\n")
    for name, content in files.items():
        file_path = device_path / "ports" / port / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"{content}\n")


def test_active_port_is_reported_with_its_fabric_identity(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange
    write_port(tmp_path, "mlx5_2", "1", ACTIVE_PORT_FILES)
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]()

    # Assert — the subnet prefix is the first four GID groups, the join key for one fabric
    assert [port.as_payload() for port in ports] == [
        {
            "ib_device": "mlx5_2",
            "ib_port": "1",
            "ib_node_guid": "9a03:9bff:fe1d:8a42",
            "ib_link_layer": "InfiniBand",
            "ib_state": "4: ACTIVE",
            "ib_phys_state": "5: LinkUp",
            "ib_rate": "100 Gb/sec (2X HDR)",
            "ib_lid": "0x4",
            "ib_sm_lid": "0x1",
            "ib_subnet_prefix": "fe80:0000:0000:0000",
        }
    ]


def test_every_device_and_port_is_listed(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — a host carries 24 mlx5 devices; disabled ones must be reported, not dropped
    write_port(tmp_path, "mlx5_2", "1", ACTIVE_PORT_FILES)
    write_port(tmp_path, "mlx5_9", "1", {**ACTIVE_PORT_FILES, "state": "1: DOWN", "phys_state": "3: Disabled"})
    write_port(tmp_path, "mlx5_9", "2", {**ACTIVE_PORT_FILES, "state": "1: DOWN", "phys_state": "3: Disabled"})
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]()

    # Assert
    assert [(port.device, port.port) for port in ports] == [
        ("mlx5_2", "1"),
        ("mlx5_9", "1"),
        ("mlx5_9", "2"),
    ]


def test_roce_port_is_reported_as_ethernet_without_lids(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange - a ConnectX-6 Dx in Ethernet mode, as measured on a rented box on 2026-08-04.
    # LIDs are an InfiniBand concept, so the driver reports 0x0 on an Ethernet port.
    write_port(
        tmp_path,
        "mlx5_0",
        "1",
        {
            "link_layer": "Ethernet",
            "state": "4: ACTIVE",
            "phys_state": "5: LinkUp",
            "rate": "100 Gb/sec (4X EDR)",
            "lid": "0x0",
            "sm_lid": "0x0",
            "gids/0": "fe80:0000:0000:0000:0ac0:ebff:fed4:1a4e",
        },
    )
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path)

    # Act
    ports = scrape["get_infiniband_ports"]()

    # Assert - RoCE hardware still lands; pairing such ports cannot lean on LIDs
    assert ports[0].link_layer == "Ethernet"
    assert ports[0].state == "4: ACTIVE"
    assert ports[0].rate == "100 Gb/sec (4X EDR)"
    assert ports[0].lid == "0x0"


def test_host_without_rdma_hardware_reports_no_ports(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — the usual case: no /sys/class/infiniband at all
    scrape["INFINIBAND_SYSFS_PATH"] = str(tmp_path / "does-not-exist")

    # Act / Assert — an empty list, not a scrape failure
    assert scrape["get_infiniband_ports"]() == []


def test_unreadable_attribute_leaves_an_empty_field(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange — older drivers do not expose every attribute
    write_port(tmp_path, "mlx5_2", "1", {key: value for key, value in ACTIVE_PORT_FILES.items() if key != "sm_lid"})
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
        "ib_link_layer",
        "ib_state",
        "ib_phys_state",
        "ib_rate",
        "ib_lid",
        "ib_sm_lid",
        "ib_subnet_prefix",
    ]

    # Act
    original_keys = _dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = _dict_literal_keys(service_module, "all_keys")

    # Assert
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys
