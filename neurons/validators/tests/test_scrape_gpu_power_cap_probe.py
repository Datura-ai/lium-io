"""DAH-2704 - see probe_gpu_power_cap_ability() in machine_scrape.py for what makes a host able to
apply `nvidia-smi -pl`: CAP_SYS_ADMIN in the executor container AND a /dev/nvidiactl its root owns.

machine_scrape.py is a script, not a module - importing it runs the whole scrape - so the helpers
are extracted by ast and executed in their own namespace (same pattern as test_scrape_ncu_profiling.py).
"""

import ast
import os
import re
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

PROBE_HELPERS = {
    "PROC_SELF_STATUS_PATH",
    "NVIDIACTL_PATH",
    "GpuPowerCapProbe",
    "probe_gpu_power_cap_ability",
}

FULL_CAPS = "000001ffffffffff"  # privileged / sysbox container
DEFAULT_DOCKER_CAPS = "00000000a80425fb"  # no CAP_SYS_ADMIN


@pytest.fixture
def scrape() -> dict[str, Any]:
    """The power-cap probe helpers, executed in a namespace of their own."""
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", PROBE_HELPERS, {"re": re, "os": os}
    )


def _status_file(tmp_path: Path, cap_eff_line: str) -> str:
    status_file = tmp_path / "status"
    status_file.write_text(f"Name:\tpython3\nUid:\t0\t0\t0\t0\n{cap_eff_line}\nSeccomp:\t0\n")
    return str(status_file)


@pytest.mark.parametrize(
    ("cap_eff_line", "expected_cap_eff"),
    [
        (f"CapEff:\t{FULL_CAPS}", FULL_CAPS),
        (f"CapEff:\t{DEFAULT_DOCKER_CAPS}", DEFAULT_DOCKER_CAPS),
        ("CapPrm:\t000001ffffffffff", ""),  # exact-key parse, no sibling CapXxx match
    ],
)
def test_probe_reads_the_effective_capability_mask(
    scrape: dict[str, Any], tmp_path: Path, cap_eff_line: str, expected_cap_eff: str
) -> None:
    # Arrange
    scrape["PROC_SELF_STATUS_PATH"] = _status_file(tmp_path, cap_eff_line)
    scrape["NVIDIACTL_PATH"] = str(tmp_path)

    # Act
    probe = scrape["probe_gpu_power_cap_ability"]()

    # Assert
    assert probe.cap_eff == expected_cap_eff
    if expected_cap_eff == "":
        assert "CapEff" in probe.scrape_error


def test_probe_reads_the_nvidiactl_owner_uid(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange - a real file stands in for /dev/nvidiactl; its owner is this test process' uid
    nvidiactl = tmp_path / "nvidiactl"
    nvidiactl.write_text("")
    scrape["PROC_SELF_STATUS_PATH"] = _status_file(tmp_path, f"CapEff:\t{FULL_CAPS}")
    scrape["NVIDIACTL_PATH"] = str(nvidiactl)

    # Act
    probe = scrape["probe_gpu_power_cap_ability"]()

    # Assert
    assert probe.nvidiactl_owner_uid == os.getuid()
    assert probe.scrape_error == ""


def test_probe_reports_both_readings_missing_without_guessing(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange - no /proc/self/status, no /dev/nvidiactl (the backend fails open on this)
    scrape["PROC_SELF_STATUS_PATH"] = str(tmp_path / "does-not-exist")
    scrape["NVIDIACTL_PATH"] = str(tmp_path / "does-not-exist-either")

    # Act
    probe = scrape["probe_gpu_power_cap_ability"]()

    # Assert
    assert probe.cap_eff == ""
    assert probe.nvidiactl_owner_uid is None
    assert "Cannot read" in probe.scrape_error and "Cannot stat" in probe.scrape_error


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


def test_probe_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped - keep all three in both."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = ["data_container_cap_eff", "data_nvidiactl_owner_uid", "data_power_cap_probe_error"]

    # Act
    original_keys = _dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = _dict_literal_keys(service_module, "all_keys")

    # Assert
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys
