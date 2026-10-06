"""DAH-2182 — see check_ncu_profiling_access() in machine_scrape.py for the RmProfilingAdminOnly
semantics these tests pin down.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are extracted by ast and executed in their own namespace (same pattern as
test_scrape_disk_breakdown.py). The obfuscation tables are parsed from source like
test_scrape_encryption_key_order.py does.
"""

import ast
import re
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"

NCU_HELPERS = {
    "NVIDIA_PARAMS_PATH",
    "BOOT_ID_PATH",
    "NcuProfilingObservation",
    "check_ncu_profiling_access",
    "get_host_boot_id",
}


@pytest.fixture
def scrape() -> dict[str, Any]:
    """The ncu helpers, executed in a namespace of their own."""
    return build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", NCU_HELPERS, {"re": re})


@pytest.mark.parametrize(
    ("flag_line", "expected_access"),
    [
        ("RmProfilingAdminOnly: 0", "unrestricted"),
        ("RmProfilingAdminOnly: 1", "restricted"),
        ("RmProfilingAdminOnly: 2", "unknown"),
        ("NotRmProfilingAdminOnly: 0", "unknown"),  # exact-key parse, no substring match
    ],
)
def test_profiling_access_parses_the_exact_flag(
    scrape: dict[str, Any], tmp_path: Path, flag_line: str, expected_access: str
) -> None:
    # Arrange - the flag sits among other params like in the real file
    params_file = tmp_path / "params"
    params_file.write_text(f"ResmanDebugLevel: 4294967295\n{flag_line}\nRegistryDwords: \"\"\n")
    scrape["NVIDIA_PARAMS_PATH"] = str(params_file)

    # Act
    observation = scrape["check_ncu_profiling_access"]()

    # Assert
    assert observation.access == expected_access
    if expected_access == "unknown":
        assert observation.scrape_error != ""
    else:
        assert observation.scrape_error == ""


def test_profiling_access_is_unknown_when_params_file_is_missing(
    scrape: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange - no NVIDIA driver loaded -> no params file
    scrape["NVIDIA_PARAMS_PATH"] = str(tmp_path / "does-not-exist")

    # Act
    observation = scrape["check_ncu_profiling_access"]()

    # Assert - fail closed, with the reason captured for the scrape_error key
    assert observation.access == "unknown"
    assert "Cannot read" in observation.scrape_error


def test_boot_id_is_read_and_stripped(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange
    boot_id_file = tmp_path / "boot_id"
    boot_id_file.write_text("ec8c2436-1039-45aa-a508-bcbe865fbcd8\n")
    scrape["BOOT_ID_PATH"] = str(boot_id_file)

    # Act / Assert
    assert scrape["get_host_boot_id"]() == "ec8c2436-1039-45aa-a508-bcbe865fbcd8"


def test_boot_id_is_empty_when_unreadable(scrape: dict[str, Any], tmp_path: Path) -> None:
    # Arrange
    scrape["BOOT_ID_PATH"] = str(tmp_path / "does-not-exist")

    # Act / Assert
    assert scrape["get_host_boot_id"]() == ""


def test_new_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped — keep all three in both."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = ["data_ncu_profiling_access", "data_ncu_profiling_scrape_error", "data_boot_id"]

    # Act
    original_keys = dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Assert - the scrape emits each key and both tables carry it
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys


def test_no_obfuscation_key_is_a_prefix_of_a_later_one() -> None:
    """ecrypt_miner_job_files() substitutes all_keys with a sequential str.replace in dict order,
    so an earlier key that prefixes a later one corrupts the later one before its turn."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())

    # Act
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Assert
    prefixed_pairs = [
        (earlier, later)
        for index, earlier in enumerate(all_keys)
        for later in all_keys[index + 1 :]
        if later.startswith(earlier)
    ]
    assert prefixed_pairs == []
