"""DAH-2182: the scrape reports the host NVIDIA profiling-counter flag as a fail-closed
observation. `RmProfilingAdminOnly: 0` in /proc/driver/nvidia/params means the counters are open
to EVERY workload on the host — the backend then treats the node as whole-host-only. Anything
unreadable must surface as "unknown", never as a guessed value.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are extracted by ast and executed in their own namespace (same pattern as
test_scrape_disk_breakdown.py). The obfuscation tables are parsed from source like
test_scrape_encryption_key_order.py does.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

NCU_HELPERS = {
    "NVIDIA_PARAMS_PATH",
    "BOOT_ID_PATH",
    "check_ncu_profiling_access",
    "get_host_boot_id",
}


def _definition_name(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return ""


@pytest.fixture
def scrape() -> dict:
    """The ncu helpers, executed in a namespace of their own."""
    tree = ast.parse((SRC / "miner_jobs" / "machine_scrape.py").read_text())
    kept = [node for node in tree.body if _definition_name(node) in NCU_HELPERS]
    assert {_definition_name(node) for node in kept} == NCU_HELPERS, (
        f"machine_scrape.py no longer defines all of {sorted(NCU_HELPERS)} at module level"
    )

    namespace = {"re": re}
    exec(compile(ast.Module(body=kept, type_ignores=[]), "machine_scrape_ncu", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("flag_line", "expected_access"),
    [
        ("RmProfilingAdminOnly: 0", "unrestricted"),
        ("RmProfilingAdminOnly: 1", "restricted"),
        ("RmProfilingAdminOnly: 2", "unknown"),
        ("NotRmProfilingAdminOnly: 0", "unknown"),  # exact-key parse, no substring match
    ],
)
def test_profiling_access_parses_the_exact_flag(scrape: dict, tmp_path: Path, flag_line: str, expected_access: str) -> None:
    # Arrange - the flag sits among other params like in the real file
    params = tmp_path / "params"
    params.write_text(f"ResmanDebugLevel: 4294967295\n{flag_line}\nRegistryDwords: \"\"\n")
    scrape["NVIDIA_PARAMS_PATH"] = str(params)

    # Act
    access, log_text = scrape["check_ncu_profiling_access"]()

    # Assert
    assert access == expected_access
    # a clean 0/1 read carries no error text; every unknown explains itself
    assert (log_text == "") == (expected_access in ("unrestricted", "restricted"))


def test_profiling_access_is_unknown_when_params_file_is_missing(scrape: dict, tmp_path: Path) -> None:
    # Arrange - no NVIDIA driver loaded -> no params file
    scrape["NVIDIA_PARAMS_PATH"] = str(tmp_path / "does-not-exist")

    # Act
    access, log_text = scrape["check_ncu_profiling_access"]()

    # Assert - fail closed, with the reason captured for the scrape_error key
    assert access == "unknown"
    assert "Cannot read" in log_text


def test_boot_id_is_read_and_stripped(scrape: dict, tmp_path: Path) -> None:
    # Arrange
    boot_id_file = tmp_path / "boot_id"
    boot_id_file.write_text("ec8c2436-1039-45aa-a508-bcbe865fbcd8\n")
    scrape["BOOT_ID_PATH"] = str(boot_id_file)

    # Act / Assert
    assert scrape["get_host_boot_id"]() == "ec8c2436-1039-45aa-a508-bcbe865fbcd8"


def test_boot_id_is_empty_when_unreadable(scrape: dict, tmp_path: Path) -> None:
    # Arrange
    scrape["BOOT_ID_PATH"] = str(tmp_path / "does-not-exist")

    # Act / Assert
    assert scrape["get_host_boot_id"]() == ""


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


def test_new_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped — keep all three in both."""
    # Arrange
    service = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = ["data_ncu_profiling_access", "data_ncu_profiling_scrape_error", "data_boot_id"]

    # Act
    original_keys = _dict_literal_keys(service, "ORIGINAL_KEYS")
    all_keys = _dict_literal_keys(service, "all_keys")

    # Assert - the scrape emits each key and both tables carry it
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys
    # the replacement sweep is a sequential str.replace in dict order: no key of ours may be a
    # strict prefix of a LATER key in all_keys, or the later one gets corrupted before its turn
    for new_key in new_keys:
        later_keys = all_keys[all_keys.index(new_key) + 1 :]
        assert not any(later.startswith(new_key) for later in later_keys)
