"""restore_dirty_defaults.sh: after the DAH-4978 revert, a host still carrying the 1 GiB / 256 MiB cap
written by executor-v1.151 goes back to the kernel's ratio defaults; any other cap is the provider's.
The script runs against a fixture /proc tree under DIRTY_CAP_SYS_ROOT."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "restore_dirty_defaults.sh"
GIB = 1024**3


@pytest.fixture
def vm(tmp_path: Path) -> Path:
    vm_dir = tmp_path / "proc" / "sys" / "vm"
    vm_dir.mkdir(parents=True)
    for name, value in (("dirty_bytes", 0), ("dirty_background_bytes", 0), ("dirty_ratio", 20), ("dirty_background_ratio", 10)):
        (vm_dir / name).write_text(f"{value}\n")
    return vm_dir


def set_cap(vm: Path, dirty_bytes: int, background_bytes: int) -> None:
    (vm / "dirty_bytes").write_text(f"{dirty_bytes}\n")
    (vm / "dirty_background_bytes").write_text(f"{background_bytes}\n")
    (vm / "dirty_ratio").write_text("0\n")
    (vm / "dirty_background_ratio").write_text("0\n")


def read(vm: Path, name: str) -> int:
    return int((vm / name).read_text())


def run_restore(vm: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "DIRTY_CAP_SYS_ROOT": str(vm.parents[2])},
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_our_cap_is_replaced_by_the_ratio_defaults(vm: Path) -> None:
    set_cap(vm, GIB, GIB // 4)

    completed = run_restore(vm)

    assert completed.returncode == 0, completed.stderr
    assert (read(vm, "dirty_ratio"), read(vm, "dirty_background_ratio")) == (20, 10)
    assert "[dirty-cap] restored dirty_ratio=20 dirty_background_ratio=10" in completed.stdout


def test_a_providers_own_cap_is_left_alone(vm: Path) -> None:
    set_cap(vm, 4 * GIB, GIB)

    completed = run_restore(vm)

    assert completed.returncode == 0, completed.stderr
    assert (read(vm, "dirty_bytes"), read(vm, "dirty_ratio")) == (4 * GIB, 0)
    assert completed.stdout == ""


def test_a_host_already_on_ratios_is_not_touched(vm: Path) -> None:
    completed = run_restore(vm)

    assert completed.returncode == 0, completed.stderr
    assert (read(vm, "dirty_ratio"), read(vm, "dirty_background_ratio")) == (20, 10)
    assert completed.stdout == ""


def test_an_unreadable_proc_fails_open_with_a_warning(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "DIRTY_CAP_SYS_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    assert "[dirty-cap] cannot read" in completed.stdout
