"""Backup and restore never follow links out of the workspace."""

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from storage.safe_path import (
    NOT_A_DIRECTORY_EXIT,
    NOT_EMPTY_EXIT,
    PATH_MISSING_EXIT,
    SYMLINK_EXIT,
    safe_enter_command,
)


def _run(root: Path, path: Path, mode: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    argv = safe_enter_command(PurePosixPath(str(root)), PurePosixPath(str(path)), mode, command)
    return subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True)


def _pwd(out: Path) -> list[str]:
    return [sys.executable, "-c", f"import os; open({str(out)!r}, 'w').write(os.getcwd())"]


@pytest.fixture
def rental(tmp_path: Path) -> Path:
    root = tmp_path / "rental"
    (root / "root" / "data").mkdir(parents=True)
    (root / "root" / "data" / "f").write_text("x")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_text("host")
    return root


@pytest.mark.parametrize("target", ["/proc/self", "../../outside", "/"])
def test_a_symlink_anywhere_in_the_path_is_refused_and_nothing_runs(
    rental: Path, tmp_path: Path, target: str
) -> None:
    os.symlink(target, rental / "root" / "link_self")
    out = tmp_path / "ran"
    for mode in ("backup", "restore", "check-backup"):
        done = _run(rental, rental / "root" / "link_self" / "x", mode, _pwd(out))
        assert done.returncode == SYMLINK_EXIT, (mode, done.stderr)
        assert "symlink in path: link_self" in done.stderr
    assert not out.exists()
    assert not (tmp_path / "outside" / "x").exists()  # restore created nothing through the link


def test_a_symlink_as_the_last_component_is_refused(rental: Path) -> None:
    os.symlink(str(rental.parent / "outside"), rental / "root" / "out")
    assert _run(rental, rental / "root" / "out", "backup", ["true"]).returncode == SYMLINK_EXIT


def test_backup_runs_inside_the_real_directory(rental: Path, tmp_path: Path) -> None:
    out = tmp_path / "cwd"
    done = _run(rental, rental / "root" / "data", "backup", _pwd(out))
    assert done.returncode == 0, done.stderr
    assert out.read_text() == str(rental / "root" / "data")


def test_backup_of_a_missing_or_empty_path_is_a_no_op(rental: Path) -> None:
    (rental / "root" / "empty").mkdir()
    assert (
        _run(rental, rental / "root" / "missing", "backup", ["false"]).returncode
        == PATH_MISSING_EXIT
    )
    assert (
        _run(rental, rental / "root" / "empty", "backup", ["false"]).returncode == PATH_MISSING_EXIT
    )


def test_a_file_is_not_a_directory(rental: Path) -> None:
    assert (
        _run(rental, rental / "root" / "data" / "f", "backup", ["true"]).returncode
        == NOT_A_DIRECTORY_EXIT
    )


def test_restore_creates_the_missing_target_and_runs_there(rental: Path, tmp_path: Path) -> None:
    out = tmp_path / "cwd"
    target = rental / "root" / "new" / "deep"
    assert _run(rental, target, "restore", _pwd(out)).returncode == 0
    assert out.read_text() == str(target)


def test_restore_empty_refuses_a_non_empty_target_and_checks_create_nothing(rental: Path) -> None:
    assert (
        _run(rental, rental / "root" / "data", "restore-empty", ["true"]).returncode
        == NOT_EMPTY_EXIT
    )
    assert (
        _run(rental, rental / "root" / "data", "check-restore-empty", []).returncode
        == NOT_EMPTY_EXIT
    )
    assert _run(rental, rental / "root" / "absent", "check-restore-empty", []).returncode == 0
    assert not (rental / "root" / "absent").exists()


def test_dot_dot_is_refused(rental: Path) -> None:
    argv = safe_enter_command(
        PurePosixPath(str(rental)), PurePosixPath(str(rental)), "backup", ["true"]
    )
    argv[6] = "root/../.."
    assert subprocess.run([sys.executable, *argv[1:]]).returncode == SYMLINK_EXIT


def test_the_whole_volume_is_the_root_itself(rental: Path, tmp_path: Path) -> None:
    out = tmp_path / "cwd"
    for mode in ("backup", "restore", "check-backup", "check-restore"):
        done = _run(rental, rental, mode, _pwd(out))
        assert done.returncode == 0, (mode, done.stderr)
        assert out.read_text() == str(rental), mode
