"""scripts/release.py and scripts/check_version.py, in a throwaway git repository holding copies of the real
version files (so the patterns are tested against the files they rewrite)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lium_protocol import PROTOCOL_VERSION

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "scripts"
MAJOR = int(PROTOCOL_VERSION.split(".")[0])
SNAPSHOT = "lium_protocol/lium_protocol/snapshots/lium_protocol.v{}.json"
FILES = [
    "lium_protocol/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    f"lium_protocol/snapshots/lium_protocol.v{MAJOR}.json",
    "lium_protocol/http.py",
]


def run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=repo, capture_output=True, text=True)


def commit(repo: Path, message: str) -> None:
    run(repo, "git", "add", "-A")
    run(repo, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name in FILES:
        (tmp_path / "lium_protocol" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PACKAGE / name, tmp_path / "lium_protocol" / name)
    run(tmp_path, "git", "init", "-q")
    commit(tmp_path, "base")
    return tmp_path


def fragment(repo: Path, name: str, bump: str, text: str = "a change.") -> None:
    (repo / "lium_protocol/changes").mkdir(exist_ok=True)
    (repo / f"lium_protocol/changes/{name}.md").write_text(f"{bump}\n{text}\n")


def release(repo: Path) -> subprocess.CompletedProcess:
    return run(repo, sys.executable, str(SCRIPTS / "release.py"))


def check(repo: Path) -> subprocess.CompletedProcess:
    commit(repo, "change")
    # from the package directory, as CI runs it
    return run(repo / "lium_protocol", sys.executable, str(SCRIPTS / "check_version.py"), "HEAD^")


def change_wire(repo: Path) -> None:
    # the wire is the committed snapshot (CI holds it equal to the models)
    path = repo / SNAPSHOT.format(MAJOR)
    path.write_text(path.read_text().replace('  "$defs": {\n', '  "$defs": {\n    "NewField": {},\n', 1))


def next_minor() -> str:
    major, minor, _ = PROTOCOL_VERSION.split(".")
    return f"{major}.{int(minor) + 1}.0"


def test_release_takes_the_highest_bump_writes_every_file_and_removes_fragments(repo: Path) -> None:
    fragment(repo, "a", "minor", "`X.a` (optional).")
    fragment(repo, "b", "major", "`Y` removed.")
    result = release(repo)
    assert result.returncode == 0, result.stderr
    new = f"{MAJOR + 1}.0.0"
    root = repo / "lium_protocol"
    assert f'PROTOCOL_VERSION = "{new}"' in (root / "lium_protocol/__init__.py").read_text()
    assert f'version = "{new}"' in (root / "pyproject.toml").read_text()
    assert f'name = "lium-protocol"\nversion = "{new}"' in (root / "uv.lock").read_text()
    readme = (root / "README.md").read_text()
    assert f"`PROTOCOL_VERSION` (`{new}`)" in readme
    assert f"- `{new}` — `X.a` (optional). `Y` removed.\n- `1.0.0` —" in readme
    # a major release moves the snapshot to the new major's name
    assert not (repo / SNAPSHOT.format(MAJOR)).exists()
    assert f'"protocol_version": "{new}"' in (repo / SNAPSHOT.format(MAJOR + 1)).read_text()
    assert list((root / "changes").iterdir()) == []
    assert f"lium-protocol-v{new}" in result.stdout


def test_release_minor(repo: Path) -> None:
    fragment(repo, "a", "minor")
    fragment(repo, "b", "minor")
    assert release(repo).returncode == 0
    assert f'"protocol_version": "{next_minor()}"' in (repo / SNAPSHOT.format(MAJOR)).read_text()


def test_release_refuses_no_fragments_and_a_bad_bump(repo: Path) -> None:
    assert release(repo).returncode != 0
    fragment(repo, "a", "patchy")
    assert release(repo).returncode != 0
    assert f'PROTOCOL_VERSION = "{PROTOCOL_VERSION}"' in (repo / "lium_protocol/lium_protocol/__init__.py").read_text()


def test_wire_change_with_a_fragment_passes(repo: Path) -> None:
    change_wire(repo)
    fragment(repo, "new-field", "minor")
    result = check(repo)
    assert result.returncode == 0, result.stdout


def test_wire_change_without_a_fragment_fails(repo: Path) -> None:
    change_wire(repo)
    result = check(repo)
    assert result.returncode == 1 and "without a fragment" in result.stdout


def test_comment_only_change_needs_no_fragment(repo: Path) -> None:
    path = repo / "lium_protocol/lium_protocol/http.py"
    path.write_text(path.read_text() + "\n# a note\n")
    assert check(repo).returncode == 0


def test_a_source_change_the_snapshot_does_not_show_needs_no_fragment(repo: Path) -> None:
    """a docstring or code comment that is not in the schema (taiberium on lium-io#2376: the schema.py docstring fix)"""
    path = repo / "lium_protocol/lium_protocol/http.py"
    path.write_text(path.read_text() + '\n\ndef helper():\n    """a reworded docstring."""\n')
    assert check(repo).returncode == 0


def test_version_change_in_a_feature_pr_fails(repo: Path) -> None:
    path = repo / "lium_protocol/lium_protocol/__init__.py"
    path.write_text(path.read_text().replace(PROTOCOL_VERSION, next_minor()))
    fragment(repo, "new-field", "minor")
    result = check(repo)
    assert result.returncode == 1 and "outside a release PR" in result.stdout


@pytest.mark.parametrize("bump", ["minor", "major"])
def test_release_pr_passes(repo: Path, bump: str) -> None:
    change_wire(repo)
    fragment(repo, "new-field", bump)
    commit(repo, "feature")
    assert release(repo).returncode == 0
    result = check(repo)
    assert result.returncode == 0, result.stdout


def test_a_release_that_keeps_the_base_version_fails(repo: Path) -> None:
    """a release rebased onto another release and resolved to the base's version deletes fragments but names no new wire"""
    change_wire(repo)
    fragment(repo, "new-field", "minor")
    commit(repo, "feature")
    (repo / "lium_protocol/changes/new-field.md").unlink()
    result = check(repo)
    assert result.returncode == 1 and "must raise PROTOCOL_VERSION" in result.stdout


def test_a_release_that_lowers_the_version_fails(repo: Path) -> None:
    fragment(repo, "new-field", "minor")
    commit(repo, "feature")
    (repo / "lium_protocol/changes/new-field.md").unlink()
    path = repo / "lium_protocol/lium_protocol/__init__.py"
    major, minor, patch = PROTOCOL_VERSION.split(".")
    lower = f"{int(major) - 1}.{minor}.{patch}" if int(major) > 0 else f"{major}.{minor}.{max(int(patch) - 1, 0)}"
    path.write_text(path.read_text().replace(PROTOCOL_VERSION, lower))
    result = check(repo)
    assert result.returncode == 1 and "must raise PROTOCOL_VERSION" in result.stdout


def test_a_release_that_keeps_a_fragment_of_its_base_fails(repo: Path) -> None:
    """taiberium on lium-io#2376: the release was cut before a major feature merged; merged onto that base, the release
    takes only its own fragment and the feature's major change would ship under the minor tag"""
    fragment(repo, "small", "minor")
    commit(repo, "a minor feature")
    run(repo, "git", "branch", "release")
    change_wire(repo)
    fragment(repo, "breaking", "major")
    commit(repo, "a major feature merged after the release was cut")
    base = run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    run(repo, "git", "checkout", "-q", "release")
    assert release(repo).returncode == 0
    commit(repo, "release")
    run(repo, "git", "checkout", "-q", base)
    run(repo, "git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", "-m", "merge", "release")
    result = run(repo / "lium_protocol", sys.executable, str(SCRIPTS / "check_version.py"), "HEAD^1")
    assert result.returncode == 1 and "breaking.md stays" in result.stdout
