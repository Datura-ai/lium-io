"""One version, one change. Two open pull requests that both take the next minor merge without a conflict on the
`PROTOCOL_VERSION` line (the same edit on both sides), and the result names two different wires by one tag. Each
change adds its own entry under `## Versions`, so the merged tree then lists that version twice: this fails there,
in the merge queue, and the PR that merges second takes the next unused minor. The other places the version is
written must agree with `PROTOCOL_VERSION`."""

import json
import re
import tomllib
from pathlib import Path

from lium_protocol import PROTOCOL_VERSION
from lium_protocol.schema import snapshot_path

ROOT = Path(__file__).resolve().parent.parent


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _listed_versions() -> list[str]:
    readme = (ROOT / "README.md").read_text()
    section = readme.split("\n## Versions\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^- `(\d+\.\d+\.\d+)` — ", section, flags=re.MULTILINE)


def test_no_version_is_listed_twice():
    listed = _listed_versions()
    assert listed, "README.md lists no versions under '## Versions'"
    twice = sorted({version for version in listed if listed.count(version) > 1})
    assert not twice, f"versions claimed by more than one change: {twice}; the later one takes the next unused minor"


def test_the_current_version_is_the_newest_listed():
    assert max(_listed_versions(), key=_version) == PROTOCOL_VERSION


def test_every_place_names_the_same_version():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    lock = re.search(r'name = "lium-protocol"\nversion = "([^"]+)"', (ROOT / "uv.lock").read_text())
    readme = re.search(r"`PROTOCOL_VERSION` \(`([^`]+)`\)", (ROOT / "README.md").read_text())
    snapshot = json.loads(snapshot_path().read_text())["protocol_version"]
    assert (pyproject, lock and lock.group(1), readme and readme.group(1), snapshot) == (PROTOCOL_VERSION,) * 4
