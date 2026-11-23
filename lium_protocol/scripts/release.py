"""Cut a lium_protocol release from the change fragments (stdlib only).

Run from anywhere inside the repository: `python lium_protocol/scripts/release.py`. It takes the highest bump
among `lium_protocol/changes/*.md` (first line `minor` or `major`), writes the new version to every file that
carries it, adds one line to the README's version list and deletes the fragments. Commit the result as the
release PR; tag the merge commit `lium-protocol-v<version>`.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

BUMPS = ("minor", "major")
INIT = "lium_protocol/lium_protocol/__init__.py"
README = "lium_protocol/README.md"
# file → the one pattern whose group 1 is the version
VERSION_FILES = {
    INIT: r'^PROTOCOL_VERSION = "([^"]+)"$',
    "lium_protocol/pyproject.toml": r'^version = "([^"]+)"$',
    "lium_protocol/uv.lock": r'^name = "lium-protocol"\nversion = "([^"]+)"$',
    README: r"`PROTOCOL_VERSION` \(`([^`]+)`\)",
}
SNAPSHOT_DIR = "lium_protocol/lium_protocol/snapshots"
SNAPSHOT_PATTERN = r'^  "protocol_version": "([^"]+)",$'
CHANGES = "lium_protocol/changes"


class Fragment(NamedTuple):
    bump: str
    changelog_entry: str


def read_fragment(path: Path) -> Fragment:
    """one fragment's bump and changelog entry; SystemExit when it is malformed."""
    first, _, rest = path.read_text().strip().partition("\n")
    bump, text = first.strip().lower(), " ".join(rest.split())
    if bump not in BUMPS or not text:
        sys.exit(f"{path}: first line must be one of {BUMPS}, then a one-line changelog entry")
    return Fragment(bump, text)


def bumped(version: str, bump: str) -> str:
    major, minor, _patch = (int(part) for part in version.split("."))
    return f"{major + 1}.0.0" if bump == "major" else f"{major}.{minor + 1}.0"


def replace_version(text: str, pattern: str, old: str, new: str, name: str) -> str:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1 or matches[0].group(1) != old:
        sys.exit(f"{name}: expected one version {old}, found {[m.group(1) for m in matches]}")
    start, end = matches[0].span(1)
    return text[:start] + new + text[end:]


def release_and_return_new_version(root: Path) -> str:
    fragments = sorted((root / CHANGES).glob("*.md"))
    if not fragments:
        sys.exit(f"no fragments in {CHANGES}/: nothing to release")
    entries = [read_fragment(path) for path in fragments]
    bump = max((entry.bump for entry in entries), key=BUMPS.index)
    old = re.search(VERSION_FILES[INIT], (root / INIT).read_text(), re.MULTILINE).group(1)
    new = bumped(old, bump)

    for name, pattern in VERSION_FILES.items():
        path = root / name
        path.write_text(replace_version(path.read_text(), pattern, old, new, name))

    # the snapshot is named after the major version; a major release moves it to the new name
    old_snapshot = root / SNAPSHOT_DIR / f"lium_protocol.v{old.split('.')[0]}.json"
    new_snapshot = root / SNAPSHOT_DIR / f"lium_protocol.v{new.split('.')[0]}.json"
    text = replace_version(old_snapshot.read_text(), SNAPSHOT_PATTERN, old, new, str(old_snapshot))
    old_snapshot.unlink()
    new_snapshot.write_text(text)

    # the version list: the new line goes above `1.0.0`, the list's last line
    readme = root / README
    line = f"- `{new}` — {' '.join(entry.changelog_entry for entry in entries)}\n"
    lines = readme.read_text().splitlines(keepends=True)
    at = next((i for i, x in enumerate(lines) if x.startswith("- `1.0.0` —")), None)
    if at is None:
        sys.exit(f"{README}: no `1.0.0` line to put the new version above")
    readme.write_text("".join(lines[:at] + [line] + lines[at:]))

    for path in fragments:
        path.unlink()
    return new


def main() -> None:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    new = release_and_return_new_version(root)
    print(
        f"lium_protocol {new}. After the release PR merges: "
        f"git tag lium-protocol-v{new} <merge commit> && git push origin lium-protocol-v{new}"
    )


if __name__ == "__main__":
    main()
