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
    # the validator image build sees only neurons/validators, so this copy carries the version into it
    "neurons/validators/validator_protocol": r"^(\d+\.\d+\.\d+)$",
}
SNAPSHOT_DIR = "lium_protocol/lium_protocol/snapshots"
SNAPSHOT_PATTERN = r'^  "protocol_version": "([^"]+)",$'
CHANGES = "lium_protocol/changes"


class Fragment(NamedTuple):
    bump: str
    changelog_entry: str


def parse_fragment(text: str, name: str) -> Fragment:
    """one fragment's bump and changelog entry from its text; SystemExit when it is malformed."""
    first, _, rest = text.strip().partition("\n")
    bump, entry = first.strip().lower(), " ".join(rest.split())
    if bump not in BUMPS or not entry:
        sys.exit(f"{name}: first line must be one of {BUMPS}, then a one-line changelog entry")
    return Fragment(bump, entry)


def read_fragment(path: Path) -> Fragment:
    return parse_fragment(path.read_text(), str(path))


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

    # every new text is computed before the first write, so a check that stops the release leaves the tree as it was
    version_texts = {
        root / name: replace_version((root / name).read_text(), pattern, old, new, name)
        for name, pattern in VERSION_FILES.items()
    }

    # the snapshot is named after the major version; a major release moves it to the new name
    old_snapshot = root / SNAPSHOT_DIR / f"lium_protocol.v{old.split('.')[0]}.json"
    new_snapshot = root / SNAPSHOT_DIR / f"lium_protocol.v{new.split('.')[0]}.json"
    snapshot_text = replace_version(old_snapshot.read_text(), SNAPSHOT_PATTERN, old, new, str(old_snapshot))

    # the version list: the new line goes above `1.0.0`, the list's last line
    readme = root / README
    line = f"- `{new}` — {' '.join(entry.changelog_entry for entry in entries)}\n"
    # README is a version file too: the list is built on its new text
    lines = version_texts.pop(readme, None) or readme.read_text()
    lines = lines.splitlines(keepends=True)
    first_release_line_index = next((i for i, x in enumerate(lines) if x.startswith("- `1.0.0` —")), None)
    if first_release_line_index is None:
        sys.exit(f"{README}: no `1.0.0` line to put the new version above")

    for path, text in version_texts.items():
        path.write_text(text)
    old_snapshot.unlink()
    new_snapshot.write_text(snapshot_text)
    readme.write_text("".join(lines[:first_release_line_index] + [line] + lines[first_release_line_index:]))

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
