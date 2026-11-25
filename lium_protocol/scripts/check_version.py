"""CI check for lium_protocol versioning (stdlib only): `python lium_protocol/scripts/check_version.py <base>`.

Compares the tree at <base> with HEAD (`git diff <base> HEAD`; CI passes the commit the PR merges onto, in the merge
queue too, so each PR is judged on its own diff; locally pass `$(git merge-base origin/main HEAD)`) and fails when
1. the version changes in a PR that is not a release PR,
2. a release PR does not set exactly the version its base's fragments call for: the highest bump among them applied
   to the base's version (`release.py`'s own rule; a release rebased onto another one, or a hand-set version, names
   the wrong wire),
3. a release PR leaves a fragment that is on its base (a feature merged after the release was cut: its change would
   ship under the release's tag without its bump),
4. a PR edits a fragment that is already on its base, or adds one `release.py` cannot read (it would stop the
   release later) or does not read (a file in a subfolder of changes/),
5. a release PR changes the wire (a change shipped under the release's tag with no fragment of its own), or
6. the wire changes and no fragment is added under lium_protocol/changes/. The wire is the committed schema snapshot
   (CI holds it equal to the models: `python -m lium_protocol.schema --check`) but its protocol_version line, so a
   comment or a docstring that is not in the schema needs no fragment.
A release PR is one that deletes at least one fragment: `scripts/release.py` does that, nothing else does.
"""

import re
import subprocess
import sys

from release import BUMPS, Fragment, bumped, parse_fragment  # scripts/release.py: one fragment format, one bump rule

WIRE = "lium_protocol/lium_protocol/snapshots/"
CHANGES = "lium_protocol/changes/"
INIT = "lium_protocol/lium_protocol/__init__.py"
# changed snapshot lines that are not the wire: the version itself, blank lines
NOT_WIRE = re.compile(r'^\s*("protocol_version": "[^"]*",?)?\s*$')


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def version(ref: str) -> str:
    return re.search(r'^PROTOCOL_VERSION = "([^"]+)"$', git("show", f"{ref}:{INIT}"), re.MULTILINE).group(1)


def is_fragment(path: str) -> bool:
    # release.py reads only changes/*.md, so a file in a subfolder is not a fragment
    return path.startswith(CHANGES) and "/" not in path[len(CHANGES):] and path.endswith(".md")


def fragments(ref: str) -> set[str]:
    return {p for p in git("ls-tree", "-r", "--name-only", "--full-name", ref, "--", f":(top){CHANGES}").splitlines() if is_fragment(p)}


def parse_fragment_at_ref(ref: str, path: str) -> Fragment | str:
    """the parsed fragment at ref, or the error text release.py would stop on"""
    try:
        return parse_fragment(git("show", f"{ref}:{path}"), path)
    except SystemExit as stop:
        return str(stop)


def problems(base: str) -> list[str]:
    status = [line.split("\t") for line in git("diff", "--name-status", "--no-renames", base, "HEAD").splitlines()]
    is_release = any(s == "D" and is_fragment(p) for s, p, *_ in status)
    new_fragments = [p for s, p, *_ in status if s == "A" and is_fragment(p)]
    found = []
    for s, p, *_ in status:
        if s != "D" and p.startswith(CHANGES) and not is_fragment(p):
            found.append(f"{p} is not a fragment release.py reads: put it directly under {CHANGES} as <name>.md")
        elif s not in ("A", "D") and is_fragment(p):
            found.append(f"{p} is a fragment already on the base: add a new fragment instead of editing it")
    for p in new_fragments:
        parsed = parse_fragment_at_ref("HEAD", p)
        if isinstance(parsed, str):
            found.append(parsed)
    old, new = version(base), version("HEAD")
    if is_release:
        parsed_base = [parse_fragment_at_ref(base, p) for p in sorted(fragments(base))]
        base_bumps = [f.bump for f in parsed_base if not isinstance(f, str)]
        due = bumped(old, max(base_bumps, key=BUMPS.index)) if base_bumps else None
        if new != due:
            found.append(
                f"a release PR must set PROTOCOL_VERSION to {due} (the highest bump among the base's fragments applied "
                f"to {old}), and it is {new}: run scripts/release.py again on the current main"
            )
    base_fragments_left_by_release = sorted(fragments(base) & fragments("HEAD")) if is_release else []
    if base_fragments_left_by_release:
        found.append(
            f"a release PR must take every fragment on its base, and {', '.join(base_fragments_left_by_release)} stays: "
            "run scripts/release.py again on the current main"
        )
    if not is_release and old != new:
        found.append(
            f"PROTOCOL_VERSION changed {old} → {new} outside a release PR: "
            "add a fragment under lium_protocol/changes/ instead; the release sets the version"
        )
    # -M: a major release renames the snapshot, which must read as a version-line change only
    diff = git("diff", "-U0", "-M", base, "HEAD", "--", f":(top){WIRE}").splitlines()
    changed = [x for x in diff if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))]
    wire = [x for x in changed if not NOT_WIRE.match(x[1:])]
    if wire and is_release:
        found.append(
            f"a release PR changed the wire ({wire[0][:80]!r}…): ship the change in its own PR with a fragment, "
            "then release it"
        )
    elif wire and not new_fragments:
        found.append(
            f"the wire changed ({wire[0][:80]!r}…) without a fragment: add lium_protocol/changes/<name>.md "
            "(first line `minor` or `major`, then one changelog line)"
        )
    return found


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    found = problems(sys.argv[1])
    for problem in found:
        print(f"::error::{problem}")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
