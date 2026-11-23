"""CI check for lium_protocol versioning (stdlib only): `python lium_protocol/scripts/check_version.py <base>`.

Compares the tree at <base> with HEAD (`git diff <base> HEAD`; CI passes the commit the PR merges onto, in the merge
queue too, so each PR is judged on its own diff; locally pass `$(git merge-base origin/main HEAD)`) and fails when
1. the version changes in a PR that is not a release PR,
2. a release PR does not raise the version above the base's (a release rebased onto another one and resolved to the
   base's version would name two wires with one version),
3. a release PR leaves a fragment that is on its base (a feature merged after the release was cut: its change would
   ship under the release's tag without its bump), or
4. the wire changes and no fragment is added under lium_protocol/changes/. The wire is the committed schema snapshot
   (CI holds it equal to the models: `python -m lium_protocol.schema --check`) but its protocol_version line, so a
   comment or a docstring that is not in the schema needs no fragment.
A release PR is one that deletes at least one fragment: `scripts/release.py` does that, nothing else does.
"""

import re
import subprocess
import sys

WIRE = "lium_protocol/lium_protocol/snapshots/"
CHANGES = "lium_protocol/changes/"
INIT = "lium_protocol/lium_protocol/__init__.py"
# changed snapshot lines that are not the wire: the version itself, blank lines
NOT_WIRE = re.compile(r'^\s*("protocol_version": "[^"]*",?)?\s*$')


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def version(ref: str) -> str:
    return re.search(r'^PROTOCOL_VERSION = "([^"]+)"$', git("show", f"{ref}:{INIT}"), re.MULTILINE).group(1)


def fragments(ref: str) -> set[str]:
    return {p for p in git("ls-tree", "-r", "--name-only", ref, "--", f":(top){CHANGES}").splitlines() if p.endswith(".md")}


def _key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def problems(base: str) -> list[str]:
    status = [line.split("\t") for line in git("diff", "--name-status", "--no-renames", base, "HEAD").splitlines()]
    is_release = any(s == "D" and p.startswith(CHANGES) for s, p, *_ in status)
    adds_fragment = any(s in ("A", "M") and p.startswith(CHANGES) and p.endswith(".md") for s, p, *_ in status)
    found = []
    old, new = version(base), version("HEAD")
    if is_release and _key(new) <= _key(old):
        found.append(
            f"a release PR must raise PROTOCOL_VERSION above the base's {old} (it is {new}): "
            "run scripts/release.py again on the current main"
        )
    kept = sorted(fragments(base) & fragments("HEAD")) if is_release else []
    if kept:
        found.append(
            f"a release PR must take every fragment on its base, and {', '.join(kept)} stays: "
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
    if wire and not adds_fragment:
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
