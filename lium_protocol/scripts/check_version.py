"""CI check for lium_protocol versioning (stdlib only): `python lium_protocol/scripts/check_version.py <base>`.

Compares the tree at <base> with HEAD (`git diff <base> HEAD`; locally pass `$(git merge-base origin/main HEAD)`)
and fails when
1. the version changes in a PR that is not a release PR, or
2. the wire (anything under lium_protocol/lium_protocol/ but a version line, a comment or a blank line) changes and
   no fragment is added under lium_protocol/changes/.
A release PR is one that deletes at least one fragment: `scripts/release.py` does that, nothing else does.
"""

import re
import subprocess
import sys

WIRE = "lium_protocol/lium_protocol/"
CHANGES = "lium_protocol/changes/"
INIT = "lium_protocol/lium_protocol/__init__.py"
# changed lines that are not the wire: the version itself, comments, blank lines
NOT_WIRE = re.compile(r'^\s*(#.*|PROTOCOL_VERSION = "[^"]*"|"protocol_version": "[^"]*",?)?\s*$')


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def version(ref: str) -> str:
    return re.search(r'^PROTOCOL_VERSION = "([^"]+)"$', git("show", f"{ref}:{INIT}"), re.MULTILINE).group(1)


def problems(base: str) -> list[str]:
    status = [line.split("\t") for line in git("diff", "--name-status", "--no-renames", base, "HEAD").splitlines()]
    is_release = any(s == "D" and p.startswith(CHANGES) for s, p, *_ in status)
    adds_fragment = any(s in ("A", "M") and p.startswith(CHANGES) and p.endswith(".md") for s, p, *_ in status)
    found = []
    if not is_release and version(base) != version("HEAD"):
        found.append(
            f"PROTOCOL_VERSION changed {version(base)} → {version('HEAD')} outside a release PR: "
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
