"""Verify the vendored liumd binary against the PR artifact that produced it.

The image build runs this after installing /usr/local/bin/liumd. It checks the target,
the embedded build sha and the binary's sha256 against neurons/executor/liumd/SOURCE.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIUMD = HERE / "liumd"
BINARY = Path("/usr/local/bin/liumd")


def main(target: str) -> None:
    source = json.loads((LIUMD / "SOURCE.json").read_text())
    entry = source["targets"][target]
    version = json.loads(subprocess.check_output([BINARY, "version"], text=True))
    if version["target"] != target:
        raise SystemExit(f"liumd target {version['target']!r}, SOURCE says {target!r}")
    if version["build_sha"] != entry["build_sha"]:
        raise SystemExit(f"liumd build_sha {version['build_sha']!r}, SOURCE says {entry['build_sha']!r}")
    digest = hashlib.sha256((LIUMD / entry["artifact"]).read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"liumd sha256 {digest}, SOURCE says {entry['sha256']}")
    print(f"liumd {version['version']} {target} build {version['build_sha']} verified")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_liumd.py <target-triple>")
    main(sys.argv[1])
