#!/usr/bin/env python3
"""Fail when a daturaai/ image tag or digest pinned in the tree is not on Docker Hub.

Providers install from `main`, so a compose file or install script that names a tag nobody has
pushed yet breaks `docker compose up` for every new executor the moment it merges. This reads
every `daturaai/` reference in the compose files, the executor YAML, the shell scripts and the
Dockerfiles, dedupes them, and asks Docker Hub for each manifest with an anonymous HEAD request
(no pull, no login; HEAD requests do not count against Docker Hub's pull rate limit).

A reference whose tag or digest is a shell or compose variable (`:${TAG}`, `@$DIGEST`) is
resolved at run time and is listed as skipped. A reference with no tag is checked as `latest`.

Exit status: 0 every reference exists, 1 at least one is missing, 2 Docker Hub could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCAN_GLOBS = (
    "neurons/**/docker-compose*.yml",
    "neurons/executor/**/*.yml",
    "neurons/**/*.sh",
    "neurons/**/Dockerfile*",
    "scripts/**/*.sh",
    "watchtower/**/*.sh",
    "watchtower/**/Dockerfile*",
)

# Bounded classes only: the input is repository text, but the scan runs on every PR.
REF_RE = re.compile(
    r"(?<![\w./-])"
    r"(?:(?:docker\.io|index\.docker\.io|registry-1\.docker\.io)/)?"
    r"(?P<name>daturaai/[a-z0-9][a-z0-9._/-]{0,127})(?P<namevar>\$)?"
    r"(?:(?P<tagsep>:)(?P<tag>\$|[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?)?"
    r"(?:(?P<digsep>@)(?P<digest>\$|sha256:[0-9a-f]{64})?)?"
)
COMMENT_RE = re.compile(r"(?:^|\s)#.*$")

MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
TOKEN_URL = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
MANIFEST_URL = "https://registry-1.docker.io/v2/{repo}/manifests/{ref}"
ATTEMPTS = 3


@dataclass(frozen=True)
class Location:
    path: str
    line: int


@dataclass
class ImageRef:
    repository: str
    reference: str | None
    variable: bool = False
    locations: list[Location] = field(default_factory=list)

    @property
    def display(self) -> str:
        if self.variable:
            return f"{self.repository} (variable tag or digest)"
        sep = "@" if self.reference and self.reference.startswith("sha256:") else ":"
        return f"{self.repository}{sep}{self.reference}"


class RegistryUnavailable(Exception):
    pass


def strip_comment(line: str) -> str:
    return COMMENT_RE.sub("", line)


def refs_in_text(text: str) -> list[tuple[int, str, str | None, bool]]:
    """(line, repository, tag-or-digest, variable) for every daturaai/ reference outside comments."""
    found = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        for m in REF_RE.finditer(strip_comment(raw)):
            name = m.group("name").rstrip("._/-")
            tag, digest = m.group("tag"), m.group("digest")
            if m.group("namevar") or tag == "$" or digest == "$":
                found.append((lineno, name, None, True))
            elif (m.group("tagsep") and not tag) or (m.group("digsep") and not digest):
                # `daturaai/x:<version>` or a truncated digest in prose is a placeholder, not a pin
                continue
            else:
                found.append((lineno, name, digest or tag or "latest", False))
    return found


def collect(root: Path) -> list[ImageRef]:
    files = sorted({p for g in SCAN_GLOBS for p in root.glob(g) if p.is_file()})
    refs: dict[tuple[str, str | None, bool], ImageRef] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        for lineno, repo, ref, variable in refs_in_text(path.read_text(errors="replace")):
            key = (repo, ref, variable)
            refs.setdefault(key, ImageRef(repo, ref, variable)).locations.append(
                Location(rel, lineno)
            )
    return sorted(refs.values(), key=lambda r: (r.variable, r.repository, r.reference or ""))


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None):
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        req = urllib.request.Request(url, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                return e.code, b""
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(2 ** (attempt + 1))
    raise RegistryUnavailable(f"{method} {url}: {last}")


def manifest_exists(repository: str, reference: str, tokens: dict[str, str]) -> bool:
    if repository not in tokens:
        status, body = _request(TOKEN_URL.format(repo=repository))
        if status != 200:
            raise RegistryUnavailable(f"token for {repository}: HTTP {status}")
        tokens[repository] = json.loads(body)["token"]
    status, _ = _request(
        MANIFEST_URL.format(repo=repository, ref=reference),
        method="HEAD",
        headers={"Authorization": f"Bearer {tokens[repository]}", "Accept": MANIFEST_ACCEPT},
    )
    if status == 200:
        return True
    # 404: no such tag or digest; 401: no such (public) repository
    if status in (401, 404):
        return False
    raise RegistryUnavailable(f"HEAD {repository}@{reference}: HTTP {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=REPO, help="tree to scan (default: this repository)"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the references and exit, no network"
    )
    args = parser.parse_args(argv)

    refs = collect(args.root.resolve())
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    missing = 0
    tokens: dict[str, str] = {}
    for ref in refs:
        where = ", ".join(f"{loc.path}:{loc.line}" for loc in ref.locations)
        if ref.variable:
            print(f"SKIP     {ref.display} — resolved at run time — {where}")
            continue
        if args.list:
            print(f"PINNED   {ref.display} — {where}")
            continue
        try:
            ok = manifest_exists(ref.repository, ref.reference, tokens)
        except RegistryUnavailable as e:
            print(f"::error::Docker Hub could not be read: {e}" if annotate else f"ERROR    {e}")
            return 2
        print(f"{'OK' if ok else 'MISSING':<8} {ref.display} — {where}")
        if not ok:
            missing += 1
            if annotate:
                for loc in ref.locations:
                    print(
                        f"::error file={loc.path},line={loc.line}::{ref.display} is not on Docker Hub; "
                        "publish it before this change merges"
                    )
    checked = sum(1 for r in refs if not r.variable)
    skipped = len(refs) - checked
    if args.list:
        print(f"{checked} pinned, {skipped} skipped")
        return 0
    print(f"{checked} pinned references checked, {missing} missing, {skipped} skipped (variable)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
