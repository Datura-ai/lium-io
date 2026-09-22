#!/usr/bin/env python3
"""Check that every image this repository pins exists in its registry.

A provider installs from main and runs `docker compose up`; a compose file that names a tag nobody
pushed yet (daturaai/lium-watchtower:1.2.0 on lium-io#1432) breaks that for every new install. This
lists every image reference in the tracked files and runs `docker manifest inspect` on each:

- `image:` values in any *.yml / *.yaml (compose files, workflow service containers), plus
  upper-case `*IMAGE` / `*IMAGE_REF` keys (Pulumi stack config);
- `FROM` in Dockerfile*, with global `ARG` defaults substituted.

Skipped, and listed as skipped: a reference with a variable nothing here resolves (`${EXECUTOR_IMAGE_SHA256}`,
`${{ ... }}`); an image a compose service builds itself (a service with `build:`) and every other use of
that tag; a `FROM` that names an earlier stage, a compose `additional_contexts` name, or `scratch`.

Exit 1 when an image is missing or could not be checked. No credentials: a provider pulls anonymously,
so a private image counts as missing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REF_RE = re.compile(
    r"^(?:(?P<registry>[a-zA-Z0-9.-]+(?::\d+)?)/)?"
    r"(?P<repo>[a-z0-9]+(?:[._-]+[a-z0-9]+)*(?:/[a-z0-9]+(?:[._-]+[a-z0-9]+)*)*)"
    r"(?::(?P<tag>[\w][\w.-]{0,127}))?"
    r"(?:@(?P<digest>sha256:[0-9a-f]{64}))?$"
)
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?[-=]([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
YAML_KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?:-\s+)?(?P<key>[A-Za-z0-9_.-]+)\s*:\s*(?P<value>.*?)\s*$"
)
IMAGE_KEY_RE = re.compile(r"^(?:image|[A-Z0-9_]*IMAGE(?:_REF)?)$")
FROM_RE = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(?P<image>\S+)(?:\s+AS\s+(?P<stage>\S+))?", re.I
)
ARG_RE = re.compile(r"^\s*ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:=(?P<default>\S*))?", re.I)
MISSING_MARKERS = (
    "no such manifest",
    "manifest unknown",
    "not found",
    "unauthorized",
    "denied",
    "requested access",
)


@dataclass(frozen=True)
class Pin:
    path: str
    line: int
    ref: str

    def where(self) -> str:
        return f"{self.path}:{self.line}"


def strip_value(raw: str) -> str:
    value = re.sub(r"\s+#.*$", "", raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_content(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def parent_key(lines: list[str], idx: int) -> str | None:
    indent = indent_of(lines[idx])
    for i in range(idx - 1, -1, -1):
        if is_content(lines[i]) and indent_of(lines[i]) < indent:
            m = YAML_KEY_RE.match(lines[i])
            return m.group("key") if m else None
    return None


def resolve(ref: str, args: dict[str, str]) -> str | None:
    """`ref` with $VAR / ${VAR} / ${VAR:-default} substituted, or None when a variable has no value here."""
    if "${{" in ref:
        return None
    unresolved = False

    def sub(m: re.Match[str]) -> str:
        nonlocal unresolved
        name = m.group(1) or m.group(3)
        if name in args:
            return args[name]
        if m.group(2) is not None and m.group(2) != "":
            return m.group(2)
        unresolved = True
        return ""

    out = VAR_RE.sub(sub, ref)
    return None if unresolved else out


def scan_yaml(path: str, text: str) -> tuple[list[tuple[Pin, str | None]], set[str], set[str]]:
    """([(pin, compose service it belongs to)], services with `build:`, additional_contexts names).

    The service is None for a key that is not a compose `image:`. `build:` is matched by service name
    because an override file (docker-compose.build.yaml) can add it to a service another file names."""
    lines = text.splitlines()
    pins: list[tuple[Pin, str | None]] = []
    built: set[str] = set()
    contexts: set[str] = set()
    for idx, line in enumerate(lines):
        m = YAML_KEY_RE.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        key, value = m.group("key"), strip_value(m.group("value"))
        if key == "build" and (service := parent_key(lines, idx)):
            built.add(service)
            continue
        if key == "additional_contexts":
            indent = indent_of(line)
            for nxt in lines[idx + 1 :]:
                if not is_content(nxt):
                    continue
                if indent_of(nxt) <= indent:
                    break
                item = nxt.strip().lstrip("- ").strip()
                contexts.add(re.split(r"[:=]", item, maxsplit=1)[0].strip())
            continue
        if not IMAGE_KEY_RE.match(key) or not value or value in ("|", ">"):
            continue
        pins.append((Pin(path, idx + 1, value), parent_key(lines, idx) if key == "image" else None))
    return pins, built, contexts


def scan_dockerfile(path: str, text: str) -> tuple[list[Pin], list[tuple[Pin, str]]]:
    """(pins, [(pin, reason skipped)]); stage names are skipped here, context names by the caller."""
    args: dict[str, str] = {}
    stages: set[str] = set()
    pins: list[Pin] = []
    skipped: list[tuple[Pin, str]] = []
    seen_from = False
    for idx, line in enumerate(text.splitlines()):
        if (m := ARG_RE.match(line)) and not seen_from and m.group("default") is not None:
            args[m.group("name")] = strip_value(m.group("default"))
            continue
        if not (m := FROM_RE.match(line)):
            continue
        seen_from = True
        raw = m.group("image")
        pin = Pin(path, idx + 1, raw)
        if m.group("stage"):
            stages.add(m.group("stage").lower())
        if raw.lower() in stages - {(m.group("stage") or "").lower()} or raw == "scratch":
            skipped.append((pin, "earlier build stage" if raw != "scratch" else "scratch"))
            continue
        pins.append(pin)
    resolved = []
    for pin in pins:
        ref = resolve(pin.ref, args)
        if ref is None:
            skipped.append((pin, "variable not resolvable here"))
        else:
            resolved.append(Pin(pin.path, pin.line, ref))
    return resolved, skipped


def is_dockerfile(path: str) -> bool:
    name = Path(path).name
    return name.startswith("Dockerfile") or name.endswith((".Dockerfile", ".dockerfile"))


def collect(files: dict[str, str]) -> tuple[dict[str, list[Pin]], list[tuple[Pin, str]]]:
    """({ref: [where it is pinned]}, [(pin, reason skipped)]) over {path: text}."""
    raw: list[Pin] = []
    skipped: list[tuple[Pin, str]] = []
    contexts: set[str] = set()
    services: list[tuple[Pin, str]] = []
    builds: dict[str, set[str]] = {}  # compose file -> services with `build:`
    for path, text in sorted(files.items()):
        if path.endswith((".yml", ".yaml")):
            pins, b, c = scan_yaml(path, text)
            if "compose" in Path(path).name:
                builds[path] = b
            contexts |= c
            for pin, service in pins:
                ref = resolve(pin.ref, {})
                if ref is None:
                    skipped.append((pin, "variable not resolvable here"))
                    continue
                raw.append(Pin(pin.path, pin.line, ref))
                if service:
                    services.append((raw[-1], service))
        elif is_dockerfile(path):
            pins, s = scan_dockerfile(path, text)
            raw += pins
            skipped += s
    # a service built in a file that gives it no image is an override (docker-compose.build.yaml) of the
    # same service in the other compose files of its directory; alternatives (app.dev.yml, app.local.yml)
    # each name their own image
    named = {(pin.path, service) for pin, service in services}
    overrides = {
        (str(Path(path).parent), s)
        for path, b in builds.items()
        for s in b
        if (path, s) not in named
    }
    built = {
        pin.ref
        for pin, service in services
        if service in builds.get(pin.path, set())
        or (str(Path(pin.path).parent), service) in overrides
    }
    refs: dict[str, list[Pin]] = {}
    for pin in raw:
        if pin.ref in built:
            skipped.append((pin, "built by a compose service here"))
        elif is_dockerfile(pin.path) and pin.ref in contexts:
            skipped.append((pin, "compose additional_contexts name"))
        elif not REF_RE.match(pin.ref):
            skipped.append((pin, "not an image reference"))
        else:
            refs.setdefault(pin.ref, []).append(pin)
    return refs, skipped


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def tracked_files(repo: Path, rev: str | None = None) -> dict[str, str]:
    """{path: text} of the YAML files and Dockerfiles in the working tree, or at `rev`."""
    listing = (
        git(repo, "ls-tree", "-r", "-z", "--name-only", rev) if rev else git(repo, "ls-files", "-z")
    )
    files = {}
    for name in listing.decode().split("\0"):
        if name and (name.endswith((".yml", ".yaml")) or is_dockerfile(name)):
            try:
                data = git(repo, "show", f"{rev}:{name}") if rev else (repo / name).read_bytes()
                files[name] = data.decode("utf-8")
            except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
                pass
    return files


def registry_head(ref: str) -> str:
    """'ok' / 'missing' / 'error: ...' from an anonymous HEAD on the manifest (HEAD is not a counted pull)."""
    m = REF_RE.match(ref)
    assert m
    registry, repo = m.group("registry"), m.group("repo")
    if not registry or ("." not in registry and ":" not in registry and registry != "localhost"):
        repo = f"{registry}/{repo}" if registry else repo
        registry = "docker.io"
    if registry == "docker.io":
        registry = "registry-1.docker.io"
        if "/" not in repo:
            repo = f"library/{repo}"
    url = (
        f"https://{registry}/v2/{repo}/manifests/{m.group('digest') or m.group('tag') or 'latest'}"
    )
    accept = ", ".join(
        [
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        ]
    )
    headers = {"Accept": accept}
    error = "missing"
    for _ in range(3):
        try:
            urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD", headers=headers), timeout=20
            )
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            challenge = e.headers.get("WWW-Authenticate", "")
            if e.code == 401 and challenge.startswith("Bearer") and "Authorization" not in headers:
                params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
                query = "&".join(f"{k}={params[k]}" for k in ("service", "scope") if k in params)
                with urllib.request.urlopen(f"{params['realm']}?{query}", timeout=20) as resp:
                    body = json.load(resp)
                headers["Authorization"] = f"Bearer {body.get('token') or body.get('access_token')}"
                continue
            if e.code == 429:
                error = "error: HTTP 429 (registry rate limit)"
                time.sleep(min(int(e.headers.get("Retry-After") or 5), 15))
                continue
            return "missing" if e.code in (401, 403) else f"error: HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            error = f"error: {e}"
            time.sleep(3)
    return error


def docker_inspect(ref: str) -> str:
    """'ok' / 'missing' / 'error: ...' from `docker manifest inspect`.

    A digest pin goes to the registry HEAD instead: docker 27 answers "manifest verification failed"
    for a multi-arch index digest that exists and for one that does not. A rate limit, or an error that
    names neither, falls back to the HEAD too."""
    if "@sha256:" in ref:
        return registry_head(ref)
    for _ in range(2):
        try:
            proc = subprocess.run(
                ["docker", "manifest", "inspect", ref], capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode == 0:
            return "ok"
        output = (proc.stderr or proc.stdout).strip().splitlines()
        err = (output[-1] if output else "").lower()
        if "toomanyrequests" not in err and any(marker in err for marker in MISSING_MARKERS):
            return "missing"
        break
    return registry_head(ref)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument(
        "--base",
        help="check only the references that are not pinned at this revision too (a PR: the merge ref's HEAD^1)",
    )
    ap.add_argument("--list", action="store_true", help="print what would be checked and exit")
    ap.add_argument(
        "--backend",
        choices=("docker", "registry"),
        default="docker",
        help="docker: `docker manifest inspect` (CI); registry: anonymous HEAD, for a machine without docker",
    )
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    refs, skipped = collect(tracked_files(repo))
    if args.base:
        # Docker Hub gives an anonymous IP about 100 manifest requests an hour, and a runner's IP is
        # shared; a PR is checked for the references it adds, main and the schedule for all of them
        known, _ = collect(tracked_files(repo, args.base))
        print(
            f"{len(refs)} reference(s) pinned; checking the {len(set(refs) - set(known))} not pinned at {args.base}"
        )
        refs = {ref: pins for ref, pins in refs.items() if ref not in known}

    for pin, reason in sorted(skipped, key=lambda s: (s[0].path, s[0].line)):
        print(f"skipped  {pin.ref}  ({reason}; {pin.where()})")
    if args.list:
        for ref, pins in sorted(refs.items()):
            print(f"check    {ref}  ({', '.join(p.where() for p in pins)})")
        return 0

    backend = docker_inspect if args.backend == "docker" else registry_head

    def check(ref: str) -> str:
        try:
            return backend(ref)
        except Exception as e:  # a token endpoint that fails mid-probe; reported, never a traceback
            return f"error: {e}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(sorted(refs), pool.map(check, sorted(refs))))

    bad = {ref: r for ref, r in results.items() if r != "ok"}
    for ref, result in results.items():
        print(f"{'ok' if result == 'ok' else result.split(':')[0]:<8} {ref}")
    print(
        f"\n{len(results)} image(s) checked, {len(bad)} missing or unchecked, {len(skipped)} reference(s) skipped"
    )
    for ref, result in sorted(bad.items()):
        where = ", ".join(p.where() for p in refs[ref])
        if result == "missing":
            print(
                f"::error::{ref} is not in its registry (pinned at {where}); push it before this merges"
            )
        else:
            print(
                f"::error::{ref} could not be checked ({result}; pinned at {where}); re-run the job"
            )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
