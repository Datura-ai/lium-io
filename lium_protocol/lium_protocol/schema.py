"""The protocol as one JSON document, and the drift check against the committed snapshot.

`build_schema()` renders every registered message, every typeless socket reply and every HTTP body to one
JSON Schema document (pydantic's `models_json_schema`: a shared `$defs` where each model and enum appears
once, and per-section directories mapping a wire `message_type` or body name to its `$ref`) under the
protocol version.
`snapshots/lium_protocol.v<major>.json` is that document, committed — rendered with pydantic 2.13, which CI
pins: a pydantic release can change the rendering (2.10 omits `additionalProperties`) without the wire
changing. CI runs `--check` and fails when the models and the file disagree, so a wire change is visible in the PR diff and reviewed as such — the
same mechanism as the visual baselines of lium-platform#199. A change is accepted by re-running
`--write` and committing the file next to the model change.

    python -m lium_protocol.schema --check    # exit 0 when the snapshot matches; exit 1 with a unified diff
    python -m lium_protocol.schema --write    # regenerate the snapshot
    python -m lium_protocol.schema --check-base <base snapshot>   # a changed schema has a version above the base's
In CI on a PR or a merge-queue group (GITHUB_EVENT_PATH) `--check` also runs the `--check-base` comparison against
the base branch's committed snapshot.
    python -m lium_protocol.schema            # print the document
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

from . import PROTOCOL_VERSION
from .backend_to_validator import BACKEND_MESSAGES, SOCKET_REPLIES
from .http import HTTP_MODELS
from .validator_to_backend import VALIDATOR_MESSAGES

SCHEMA_DIR = Path(__file__).parent / "snapshots"


def snapshot_path(version: str = PROTOCOL_VERSION) -> Path:
    return SCHEMA_DIR / f"lium_protocol.v{version.split('.')[0]}.json"


def build_schema() -> dict[str, Any]:
    """One document: the four directories map a wire `message_type` (or a reply / body name) to a `$ref`
    into the shared `$defs`, where every model and enum appears once."""
    sections = {
        "validator_to_backend": VALIDATOR_MESSAGES.models(),
        "backend_to_validator": BACKEND_MESSAGES.models(),
        "socket_replies": SOCKET_REPLIES,
        "http": HTTP_MODELS,
    }
    entries = [(model, "validation") for models in sections.values() for model in models.values()]
    refs, document = models_json_schema(entries, ref_template="#/$defs/{model}", title="lium_protocol")
    return {
        "protocol_version": PROTOCOL_VERSION,
        **{
            section: {name: refs[(model, "validation")] for name, model in models.items()}
            for section, models in sections.items()
        },
        "$defs": document["$defs"],
    }


def render(schema: dict[str, Any] | None = None) -> str:
    return json.dumps(schema if schema is not None else build_schema(), indent=2, sort_keys=True) + "\n"


def diff_against_snapshot(path: Path | None = None) -> str:
    """Empty when the snapshot equals the rendered models; else a unified diff (snapshot → models)."""
    path = path or snapshot_path()
    current = render()
    committed = path.read_text() if path.exists() else ""
    if committed == current:
        return ""
    return "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=str(path.relative_to(SCHEMA_DIR.parent.parent))
            if path.is_relative_to(SCHEMA_DIR.parent.parent)
            else str(path),
            tofile="models",
        )
    )


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def base_problem(base_snapshot: str) -> str | None:
    """None when this tree's protocol may follow the base branch's committed snapshot; else why not.

    Several PRs can change the wire at once, and each merges onto whatever the base branch holds then: a changed
    schema must carry a version above the base's, and the version never goes down, so one version always names
    one schema and the version history only climbs (merge order does not matter; the later PR rebases and
    takes the next number)."""
    try:
        base = json.loads(base_snapshot)
        base_version = str(base["protocol_version"])
        before = _version(base_version)
    except (ValueError, KeyError, TypeError) as exc:
        return f"the base snapshot cannot be read ({type(exc).__name__}: {exc})"
    now = _version(PROTOCOL_VERSION)
    if now < before:
        return f"protocol {PROTOCOL_VERSION} is below the base branch's {base_version}: take the next version above it"
    current = build_schema()
    changed = {k: v for k, v in base.items() if k != "protocol_version"} != {
        k: v for k, v in json.loads(render(current)).items() if k != "protocol_version"
    }
    if changed and now == before:
        return (
            f"the schema changed but protocol {PROTOCOL_VERSION} equals the base branch's: one version would name "
            "two schemas; take the next version above it"
        )
    return None


def ci_base_snapshot() -> tuple[str, str] | None:
    """(base branch, its committed snapshot) for a CI run on a PR, its approving review or a merge-queue group
    (GITHUB_EVENT_PATH names the base); None outside such a run. Read with git from the checkout's origin."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    event = json.loads(Path(event_path).read_text())
    ref = (event.get("pull_request") or {}).get("base", {}).get("ref") or (event.get("merge_group") or {}).get(
        "base_ref"
    )
    if not ref:
        return None  # a push to the branch itself: no base to follow
    base = ref.removeprefix("refs/heads/")
    subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", base], check=True, capture_output=True)
    rel = f"lium_protocol/lium_protocol/snapshots/{snapshot_path().name}"  # in the repo, wherever this is installed
    shown = subprocess.run(["git", "show", f"FETCH_HEAD:{rel}"], capture_output=True, text=True)
    if shown.returncode != 0:
        return base, ""  # no snapshot of this major on the base yet: nothing to follow
    return base, shown.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lium_protocol.schema",
        description="Render the protocol's JSON Schema; --check compares it with the committed snapshot.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="exit 1 with a diff when the snapshot is stale")
    group.add_argument("--write", action="store_true", help="regenerate the snapshot file")
    group.add_argument(
        "--check-base",
        metavar="SNAPSHOT",
        help="exit 1 when this protocol cannot follow the base branch's committed snapshot file (CI on a PR)",
    )
    args = parser.parse_args(argv)
    if args.check_base:
        problem = base_problem(Path(args.check_base).read_text())
        if problem:
            print(f"protocol version: {problem}")
            return 1
        print(f"protocol version ok: {PROTOCOL_VERSION} follows the base branch")
        return 0
    path = snapshot_path()
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render())
        print(f"wrote {path}")
        return 0
    if args.check:
        diff = diff_against_snapshot(path)
        if diff:
            sys.stdout.write(diff)
            print(
                f"\nschema drift: the models no longer match {path.name}. Review the change, then "
                "`python -m lium_protocol.schema --write` and commit the snapshot."
            )
            return 1
        print(f"schema ok: {path.name} matches the models (protocol {PROTOCOL_VERSION})")
        # in CI on a PR or a merge-queue group: a changed schema also takes a version above the base branch's
        found = ci_base_snapshot()
        if found and found[1]:
            problem = base_problem(found[1])
            if problem:
                print(f"protocol version against {found[0]}: {problem}")
                return 1
            print(f"protocol version ok: {PROTOCOL_VERSION} follows {found[0]}")
        return 0
    sys.stdout.write(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
