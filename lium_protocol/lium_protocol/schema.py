"""The protocol as one JSON document, and the drift check against the committed snapshot.

`build_schema()` renders every registered message, every typeless socket reply and every HTTP body to one
JSON Schema document (pydantic's `models_json_schema`: a shared `$defs` where each model and enum appears
once, and per-section directories mapping a wire `message_type` or body name to its `$ref`) under the
protocol version.
`snapshots/lium_protocol.v<major>.json` is that document, committed; CI runs `--check` and fails when the
models and the file disagree, so a wire change is visible in the PR diff and reviewed as such — the
same mechanism as the visual baselines of lium-platform#199. A change is accepted by re-running
`--write` and committing the file next to the model change.

    python -m lium_protocol.schema --check    # exit 0 when the snapshot matches; exit 1 with a unified diff
    python -m lium_protocol.schema --write    # regenerate the snapshot
    python -m lium_protocol.schema            # print the document
"""

from __future__ import annotations

import argparse
import difflib
import json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lium_protocol.schema",
        description="Render the protocol's JSON Schema; --check compares it with the committed snapshot.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="exit 1 with a diff when the snapshot is stale")
    group.add_argument("--write", action="store_true", help="regenerate the snapshot file")
    args = parser.parse_args(argv)
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
        return 0
    sys.stdout.write(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
