"""The committed JSON-schema snapshot equals the models, and the check notices when it does not."""

import json
import subprocess
import sys
from pathlib import Path

from lium_protocol import PROTOCOL_VERSION
from lium_protocol.schema import build_schema, diff_against_snapshot, render, snapshot_path

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_snapshot_matches_the_models() -> None:
    assert snapshot_path().exists(), "run `python -m lium_protocol.schema --write`"
    assert diff_against_snapshot() == ""


def test_snapshot_names_the_protocol_version_and_every_section() -> None:
    document = json.loads(snapshot_path().read_text())
    assert document["protocol_version"] == PROTOCOL_VERSION
    assert set(document) == {
        "protocol_version",
        "validator_to_backend",
        "backend_to_validator",
        "socket_replies",
        "http",
        "$defs",
    }
    # every directory entry points into the shared $defs
    for section in ("validator_to_backend", "backend_to_validator", "socket_replies", "http"):
        for name, ref in document[section].items():
            target = ref["$ref"].removeprefix("#/$defs/")
            assert target in document["$defs"], (section, name)


def test_a_model_change_is_a_visible_diff(tmp_path: Path) -> None:
    """Negative control: a snapshot with one field missing fails the check with a diff naming it."""
    document = build_schema()
    del document["$defs"]["ContainerDeleted"]["properties"]["pod_id"]
    stale = tmp_path / "lium_protocol.v1.json"
    stale.write_text(render(document))
    diff = diff_against_snapshot(stale)
    assert diff and '+        "pod_id"' in diff


def test_check_command_exits_zero_on_the_committed_snapshot() -> None:
    run = subprocess.run(
        [sys.executable, "-m", "lium_protocol.schema", "--check"],
        cwd=PACKAGE_DIR,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == f"schema ok: lium_protocol.v1.json matches the models (protocol {PROTOCOL_VERSION})"
