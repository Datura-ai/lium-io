from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Mapping

from storage.models import OperationSpecError, StorageOperationSpec
from storage.restic import JsonEventWriter, ResticOperationError, ResticResult, ResticStorageRunner
from storage.workspace import WorkspaceResolutionError, WorkspaceResolver


def main() -> int:
    arguments = _parse_arguments()
    event_writer: JsonEventWriter | None = None
    try:
        payload = _load_operation(arguments.spec)
        operation = StorageOperationSpec.from_mapping(payload)
        event_writer = JsonEventWriter(
            str(operation.operation_id),
            operation.progress_interval_seconds,
            operation.heartbeat_interval_seconds,
        )
        workspace = WorkspaceResolver().resolve(operation)
        ResticStorageRunner(operation, workspace, event_writer=event_writer).run()
        return 0
    except (OperationSpecError, WorkspaceResolutionError, ResticOperationError, OSError) as error:
        if event_writer:
            event_writer.diagnostic(str(error))
            event_writer.result(ResticResult("FAILED", None, None, 1))
        else:
            print(json.dumps({"event": "result", "status": "FAILED", "message": str(error)}))
        return 1


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Lium restic backup or restore operation")
    parser.add_argument(
        "--spec",
        default="-",
        help="operation JSON file with mode 0600, or '-' to read JSON from stdin",
    )
    return parser.parse_args()


def _load_operation(spec_path: str) -> Mapping[str, object]:
    if spec_path == "-":
        return _parse_object(sys.stdin.read())

    path = Path(spec_path)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise OperationSpecError(f"operation spec must not be group/world accessible: {oct(mode)}")
    return _parse_object(path.read_text())


def _parse_object(value: str) -> Mapping[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise OperationSpecError("operation spec is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise OperationSpecError("operation spec must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
