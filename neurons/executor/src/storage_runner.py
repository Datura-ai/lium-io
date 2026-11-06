from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from storage.models import OperationSpecError, StorageOperationSpec
from storage.reporting import ReportingLeaseExpired, StorageEventReporter
from storage.restic import (
    SPEC_ENV,
    JsonEventWriter,
    ResticOperationError,
    ResticResult,
    ResticStorageRunner,
    StorageOperationCancelled,
)
from storage.workspace import (
    LocalWorkspace,
    NamespaceWorkspace,
    WorkspaceResolutionError,
    WorkspaceResolver,
)


def _cancel_on_sigterm(signum: int, frame: object) -> None:
    # Raising unwinds through ResticStorageRunner, which stops restic; main reports CANCELLED.
    raise StorageOperationCancelled("storage runner received SIGTERM")


def main() -> int:
    signal.signal(signal.SIGTERM, _cancel_on_sigterm)
    arguments = _parse_arguments()
    event_writer: JsonEventWriter | None = None
    try:
        # in the helper the spec comes from its environment; drop it there once read
        payload = _parse_object(os.environ.pop(SPEC_ENV, "")) if arguments.in_helper else _load_operation(arguments.spec)
        operation = StorageOperationSpec.from_mapping(payload)
        reporter = (
            StorageEventReporter(operation.operation_id, operation.action, operation.reporter)
            if operation.reporter
            else None
        )
        event_writer = JsonEventWriter(
            str(operation.operation_id),
            operation.progress_interval_seconds,
            operation.heartbeat_interval_seconds,
            reporter=reporter,
            cancellation_probe=(
                lambda: Path(arguments.cancel_file).exists() if arguments.cancel_file else False
            ),
        )
        if arguments.in_helper:
            # the helper's entrypoint: restic as a child, retries and reporting, all outside the executor container
            workspace = (
                NamespaceWorkspace(arguments.namespace_pid, PurePosixPath(arguments.workspace))
                if arguments.namespace_pid
                else LocalWorkspace(Path(arguments.workspace))
            )
            ResticStorageRunner(operation, workspace, event_writer=event_writer).run()
            return 0
        if operation.action.value == "restore":
            event_writer.stage("PREPARING")
        # the executor only resolves the workspace and starts the helper; an executor restart leaves the run alone
        workspace = WorkspaceResolver().resolve(operation)
        ResticStorageRunner(operation, workspace, event_writer=event_writer).launch_wrapper_helper(
            sys.executable, str(Path(__file__).resolve()), json.dumps(payload)
        )
        return 0
    except StorageOperationCancelled as error:
        if event_writer:
            event_writer.diagnostic(str(error))
            event_writer.result(ResticResult("CANCELLED", None, None, 130))
        return 130
    except ReportingLeaseExpired as error:
        print(
            json.dumps(
                {
                    "event": "result",
                    "status": "FAILED",
                    "message": str(error),
                }
            )
        )
        return 1
    except ResticOperationError as error:
        if event_writer:
            event_writer.diagnostic(str(error))
            event_writer.result(ResticResult("FAILED", None, None, 1, error_code=error.error_code))
        else:
            print(json.dumps({"event": "result", "status": "FAILED", "message": str(error)}))
        return 1
    except (OperationSpecError, WorkspaceResolutionError, OSError) as error:
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
    parser.add_argument(
        "--cancel-file",
        help="local marker created by the validator when cancellation is requested",
    )
    parser.add_argument("--in-helper", action="store_true", help="run the operation (the helper's entrypoint)")
    parser.add_argument("--workspace", help="in the helper: the workspace path")
    parser.add_argument("--namespace-pid", type=int, help="in the helper: the encrypted rental's PID for nsenter")
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
