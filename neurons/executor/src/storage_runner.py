from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import time
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from storage import restic
from storage.models import OperationSpecError, StorageOperationSpec
from storage.reporting import ReportingLeaseExpired, StorageEventReporter
from storage.restic import (
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


def _take_helper_spec() -> str:
    path = Path(restic.HELPER_SPEC_PATH)
    deadline = time.monotonic() + restic.HELPER_SPEC_WAIT_SECONDS
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("the storage helper's spec never arrived")
        time.sleep(0.1)
    try:
        return path.read_text()
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    signal.signal(signal.SIGTERM, _cancel_on_sigterm)
    arguments = _parse_arguments()
    event_writer: JsonEventWriter | None = None
    try:
        # in the helper the spec arrives in its tmpfs after the start; removed there once read
        payload = (
            _parse_object(_take_helper_spec())
            if arguments.in_helper
            else _load_operation(arguments.spec)
        )
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
        runner = ResticStorageRunner(operation, workspace, event_writer=event_writer)
        runner.launch_wrapper_helper(sys.executable, str(Path(__file__).resolve()), json.dumps(payload))
        # detaching is opt-in (spec "detach", ignored by older executors): an older validator reads this exit as the end
        return 0 if payload.get("detach") is True else runner.follow_wrapper_helper()
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
    parser.add_argument(
        "--in-helper", action="store_true", help="run the operation (the helper's entrypoint)"
    )
    parser.add_argument("--workspace", help="in the helper: the workspace path")
    parser.add_argument(
        "--namespace-pid", type=int, help="in the helper: the encrypted rental's PID for nsenter"
    )
    return parser.parse_args()


def _load_operation(spec_path: str) -> Mapping[str, object]:
    if spec_path == "-":
        return _parse_object(sys.stdin.read())

    path = Path(spec_path)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise OperationSpecError(f"operation spec must not be group/world accessible: {oct(mode)}")
    text = path.read_text()
    path.unlink(missing_ok=True)
    payload = _parse_object(text)
    # the repository credentials must not outlive the read on the provider's disk; the validator's cancel
    # still reports through the operation's own reporter, so only that is written back (DAH-6463)
    reporter = payload.get("reporter")
    if isinstance(reporter, Mapping):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as kept:
            json.dump({"reporter": reporter}, kept)
    return payload


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
