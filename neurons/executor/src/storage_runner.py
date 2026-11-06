from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

from storage.models import OperationSpecError, StorageOperationSpec
from storage.reporting import ReportingLeaseExpired, StorageEventReporter
from storage.restic import (
    HELPER_LABEL,
    JsonEventWriter,
    ResticOperationError,
    ResticResult,
    ResticStorageRunner,
    StorageOperationCancelled,
)
from storage.workspace import WorkspaceResolutionError, WorkspaceResolver

DOCKER_BINARY = "/usr/bin/docker"
# Set inside the detached runner container: it runs the operation itself instead of launching another one.
DETACHED_ENV = "LIUM_STORAGE_DETACHED"
# The detached runner prints this once it has read its spec from stdin; the spec is never in argv, env or a file.
SPEC_RECEIVED = "storage-runner-spec-received"
SPEC_HANDOFF_TIMEOUT_SECONDS = 60
# A runner container that got no spec in time exits with this (EX_TEMPFAIL); the validator relaunches it.
SPEC_TIMEOUT_EXIT = 75


def _cancel_on_sigterm(signum: int, frame: object) -> None:
    # Raising unwinds through ResticStorageRunner's finally, which removes the helper container.
    raise StorageOperationCancelled("storage runner received SIGTERM")


def main() -> int:
    arguments = _parse_arguments()
    if os.environ.get(DETACHED_ENV) != "1":
        return _run_detached(arguments)
    signal.signal(signal.SIGTERM, _cancel_on_sigterm)
    spec_line = _read_spec_line(SPEC_HANDOFF_TIMEOUT_SECONDS)
    if spec_line is None:
        # the launcher died before the handoff (an executor restart): exit instead of blocking forever
        message = f"storage runner received no spec within {SPEC_HANDOFF_TIMEOUT_SECONDS}s"
        print(json.dumps({"event": "result", "status": "FAILED", "message": message}), flush=True)
        return SPEC_TIMEOUT_EXIT
    event_writer: JsonEventWriter | None = None
    try:
        payload = _parse_object(spec_line)
        print(SPEC_RECEIVED, flush=True)
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
        if operation.action.value == "restore":
            event_writer.stage("PREPARING")
        workspace = WorkspaceResolver().resolve(operation)
        ResticStorageRunner(operation, workspace, event_writer=event_writer).run()
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


def runner_container_name(operation_id: str) -> str:
    return f"lium-storage-runner-{operation_id[:12]}"


def _run_detached(arguments: argparse.Namespace) -> int:
    """Run the operation in its own container, so reporting and restic outlive an executor restart.

    This process only follows that container for the validator's log/status files: if the executor
    container is replaced, this process dies and the runner container carries on and reports to the
    backend. A cancel file is forwarded as SIGTERM, which the runner turns into a CANCELLED result.
    """
    try:
        payload = _load_operation(arguments.spec)
        operation_id = str(payload.get("operation_id") or "")
        if not operation_id:
            raise OperationSpecError("operation spec is missing operation_id")
        image = WorkspaceResolver()._helper_image()
    except (OperationSpecError, WorkspaceResolutionError, OSError) as error:
        print(
            json.dumps({"event": "result", "status": "FAILED", "message": str(error)}), flush=True
        )
        return 1
    name = runner_container_name(operation_id)
    started = subprocess.run(
        _runner_container_command(name, operation_id, image),
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        message = f"storage runner container did not start: {(started.stderr or '').strip()[-500:]}"
        print(json.dumps({"event": "result", "status": "FAILED", "message": message}), flush=True)
        return 1
    if not _send_spec(name, payload):
        subprocess.run([DOCKER_BINARY, "rm", "--force", name], capture_output=True)
        message = (
            f"storage runner container did not read its spec within {SPEC_HANDOFF_TIMEOUT_SECONDS}s"
        )
        print(json.dumps({"event": "result", "status": "FAILED", "message": message}), flush=True)
        return 1

    def cancel_requested() -> bool:
        return bool(arguments.cancel_file) and Path(arguments.cancel_file).exists()

    def on_sigterm(signum: int, frame: object) -> None:
        # A cancel stops the runner; any other SIGTERM (executor shutdown) leaves it running.
        if cancel_requested():
            subprocess.run([DOCKER_BINARY, "kill", "--signal", "TERM", name], capture_output=True)
            return
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, on_sigterm)
    logs = subprocess.Popen([DOCKER_BINARY, "logs", "--follow", name], stderr=subprocess.STDOUT)
    stop_forwarding = threading.Event()
    threading.Thread(
        target=_forward_cancel, args=(name, cancel_requested, stop_forwarding), daemon=True
    ).start()
    waited = subprocess.run([DOCKER_BINARY, "wait", name], capture_output=True, text=True)
    stop_forwarding.set()
    try:
        logs.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logs.kill()
    subprocess.run([DOCKER_BINARY, "rm", "--force", name], capture_output=True)
    try:
        return int(waited.stdout.strip())
    except ValueError:
        return 1


def _read_spec_line(timeout_seconds: float) -> str | None:
    """One line from stdin (a detached container's stdin never closes), or None if none arrives in time."""
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.append(sys.stdin.readline()), daemon=True)
    reader.start()
    reader.join(timeout_seconds)
    return lines[0] if lines else None


def _send_spec(name: str, payload: Mapping[str, object]) -> bool:
    """Write the spec to the runner's stdin; False when the runner neither confirms nor exits in time."""
    attach = subprocess.Popen(
        [DOCKER_BINARY, "attach", "--sig-proxy=false", name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    finished = threading.Event()

    def hand_over() -> None:
        try:
            attach.stdin.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]
            attach.stdin.flush()  # type: ignore[union-attr]
            for line in attach.stdout:  # type: ignore[union-attr]
                if SPEC_RECEIVED in line:
                    break
        except (BrokenPipeError, ValueError, OSError):
            pass
        # a confirmation or an exited runner (EOF; docker wait then reports its status) both end the handoff
        finished.set()

    threading.Thread(target=hand_over, daemon=True).start()
    in_time = finished.wait(SPEC_HANDOFF_TIMEOUT_SECONDS)
    attach.kill()
    attach.wait()
    return in_time


def _forward_cancel(name: str, cancel_requested: object, stop: threading.Event) -> None:
    while not stop.wait(1):
        if cancel_requested():  # type: ignore[operator]
            subprocess.run([DOCKER_BINARY, "kill", "--signal", "TERM", name], capture_output=True)
            return


def _runner_container_command(name: str, operation_id: str, image: str) -> list[str]:
    return [
        DOCKER_BINARY,
        "run",
        "--detach",
        "--name",
        name,
        # the helper label lets a later operation remove this container once it has long exited
        "--label",
        f"{HELPER_LABEL}={operation_id}",
        # the runner inspects rental containers and their /proc entries and drives docker, like the executor
        "--privileged",
        "--pid",
        "host",
        "--volume",
        "/var/run/docker.sock:/var/run/docker.sock",
        "--interactive",
        "--env",
        f"{DETACHED_ENV}=1",
        "--env",
        f"LIUM_STORAGE_HELPER_IMAGE={image}",
        "--entrypoint",
        sys.executable,
        image,
        str(Path(__file__).resolve()),
    ]


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
