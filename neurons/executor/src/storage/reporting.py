from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from uuid import UUID

import requests

from storage.models import ReporterResource, ReporterSpec, StorageAction

# One progress PUT per interval at most; a hot loop of PUTs tripped the CloudFront WAF rate rule.
PROGRESS_MIN_INTERVAL_SECONDS = 10.0
RETRY_BACKOFF_CAP_SECONDS = 60.0
# Queued log lines stay well under the WAF's 64 KB body limit, leaving room for the rest of the payload.
PENDING_LOGS_MAX_BYTES = 32_000


class ReportingLeaseExpired(RuntimeError):
    pass


@dataclass
class ProgressState:
    progress: float = 0.0
    total_files: int | None = None
    processed_files: int | None = None
    total_bytes: int | None = None
    processed_bytes: int | None = None
    stage: str | None = None


class StorageEventReporter:
    def __init__(
        self,
        operation_id: UUID,
        action: StorageAction,
        spec: ReporterSpec,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._operation_id = operation_id
        self._spec = spec
        self._session = session or requests.Session()
        self._clock = clock
        self._sleeper = sleeper
        self._last_success_at = self._clock()
        initial_stage = "PREPARING" if action is StorageAction.RESTORE else action.value.upper()
        self._state = ProgressState(stage=initial_stage)
        self._cancel_requested = False
        self._restic_failure_detail: str | None = None
        self._diagnostic_detail: str | None = None
        self._next_progress_at = self._clock()
        self._progress_failures = 0
        self._pending_logs: list[str] = []

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def send(self, event: Mapping[str, object]) -> None:
        event_type = event.get("event")
        if event_type == "stage":
            stage = event.get("stage")
            if isinstance(stage, str) and stage:
                self._state.stage = stage
                self._put(self._progress_payload(status="IN_PROGRESS"), terminal=False)
            return
        if event_type == "restic":
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                return
            self._apply_progress(payload)
            if self._state.stage == "PREPARING" and self._restore_has_started():
                self._state.stage = "RESTORING"
            self._remember_restic_error(payload)
            self._put(self._progress_payload(status="IN_PROGRESS"), terminal=False)
            return
        if event_type == "heartbeat":
            self._put(self._progress_payload(status="IN_PROGRESS"), terminal=False)
            return
        if event_type == "diagnostic":
            message = event.get("message")
            if isinstance(message, str) and message:
                if self._diagnostic_detail is None:
                    self._diagnostic_detail = message[:2000]
                # Kept until a PUT succeeds, so throttled or failed updates do not drop them.
                self._pending_logs = _newest_within_bytes(
                    [*self._pending_logs, message[:2000]][-20:]
                )
                self._put(self._progress_payload(status="IN_PROGRESS"), terminal=False)
            return
        if event_type == "result":
            status = event.get("status")
            if not isinstance(status, str):
                return
            result_quality = _optional_string(event.get("result_quality"))
            if status == "COMPLETED":
                self._state.progress = 100.0
                if result_quality == "FULL":
                    self._state.processed_files = self._state.total_files
                    self._state.processed_bytes = self._state.total_bytes
            self._put(
                self._progress_payload(
                    status=status,
                    result_quality=result_quality,
                    snapshot_id=_optional_string(event.get("snapshot_id")),
                    exit_code=_optional_integer(event.get("exit_code")),
                    error_code=_optional_string(event.get("error_code")),
                    error_message=(self._restic_failure_detail or self._diagnostic_detail)
                    if status != "COMPLETED"
                    else None,
                ),
                terminal=True,
            )

    def _apply_progress(self, event: Mapping[str, object]) -> None:
        percent_done = event.get("percent_done")
        if isinstance(percent_done, int | float) and not isinstance(percent_done, bool):
            self._state.progress = max(self._state.progress, min(float(percent_done) * 100.0, 99.9))
        self._state.total_files = _updated_integer(
            event,
            self._state.total_files,
            "total_files",
            "total_file_count",
            "total_files_processed",
        )
        self._state.processed_files = _updated_integer(
            event,
            self._state.processed_files,
            "files_done",
            "files_restored",
            "total_files_processed",
        )
        self._state.total_bytes = _updated_integer(
            event,
            self._state.total_bytes,
            "total_bytes",
            "total_size",
            "total_bytes_processed",
        )
        self._state.processed_bytes = _updated_integer(
            event,
            self._state.processed_bytes,
            "bytes_done",
            "bytes_restored",
            "total_bytes_processed",
        )

    def _remember_restic_error(self, event: Mapping[str, object]) -> None:
        if self._restic_failure_detail is not None:
            return
        message_type = event.get("message_type")
        if message_type not in ("error", "exit_error"):
            return
        error = event.get("error")
        if isinstance(error, Mapping):
            message = _optional_string(error.get("message"))
            if message:
                self._restic_failure_detail = message[:2000]
                return
        message = _optional_string(event.get("message"))
        if message:
            self._restic_failure_detail = message[:2000]

    def _restore_has_started(self) -> bool:
        return bool(
            self._state.progress > 0
            or (self._state.processed_files or 0) > 0
            or (self._state.processed_bytes or 0) > 0
        )

    def _progress_payload(
        self,
        *,
        status: str,
        result_quality: str | None = None,
        snapshot_id: str | None = None,
        exit_code: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]:
        return {
            "operation_id": str(self._operation_id),
            "progress": self._state.progress,
            "status": status,
            "stage": self._state.stage,
            "total_files": self._state.total_files,
            "processed_files": self._state.processed_files,
            "total_bytes": self._state.total_bytes,
            "processed_bytes": self._state.processed_bytes,
            "result_quality": result_quality,
            "snapshot_id": snapshot_id,
            "exit_code": exit_code,
            "error_code": error_code,
            "error_message": error_message,
            "logs": list(self._pending_logs) or None,
        }

    def _put(self, payload: Mapping[str, object], *, terminal: bool) -> None:
        # Progress state is cumulative, so a skipped update is carried by the next one.
        if not terminal and self._clock() < self._next_progress_at:
            return
        retry_delay_seconds = 1.0
        while True:
            try:
                response = self._session.put(
                    self._url(),
                    json=dict(payload),
                    headers={"Authorization": f"Bearer {self._spec.auth_token}"},
                    timeout=10,
                )
                if getattr(response, "status_code", None) in (404, 410):
                    raise ReportingLeaseExpired("storage operation is no longer active")
                response.raise_for_status()
                body = response.json()
                self._last_success_at = self._clock()
                self._next_progress_at = self._last_success_at + PROGRESS_MIN_INTERVAL_SECONDS
                self._progress_failures = 0
                self._pending_logs = []
                if isinstance(body, Mapping) and body.get("cancel_requested") is True:
                    self._cancel_requested = True
                return
            except (requests.RequestException, ValueError) as error:
                # This lease expires before the backend's stale-operation sweep so
                # the runner stops owning the repository before it can be released.
                remaining_seconds = self._spec.failure_timeout_seconds - (
                    self._clock() - self._last_success_at
                )
                if remaining_seconds <= 0:
                    raise ReportingLeaseExpired(
                        "storage operation stopped because progress reporting was unavailable"
                    ) from error
                if not terminal:
                    self._progress_failures += 1
                    backoff_seconds = min(
                        PROGRESS_MIN_INTERVAL_SECONDS * 2 ** min(self._progress_failures - 1, 3),
                        RETRY_BACKOFF_CAP_SECONDS,
                    )
                    self._next_progress_at = self._clock() + backoff_seconds * random.uniform(
                        0.5, 1.0
                    )
                    return
                self._sleeper(
                    min(retry_delay_seconds * random.uniform(0.5, 1.0), remaining_seconds)
                )
                retry_delay_seconds = min(retry_delay_seconds * 2, RETRY_BACKOFF_CAP_SECONDS)

    def renew_credentials(self) -> dict[str, str] | None:
        """DAH-4945: fresh scoped credentials for this backup's repository, or None if refused"""
        if self._spec.resource is not ReporterResource.BACKUP:
            return None
        for attempt in range(3):
            try:
                response = self._session.post(
                    self._url("credentials"),
                    headers={"Authorization": f"Bearer {self._spec.auth_token}"},
                    timeout=30,
                )
                if getattr(response, "status_code", None) in (401, 404, 409, 410):
                    return None
                response.raise_for_status()
                body = response.json()
                keys = ("access_key_id", "secret_access_key", "session_token")
                if isinstance(body, Mapping) and all(
                    _optional_string(body.get(key)) for key in keys
                ):
                    return {key: body[key] for key in keys}
                return None
            except (requests.RequestException, ValueError):
                if attempt < 2:
                    self._sleeper(5.0 * (attempt + 1))
        return None

    def _url(self, endpoint: str = "progress") -> str:
        resource_path = (
            "backup-logs" if self._spec.resource is ReporterResource.BACKUP else "restore-logs"
        )
        return f"{self._spec.api_url}/{resource_path}/{self._operation_id}/{endpoint}"


def _first_integer(value: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            return item
    return None


def _updated_integer(value: Mapping[str, object], current: int | None, *keys: str) -> int | None:
    updated = _first_integer(value, *keys)
    return current if updated is None else updated


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _newest_within_bytes(lines: list[str]) -> list[str]:
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        size += len(json.dumps(line))
        if size > PENDING_LOGS_MAX_BYTES:
            break
        kept.append(line)
    return kept[::-1]
