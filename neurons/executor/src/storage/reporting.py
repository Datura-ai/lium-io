from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

import requests

from storage.models import ReporterResource, ReporterSpec, StorageAction


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
    ) -> None:
        self._operation_id = operation_id
        self._spec = spec
        self._session = session or requests.Session()
        self._state = ProgressState(stage=action.value.upper())
        self._cancel_requested = False
        self._failure_detail: str | None = None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def send(self, event: Mapping[str, object]) -> None:
        event_type = event.get("event")
        if event_type == "restic":
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                return
            self._apply_progress(payload)
            self._remember_restic_error(payload)
            self._put(self._progress_payload(status="IN_PROGRESS"), terminal=False)
            return
        if event_type == "heartbeat":
            self._put(self._progress_payload(status="IN_PROGRESS"), terminal=False)
            return
        if event_type == "diagnostic":
            message = event.get("message")
            if isinstance(message, str) and message:
                if self._failure_detail is None:
                    self._failure_detail = message[:2000]
                self._put(
                    self._progress_payload(status="IN_PROGRESS", logs=[message[:2000]]),
                    terminal=False,
                )
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
                    error_message=self._failure_detail if status != "COMPLETED" else None,
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
        if self._failure_detail is not None:
            return
        message_type = event.get("message_type")
        if message_type not in ("error", "exit_error"):
            return
        error = event.get("error")
        if isinstance(error, Mapping):
            message = _optional_string(error.get("message"))
            if message:
                self._failure_detail = message[:2000]
                return
        message = _optional_string(event.get("message"))
        if message:
            self._failure_detail = message[:2000]

    def _progress_payload(
        self,
        *,
        status: str,
        logs: list[str] | None = None,
        result_quality: str | None = None,
        snapshot_id: str | None = None,
        exit_code: int | None = None,
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
            "error_message": error_message,
            "logs": logs,
        }

    def _put(self, payload: Mapping[str, object], *, terminal: bool) -> None:
        attempts = 3 if terminal else 1
        for attempt in range(attempts):
            try:
                response = self._session.put(
                    self._url(),
                    json=dict(payload),
                    headers={"Authorization": f"Bearer {self._spec.auth_token}"},
                    timeout=10,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body, Mapping) and body.get("cancel_requested") is True:
                    self._cancel_requested = True
                return
            except (requests.RequestException, ValueError):
                if attempt + 1 < attempts:
                    time.sleep(1 + attempt)

    def _url(self) -> str:
        resource_path = "backup-logs" if self._spec.resource is ReporterResource.BACKUP else "restore-logs"
        return f"{self._spec.api_url}/{resource_path}/{self._operation_id}/progress"


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
