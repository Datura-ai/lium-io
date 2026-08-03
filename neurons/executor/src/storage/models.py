from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from uuid import UUID


class StorageAction(StrEnum):
    BACKUP = "backup"
    RESTORE = "restore"


class StorageEngine(StrEnum):
    RESTIC = "restic"
    TAR_AWS_CLI = "tar_aws_cli"


class WorkspaceMode(StrEnum):
    PLAIN_VOLUME = "plain_volume"
    ENCRYPTED_RUNNING = "encrypted_running"
    ENCRYPTED_BOOTSTRAP = "encrypted_bootstrap"


class ReporterResource(StrEnum):
    BACKUP = "backup"
    RESTORE = "restore"


class OperationResultQuality(StrEnum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"


class OperationSpecError(ValueError):
    pass


@dataclass(frozen=True)
class RepositorySpec:
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    password: str | None = field(default=None, repr=False)
    session_token: str | None = field(default=None, repr=False)
    region: str = "us-east-1"
    endpoint: str = "s3.amazonaws.com"
    s3_connections: int = 64

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RepositorySpec:
        return cls(
            bucket=_required_string(value, "bucket"),
            access_key_id=_required_string(value, "access_key_id"),
            secret_access_key=_required_string(value, "secret_access_key"),
            password=_optional_nullable_string(value, "password"),
            session_token=_optional_nullable_string(value, "session_token"),
            region=_optional_string(value, "region", "us-east-1"),
            endpoint=_optional_string(value, "endpoint", "s3.amazonaws.com"),
            s3_connections=_optional_bounded_integer(
                value,
                "s3_connections",
                default=64,
                minimum=1,
                maximum=128,
            ),
        )

    def url_for_pod(self, pod_id: UUID) -> str:
        endpoint = self.endpoint.rstrip("/")
        prefix = f"restic/v1/pods/{pod_id}"
        return f"s3:{endpoint}/{self.bucket}/{prefix}"


@dataclass(frozen=True)
class WorkspaceSpec:
    mode: WorkspaceMode
    volume_name: str
    volume_path: PurePosixPath
    requested_path: PurePosixPath
    container_name: str | None = None
    volume_passphrase: str | None = field(default=None, repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> WorkspaceSpec:
        try:
            mode = WorkspaceMode(_required_string(value, "mode"))
        except ValueError as error:
            supported = ", ".join(item.value for item in WorkspaceMode)
            raise OperationSpecError(f"workspace.mode must be one of: {supported}") from error

        container_name = _optional_nullable_string(value, "container_name")
        volume_passphrase = _optional_nullable_string(value, "volume_passphrase")
        if mode is WorkspaceMode.ENCRYPTED_RUNNING and not container_name:
            raise OperationSpecError("workspace.container_name is required for encrypted_running")
        if mode is WorkspaceMode.ENCRYPTED_BOOTSTRAP and not volume_passphrase:
            raise OperationSpecError("workspace.volume_passphrase is required for encrypted_bootstrap")

        return cls(
            mode=mode,
            volume_name=_safe_identifier(_required_string(value, "volume_name"), "workspace.volume_name"),
            volume_path=_absolute_path(_required_string(value, "volume_path"), "workspace.volume_path"),
            requested_path=_absolute_path(_required_string(value, "requested_path"), "workspace.requested_path"),
            container_name=_safe_identifier(container_name, "workspace.container_name") if container_name else None,
            volume_passphrase=volume_passphrase,
        )


@dataclass(frozen=True)
class ReporterSpec:
    api_url: str
    auth_token: str = field(repr=False)
    resource: ReporterResource

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ReporterSpec:
        try:
            resource = ReporterResource(_required_string(value, "resource"))
        except ValueError as error:
            raise OperationSpecError("reporter.resource must be backup or restore") from error
        api_url = _required_string(value, "api_url").rstrip("/")
        if not api_url.startswith(("http://", "https://")):
            raise OperationSpecError("reporter.api_url must be an HTTP(S) URL")
        return cls(
            api_url=api_url,
            auth_token=_required_string(value, "auth_token"),
            resource=resource,
        )


@dataclass(frozen=True)
class StorageOperationSpec:
    operation_id: UUID
    pod_id: UUID
    repository_pod_id: UUID
    action: StorageAction
    engine: StorageEngine
    repository: RepositorySpec
    workspace: WorkspaceSpec
    snapshot_id: str | None = None
    legacy_object_key: str | None = None
    legacy_object_size_bytes: int | None = None
    reporter: ReporterSpec | None = None
    progress_interval_seconds: float = 10.0
    heartbeat_interval_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StorageOperationSpec:
        try:
            operation_id = UUID(_required_string(value, "operation_id"))
            pod_id = UUID(_required_string(value, "pod_id"))
            repository_pod_id_value = _optional_nullable_string(value, "repository_pod_id")
            repository_pod_id = UUID(repository_pod_id_value) if repository_pod_id_value else pod_id
        except ValueError as error:
            raise OperationSpecError("operation_id, pod_id, and repository_pod_id must be UUIDs") from error

        try:
            action = StorageAction(_required_string(value, "action"))
        except ValueError as error:
            raise OperationSpecError("action must be backup or restore") from error

        try:
            engine = StorageEngine(_optional_string(value, "engine", StorageEngine.RESTIC.value))
        except ValueError as error:
            raise OperationSpecError("engine must be restic or tar_aws_cli") from error

        repository_value = _required_mapping(value, "repository")
        workspace_value = _required_mapping(value, "workspace")
        repository = RepositorySpec.from_mapping(repository_value)
        workspace = WorkspaceSpec.from_mapping(workspace_value)
        snapshot_id = _optional_nullable_string(value, "snapshot_id")
        legacy_object_key = _optional_nullable_string(value, "legacy_object_key")
        legacy_object_size_bytes = _optional_nullable_nonnegative_integer(value, "legacy_object_size_bytes")
        if engine is StorageEngine.RESTIC and not repository.password:
            raise OperationSpecError("repository.password is required for restic")
        if action is StorageAction.RESTORE and engine is StorageEngine.RESTIC and not snapshot_id:
            raise OperationSpecError("snapshot_id is required for restic restore")
        if engine is StorageEngine.TAR_AWS_CLI:
            if action is not StorageAction.RESTORE:
                raise OperationSpecError("tar_aws_cli is supported only for legacy restore")
            if not legacy_object_key:
                raise OperationSpecError("legacy_object_key is required for tar_aws_cli restore")
        if workspace.mode is WorkspaceMode.ENCRYPTED_BOOTSTRAP and action is not StorageAction.RESTORE:
            raise OperationSpecError("encrypted_bootstrap is supported only for restore")
        if snapshot_id and not re.fullmatch(r"[0-9a-f]{8,64}", snapshot_id):
            raise OperationSpecError("snapshot_id must be a hexadecimal restic snapshot ID")

        interval_value = value.get("progress_interval_seconds", 10.0)
        if not isinstance(interval_value, int | float) or isinstance(interval_value, bool):
            raise OperationSpecError("progress_interval_seconds must be a number")
        if interval_value < 0:
            raise OperationSpecError("progress_interval_seconds must not be negative")

        heartbeat_value = value.get("heartbeat_interval_seconds", 30.0)
        if not isinstance(heartbeat_value, int | float) or isinstance(heartbeat_value, bool):
            raise OperationSpecError("heartbeat_interval_seconds must be a number")
        if heartbeat_value <= 0:
            raise OperationSpecError("heartbeat_interval_seconds must be positive")

        return cls(
            operation_id=operation_id,
            pod_id=pod_id,
            repository_pod_id=repository_pod_id,
            action=action,
            engine=engine,
            repository=repository,
            workspace=workspace,
            snapshot_id=snapshot_id,
            legacy_object_key=legacy_object_key,
            legacy_object_size_bytes=legacy_object_size_bytes,
            reporter=ReporterSpec.from_mapping(_required_mapping(value, "reporter"))
            if value.get("reporter") is not None
            else None,
            progress_interval_seconds=float(interval_value),
            heartbeat_interval_seconds=float(heartbeat_value),
        )


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise OperationSpecError(f"{key} must be an object")
    return item


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise OperationSpecError(f"{key} must be a non-empty string")
    return item.strip()


def _optional_string(value: Mapping[str, object], key: str, default: str) -> str:
    item = value.get(key, default)
    if not isinstance(item, str) or not item.strip():
        raise OperationSpecError(f"{key} must be a non-empty string")
    return item.strip()


def _optional_nullable_string(value: Mapping[str, object], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise OperationSpecError(f"{key} must be a non-empty string when provided")
    return item.strip()


def _optional_bounded_integer(
    value: Mapping[str, object],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    item = value.get(key, default)
    if not isinstance(item, int) or isinstance(item, bool):
        raise OperationSpecError(f"{key} must be an integer")
    if item < minimum or item > maximum:
        raise OperationSpecError(f"{key} must be between {minimum} and {maximum}")
    return item


def _optional_nullable_nonnegative_integer(value: Mapping[str, object], key: str) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise OperationSpecError(f"{key} must be a non-negative integer when provided")
    return item


def _safe_identifier(value: str, key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise OperationSpecError(f"{key} contains unsupported characters")
    return value


def _absolute_path(value: str, key: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise OperationSpecError(f"{key} must be an absolute path without '..'")
    return path
