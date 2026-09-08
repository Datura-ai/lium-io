"""Messages the backend sends down the validator's WebSocket: container lifecycle, ssh keys, backups,
Jupyter, and the answers to the validator's own requests.

Wire values are `payload_models.ContainerRequestType` on the validator and `ContainerRequestType` in the
backend's `src/protocol/compute_app_requests.py`; this enum is their union. Fields follow the same rule as
validator_to_backend.py: a receiver must accept what either peer sends today, so a field one side makes
required and the other optional is optional here.
"""

from __future__ import annotations

import enum
from typing import Any, Literal

import pydantic

from .base import Message, Registry, WorkloadKind

# storage operations report their own failure when the peer stays silent this long (both peers default it)
STORAGE_OPERATION_REPORTING_FAILURE_TIMEOUT_SECONDS = 600


class BackendMessageType(enum.Enum):
    ContainerCreateRequest = "ContainerCreateRequest"
    ContainerStartRequest = "ContainerStartRequest"
    ContainerStopRequest = "ContainerStopRequest"
    ContainerDeleteRequest = "ContainerDeleteRequest"
    AddSshPublicKey = "AddSshPublicKey"
    RemoveSshPublicKeysRequest = "RemoveSshPublicKeysRequest"
    DuplicateExecutorsResponse = "DuplicateExecutorsResponse"
    ExecutorRentFinished = "ExecutorRentFinished"
    GetPodLogsRequestFromServer = "GetPodLogsRequestFromServer"
    AddDebugSshKeyRequest = "AddDebugSshKeyRequest"
    BackupContainerRequest = "BackupContainerRequest"
    RestoreContainerRequest = "RestoreContainerRequest"
    CancelStorageOperationRequest = "CancelStorageOperationRequest"
    InstallJupyterServer = "InstallJupyterServer"
    # staging only: start the validation cycle now; both peers spell it as a Literal today
    ForcedValidationCycleRequest = "ForcedValidationCycleRequest"
    # ask the validator for an exact rental estimate; it answers with EstimateResponse
    GetEstimateRequest = "GetEstimateRequest"


BACKEND_MESSAGES: Registry[BackendMessage] = Registry(BackendMessageType)


class BackendMessage(Message):
    message_type: BackendMessageType


class ServerRequest(BackendMessage):
    """A request about one executor of one miner."""

    miner_hotkey: str
    executor_id: str
    # the validator's copy carries where the miner listens; the backend does not send them
    miner_address: str | None = None
    miner_port: int | None = None


class ContainerRequest(ServerRequest):
    pod_id: str
    workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL


class CustomOptions(pydantic.BaseModel):
    volumes: list[str] | None = None
    environment: dict[str, str] | None = None
    entrypoint: str | None = None
    internal_ports: list[int] | None = None
    startup_commands: str | None = None
    shm_size: str | None = None
    initial_port_count: int | None = None


class ExternalVolumeInfo(pydantic.BaseModel):
    name: str
    plugin: str
    iam_user_access_key: str = pydantic.Field(repr=False)
    iam_user_secret_key: str = pydantic.Field(repr=False)
    session_token: str | None = pydantic.Field(default=None, repr=False)


class BootstrapRestoreSpec(pydantic.BaseModel):
    restore_log_id: str
    backup_engine: str
    repository_pod_id: str
    repository_password: str | None = pydantic.Field(default=None, repr=False)
    backup_volume_info: ExternalVolumeInfo
    snapshot_id: str | None = None
    legacy_object_key: str | None = None
    legacy_object_size_bytes: int | None = None
    auth_token: str = pydantic.Field(repr=False)
    restore_path: str
    failure_timeout_seconds: int = pydantic.Field(default=STORAGE_OPERATION_REPORTING_FAILURE_TIMEOUT_SECONDS, gt=0)


class PayloadPortMapping(pydantic.BaseModel):
    docker_port: int | None = None
    internal_port: int
    external_port: int


class GpuPowerLimit(pydantic.BaseModel):
    """DAH-2356: one GPU's power cap for the Lium PEARL default job; the validator rejects a non-positive
    value at the boundary."""

    gpu_uuid: str
    watts: int = pydantic.Field(gt=0)


class CacheVolume(pydantic.BaseModel):
    """DAH-2475: a persistent named docker volume mounted into a FILLER container (`name` must not use the
    ephemeral `volume_` prefix)."""

    name: str
    target: str


class ClusterMembership(pydantic.BaseModel):
    """DAH-2620: this pod is one node of a multi-node group rental; the backend mints the WireGuard mesh."""

    node_index: int
    wireguard_conf: str = pydantic.Field(repr=False)
    # DAH-2664: the login the group's pods share; empty from an older backend
    ssh_private_key: str = pydantic.Field(default="", repr=False)
    ssh_authorized_key: str = ""


@BACKEND_MESSAGES.register
class ContainerCreateRequest(ContainerRequest):
    """Create (or edit) a pod's container. Disk sizing: with `disk_share` set the validator sizes from
    fresh on-host disk state and the two `*_limit_gb` are caps; without it they are exact sizes."""

    message_type: BackendMessageType = BackendMessageType.ContainerCreateRequest
    docker_image: str
    user_public_keys: list[str] = []
    gpu_uuids: list[str]
    cpu_count: int | None = None
    memory_gb: int | None = None
    custom_options: CustomOptions | None = None
    debug: bool | None = None
    local_volume: str | None = None
    volume_limit_gb: int | None = None
    storage_limit_gb: int | None = None
    disk_share: float | None = None
    min_volume_gb: int | None = None
    external_volume_info: ExternalVolumeInfo | None = None
    is_sysbox: bool | None = None
    docker_username: str | None = None
    docker_password: str | None = pydantic.Field(default=None, repr=False)
    timestamp: int | None = None
    # DAH-2458: backend-measured pre-dispatch spans ({name, duration}), seeding the validator's profile
    pre_dispatch_profilers: list[dict[str, Any]] = []
    backup_log_id: str | None = None
    restore_path: str | None = None
    bootstrap_restore: BootstrapRestoreSpec | None = None
    enable_jupyter: bool | None = None
    enable_volume_encryption: bool | None = None
    available_ports: list[PayloadPortMapping] | None = None
    pod_mapping: list[PayloadPortMapping] | None = None
    active_container_names: list[str] | None = None
    active_volume_names: list[str] | None = None
    cluster_membership: ClusterMembership | None = None
    # DAH-2211: build from this Dockerfile on the host instead of pulling `docker_image`
    dockerfile_content: str | None = None
    # DAH-1524: the image ships sshd (and runs Jupyter itself); None keeps the validator's bootstrap
    ships_sshd: bool | None = None
    gpu_power_limits: list[GpuPowerLimit] | None = None
    cache_volumes: list[CacheVolume] | None = None


@BACKEND_MESSAGES.register
class ContainerStartRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.ContainerStartRequest
    container_name: str
    local_volume_path: str


@BACKEND_MESSAGES.register
class ContainerStopRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.ContainerStopRequest
    container_name: str


@BACKEND_MESSAGES.register
class ContainerDeleteRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.ContainerDeleteRequest
    container_name: str
    local_volume: str | None = None
    external_volume: str | None = None


@BACKEND_MESSAGES.register
class AddSshPublicKeyRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.AddSshPublicKey
    container_name: str
    user_public_keys: list[str] = []


@BACKEND_MESSAGES.register
class RemoveSshPublicKeysRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.RemoveSshPublicKeysRequest
    container_name: str
    user_public_keys: list[str] = []


@BACKEND_MESSAGES.register
class DuplicateExecutorsResponse(BackendMessage):
    """The backend's answer to `DuplicateExecutorsRequest`: executors by miner that share GPU uuids."""

    message_type: BackendMessageType = BackendMessageType.DuplicateExecutorsResponse
    executors: dict[str, list[Any]]
    rental_succeed_executors: list[str] | None = None


@BACKEND_MESSAGES.register
class ExecutorRentFinishedRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.ExecutorRentFinished


@BACKEND_MESSAGES.register
class GetPodLogsRequestFromServer(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.GetPodLogsRequestFromServer
    container_name: str


@BACKEND_MESSAGES.register
class AddDebugSshKeyRequest(ServerRequest):
    message_type: BackendMessageType = BackendMessageType.AddDebugSshKeyRequest
    public_key: str


@BACKEND_MESSAGES.register
class BackupContainerRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.BackupContainerRequest
    source_volume: str
    backup_volume_info: ExternalVolumeInfo
    backup_path: str
    source_volume_path: str
    backup_target_path: str
    auth_token: str = pydantic.Field(repr=False)
    backup_log_id: str
    volume_encrypted: bool = False
    container_name: str | None = None
    backup_engine: str = "tar_aws_cli"
    repository_pod_id: str | None = None
    repository_password: str | None = pydantic.Field(default=None, repr=False)
    failure_timeout_seconds: int = pydantic.Field(default=STORAGE_OPERATION_REPORTING_FAILURE_TIMEOUT_SECONDS, gt=0)


@BACKEND_MESSAGES.register
class RestoreContainerRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.RestoreContainerRequest
    target_volume: str
    backup_volume_info: ExternalVolumeInfo
    restore_path: str
    backup_source_path: str
    target_volume_path: str
    auth_token: str = pydantic.Field(repr=False)
    restore_log_id: str
    volume_encrypted: bool = False
    container_name: str | None = None
    backup_engine: str = "tar_aws_cli"
    repository_pod_id: str | None = None
    repository_password: str | None = pydantic.Field(default=None, repr=False)
    snapshot_id: str | None = None
    legacy_object_size_bytes: int | None = None
    failure_timeout_seconds: int = pydantic.Field(default=STORAGE_OPERATION_REPORTING_FAILURE_TIMEOUT_SECONDS, gt=0)


@BACKEND_MESSAGES.register
class CancelStorageOperationRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.CancelStorageOperationRequest
    operation_id: str


@BACKEND_MESSAGES.register
class InstallJupyterServerRequest(ContainerRequest):
    message_type: BackendMessageType = BackendMessageType.InstallJupyterServer
    container_name: str
    jupyter_port_map: tuple[int, int]
    local_volume: str | None = None
    local_volume_path: str | None = None


@BACKEND_MESSAGES.register
class GetEstimateRequest(BackendMessage):
    """Ask the validator for the exact incentive estimate of a node shape; answered by
    `validator_to_backend.EstimateResponse` with the same `request_id`. The validator's copy has no
    `message_type` field and ignores unknown keys, so the backend's copy having carried delivery stamps on
    this message never mattered."""

    message_type: BackendMessageType = BackendMessageType.GetEstimateRequest
    request_id: str = ""
    gpu_model: str
    gpu_count: int = 1
    is_rented: bool = False
    gpu_splitting: bool = False
    gpu_splitting_min_count: int | None = None
    sysbox_runtime: bool = True
    collateral_deposited: bool = True


@BACKEND_MESSAGES.register
class ForcedValidationCycleRequest(BackendMessage):
    """Staging only: start the validation cycle now. No executor — the cycle validates the whole fleet."""

    message_type: BackendMessageType = BackendMessageType.ForcedValidationCycleRequest


# --- typeless replies on the socket -----------------------------------------------------------------


class Error(pydantic.BaseModel, extra="allow"):
    msg: str
    type: str
    help: str = ""


class Response(pydantic.BaseModel, extra="forbid"):
    """The backend's answer to `AuthenticateRequest`. The one strict model of the protocol: the validator's
    copy forbids unknown keys and this mirrors it, so an unexpected key here is an error, not ignored."""

    status: Literal["error", "success"]
    errors: list[Error] = []


class RentedContainer(pydantic.BaseModel):
    name: str
    pod_id: str
    # sent by the backend; the validator's copy does not declare it (test_protocol_compat.VALIDATOR_IGNORES)
    rented_ports: list[int] = []


class RentedMachine(pydantic.BaseModel):
    miner_hotkey: str
    executor_id: str
    executor_ip_address: str
    executor_ip_port: str
    containers: list[RentedContainer]
    owner_flag: bool = False
    rented_ports: list[int] = []  # as above


class RentedMachineResponse(pydantic.BaseModel):
    """The answer to `RentedMachineRequest`: every rented machine and the current bans."""

    machines: list[RentedMachine]
    banned_guids: list[str] = []
    banned_hotkeys: list[str] = []
    banned_coldkeys: list[str] = []
    banned_provider_guids: list[str] = []


class RevenuePerGpuTypeResponse(pydantic.BaseModel):
    """The answer to `RevenuePerGpuTypeRequest`."""

    revenues: dict[str, float]


SOCKET_REPLIES: dict[str, type[pydantic.BaseModel]] = {
    model.__name__: model for model in (Response, RentedMachineResponse, RevenuePerGpuTypeResponse)
}
