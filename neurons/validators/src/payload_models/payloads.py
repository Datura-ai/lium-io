import enum

from datura.requests.base import BaseRequest
from datura.requests.miner_requests import PodLog
from pydantic import BaseModel, field_validator


class CustomOptions(BaseModel):
    volumes: list[str] | None = None
    environment: dict[str, str] | None = None
    entrypoint: str | None = None
    internal_ports: list[int] | None = None
    startup_commands: str | None = None
    shm_size: str | None = None


class MinerJobRequestPayload(BaseModel):
    job_batch_id: str
    miner_hotkey: str
    miner_coldkey: str
    miner_address: str
    miner_port: int


class MinerJobEnryptedFiles(BaseModel):
    encrypt_key: str
    all_keys: dict
    tmp_directory: str
    machine_scrape_file_name: str
    # score_file_name: str


class ResourceType(BaseModel):
    cpu: int
    gpu: int
    memory: str
    volume: str

    @field_validator("cpu", "gpu")
    def validate_positive_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"{v} should be a valid non-negative integer string.")
        return v

    @field_validator("memory", "volume")
    def validate_memory_format(cls, v: str) -> str:
        if not v[:-2].isdigit() or v[-2:].upper() not in ["MB", "GB"]:
            raise ValueError(f"{v} is not a valid format.")
        return v


class ContainerRequestType(enum.Enum):
    ContainerCreateRequest = "ContainerCreateRequest"
    ContainerStartRequest = "ContainerStartRequest"
    ContainerStopRequest = "ContainerStopRequest"
    ContainerDeleteRequest = "ContainerDeleteRequest"
    AddSshPublicKey = "AddSshPublicKey"
    DuplicateExecutorsResponse = "DuplicateExecutorsResponse"
    ExecutorRentFinished = "ExecutorRentFinished"
    GetPodLogsRequestFromServer = "GetPodLogsRequestFromServer"
    AddDebugSshKeyRequest = "AddDebugSshKeyRequest"
    BackupContainerRequest = "BackupContainerRequest"
    RestoreContainerRequest = "RestoreContainerRequest"


class ContainerBaseRequest(BaseRequest):
    message_type: ContainerRequestType
    miner_hotkey: str
    miner_address: str | None = None
    miner_port: int | None = None
    executor_id: str


class ExternalVolumeInfo(BaseModel):
    name: str
    plugin: str
    iam_user_access_key: str
    iam_user_secret_key: str


class ContainerCreateRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerCreateRequest
    docker_image: str
    user_public_keys: list[str] = []
    custom_options: CustomOptions | None = None
    debug: bool | None = None
    local_volume: str | None = None
    external_volume_info: ExternalVolumeInfo | None = None
    is_sysbox: bool | None = None
    docker_username: str | None = None  # when edit pod, docker_username is required
    docker_password: str | None = None  # when edit pod, docker_password is required
    timestamp: int | None = None


class ExecutorRentFinishedRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ExecutorRentFinished


class ContainerStartRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStartRequest
    container_name: str


class AddSshPublicKeyRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.AddSshPublicKey
    container_name: str
    user_public_keys: list[str] = []


class AddDebugSshKeyRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.AddDebugSshKeyRequest
    public_key: str


class ContainerStopRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStopRequest
    container_name: str


class ContainerDeleteRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerDeleteRequest
    container_name: str
    local_volume: str | None = None
    external_volume: str | None = None


class GetPodLogsRequestFromServer(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.GetPodLogsRequestFromServer
    container_name: str


class BackupContainerRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.BackupContainerRequest
    source_volume: str
    backup_volume_info: ExternalVolumeInfo  # S3 backup volume with credentials
    backup_path: str
    source_volume_path: str
    backup_target_path: str
    auth_token: str  # JWT for progress updates
    backup_log_id: str


class RestoreContainerRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.RestoreContainerRequest
    target_volume: str
    backup_volume_info: ExternalVolumeInfo  # S3 backup volume with credentials
    backup_source_path: str  # path in backup S3 volume
    target_volume_path: str  # local volume mounted path
    auth_token: str  # JWT for progress updates
    restore_log_id: str
    restore_path: str


##############################################################
# Response payloads
##############################################################

class ContainerResponseType(enum.Enum):
    ContainerCreated = "ContainerCreated"
    ContainerStarted = "ContainerStarted"
    ContainerStopped = "ContainerStopped"
    ContainerDeleted = "ContainerDeleted"
    SshPubKeyAdded = "SshPubKeyAdded"
    FailedRequest = "FailedRequest"
    PodLogsResponseToServer = "PodLogsResponseToServer"
    FailedGetPodLogs = "FailedGetPodLogs"
    DebugSshKeyAdded = "DebugSshKeyAdded"
    FailedAddDebugSshKey = "FailedAddDebugSshKey"


class ContainerBaseResponse(BaseRequest):
    message_type: ContainerResponseType
    miner_hotkey: str
    executor_id: str


class ContainerCreated(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerCreated
    container_name: str
    volume_name: str
    port_maps: list[tuple[int, int]]
    profilers: list[dict] = []


class ContainerStarted(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerStarted
    container_name: str


class ContainerStopped(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerStopped
    container_name: str


class ContainerDeleted(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerDeleted


class SshPubKeyAdded(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.SshPubKeyAdded
    user_public_keys: list[str] = []


class DebugSshKeyAdded(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.DebugSshKeyAdded
    address: str
    port: int
    ssh_username: str
    ssh_port: int


class FailedContainerErrorCodes(enum.Enum):
    UnknownError = "UnknownError"
    NoSshKeys = "NoSshKeys"
    ContainerNotRunning = "ContainerNotRunning"
    NoPortMappings = "NoPortMappings"
    InvalidExecutorId = "InvalidExecutorId"
    ExceptionError = "ExceptionError"
    FailedMsgFromMiner = "FailedMsgFromMiner"
    RentingInProgress = "RentingInProgress"


class FailedContainerErrorTypes(enum.Enum):
    ContainerCreationFailed = "ContainerCreationFailed"
    ContainerDeletionFailed = "ContainerDeletionFailed"
    ContainerStopFailed = "ContainerStopFailed"
    ContainerStartFailed = "ContainerStartFailed"
    AddSSkeyFailed = "AddSSkeyFailed"
    UnknownRequest = "UnknownRequest"


class FailedContainerRequest(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedRequest
    error_type: FailedContainerErrorTypes = FailedContainerErrorTypes.ContainerCreationFailed
    msg: str
    error_code: FailedContainerErrorCodes | None = None


class DuplicateExecutorsResponse(BaseModel):
    message_type: ContainerRequestType = ContainerRequestType.DuplicateExecutorsResponse
    executors: dict[str, list]
    rental_succeed_executors: list[str] | None = None


class PodLogsResponseToServer(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.PodLogsResponseToServer
    container_name: str
    logs: list[PodLog] = []


class FailedGetPodLogs(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedGetPodLogs
    container_name: str
    msg: str


class FailedAddDebugSshKey(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedAddDebugSshKey
    msg: str
