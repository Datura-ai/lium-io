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


class ContainerBaseRequest(BaseRequest):
    message_type: ContainerRequestType
    miner_hotkey: str
    miner_address: str | None = None
    miner_port: int | None = None
    executor_id: str


class ContainerCreateRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerCreateRequest
    docker_image: str
    user_public_keys: list[str] = []
    custom_options: CustomOptions | None = None
    debug: bool | None = None
    volume_name: str | None = None  # when edit pod, volume_name is required
    is_sysbox: bool | None = None


class ExecutorRentFinishedRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ExecutorRentFinished


class ContainerStartRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStartRequest
    container_name: str


class AddSshPublicKeyRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.AddSshPublicKey
    container_name: str
    user_public_keys: list[str] = []


class ContainerStopRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStopRequest
    container_name: str


class ContainerDeleteRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerDeleteRequest
    container_name: str
    volume_name: str


class GetPodLogsRequestFromServer(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.GetPodLogsRequestFromServer
    container_name: str


class ContainerResponseType(enum.Enum):
    ContainerCreated = "ContainerCreated"
    ContainerStarted = "ContainerStarted"
    ContainerStopped = "ContainerStopped"
    ContainerDeleted = "ContainerDeleted"
    SshPubKeyAdded = "SshPubKeyAdded"
    FailedRequest = "FailedRequest"
    PodLogsResponseToServer = "PodLogsResponseToServer"
    FailedGetPodLogs = "FailedGetPodLogs"


class ContainerBaseResponse(BaseRequest):
    message_type: ContainerResponseType
    miner_hotkey: str
    executor_id: str


class ContainerCreated(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerCreated
    container_name: str
    volume_name: str
    port_maps: list[tuple[int, int]]


class ContainerStarted(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerStarted
    container_name: str


class ContainerStopped(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerStopped
    container_name: str


class ContainerDeleted(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.ContainerDeleted
    container_name: str
    volume_name: str


class SshPubKeyAdded(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.SshPubKeyAdded
    user_public_keys: list[str] = []


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
