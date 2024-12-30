import enum

from datura.requests.base import BaseRequest
from pydantic import BaseModel, field_validator


class CustomOptions(BaseModel):
    volumes: list[str] | None = None
    environment: dict[str, str] | None = None
    entrypoint: str | None = None
    internal_ports: list[int] | None = None
    startup_commands: str | None = None


class MinerJobRequestPayload(BaseModel):
    job_batch_id: str
    miner_hotkey: str
    miner_address: str
    miner_port: int


class MinerJobEnryptedFiles(BaseModel):
    encrypt_key: str
    tmp_directory: str
    machine_scrape_file_name: str
    score_file_name: str


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
    DuplicateContainersResponse = "DuplicateContainersResponse"


class ContainerBaseRequest(BaseRequest):
    message_type: ContainerRequestType
    miner_hotkey: str
    miner_address: str | None = None
    miner_port: int | None = None
    executor_id: str


class ContainerCreateRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerCreateRequest
    docker_image: str
    user_public_key: str
    custom_options: CustomOptions | None = None
    debug: bool | None = None


class ContainerStartRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStartRequest
    container_name: str


class ContainerStopRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerStopRequest
    container_name: str


class ContainerDeleteRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerDeleteRequest
    container_name: str
    volume_name: str


class ContainerResponseType(enum.Enum):
    ContainerCreated = "ContainerCreated"
    ContainerStarted = "ContainerStarted"
    ContainerStopped = "ContainerStopped"
    ContainerDeleted = "ContainerDeleted"
    FailedRequest = "FailedRequest"


class ContainerBaseResponse(BaseRequest):
    message_type: ContainerResponseType
    miner_hotkey: str
    executor_id: str


class ContainerCreatedResult(BaseModel):
    container_name: str
    volume_name: str
    port_maps: list[tuple[int, int]]


class ContainerCreated(ContainerBaseResponse, ContainerCreatedResult):
    message_type: ContainerResponseType = ContainerResponseType.ContainerCreated


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


class FailedContainerErrorCodes(enum.Enum):
    UnknownError = "UnknownError"
    ContainerNotRunning = "ContainerNotRunning"
    NoPortMappings = "NoPortMappings"
    InvalidExecutorId = "InvalidExecutorId"
    ExceptionError = "ExceptionError"
    FailedMsgFromMiner = "FailedMsgFromMiner"


class FailedContainerRequest(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FailedRequest
    msg: str
    error_code: FailedContainerErrorCodes | None = None


class DuplicateContainersResponse(BaseModel):
    message_type: ContainerRequestType = ContainerRequestType.DuplicateContainersResponse
    containers: dict[str, list]
