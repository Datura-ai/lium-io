from typing import List, Tuple
from pydantic import BaseModel, field_validator
import enum
from datura.requests.base import BaseRequest


class MinerJobRequestPayload(BaseModel):
    miner_hotkey: str
    miner_address: str
    miner_port: int


class ResourceType(BaseModel):
    cpu: int
    gpu: int
    memory: str
    volume: str

    @field_validator('cpu', 'gpu')
    def validate_positive_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f'{v} should be a valid non-negative integer string.')
        return v

    @field_validator('memory', 'volume')
    def validate_memory_format(cls, v: str) -> str:
        if not v[:-2].isdigit() or v[-2:].upper() not in ['MB', 'GB']:
            raise ValueError(f'{v} is not a valid format.')
        return v


class ContainerRequestType(enum.Enum):
    ContainerCreateRequest = "ContainerCreateRequest"
    ContainerStartRequest = "ContainerStartRequest"
    ContainerStopRequest = "ContainerStopRequest"
    ContainerDeleteRequest = "ContainerDeleteRequest"


class ContainerBaseRequest(BaseRequest):
    message_type: ContainerRequestType
    miner_hotkey: str
    miner_address: str
    miner_port: int
    executor_id: str


class ContainerCreateRequest(ContainerBaseRequest):
    message_type: ContainerRequestType = ContainerRequestType.ContainerCreateRequest
    docker_image: str
    user_public_key: str
    resources: ResourceType


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
    ContainerStarted = "ContainerStarted",
    ContainerStopped = "ContainerStopped"
    ContainerDeleted = "ContainerDeleted"
    FaildRequest = "FailedRequest"


class ContainerBaseResponse(BaseRequest):
    message_type: ContainerResponseType
    miner_hotkey: str
    executor_id: str


class ContainerCreatedResult(BaseModel):
    container_name: str
    volume_name: str
    port_maps: List[Tuple[int, int]]


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


class FaildContainerRequest(ContainerBaseResponse):
    message_type: ContainerResponseType = ContainerResponseType.FaildRequest
    msg: str
