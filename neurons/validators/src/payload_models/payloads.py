from pydantic import BaseModel, field_validator
from datura.requests.miner_requests import BaseRequest
import enum


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


class ContainerCreateRequestPayload(MinerJobRequestPayload):
    docker_image: str
    user_public_key: str
    resources: ResourceType


class ContainerStartStopRequestPayload(MinerJobRequestPayload):
    container_name: str


class ContainerDeleteRequestPayload(ContainerStartStopRequestPayload):
    volume_name: str


# validator/app BE
class AppRequestType(enum.Enum):
    MinerResourceSpecs = "MinerResourceSpecs"
    CreateContainer = "CreateContainer"
    StartContainer = "StartContainer"
    StopContainer = "StopContainer"
    DeleteContainer = "DeleteContainer"


class MinerResourceSpecsRequest(BaseRequest):
    message_type = AppRequestType.MinerResourceSpecs


class ContainerCreateRequest(BaseRequest):
    message_type = AppRequestType.CreateContainer
    payload = ContainerCreateRequestPayload
