from typing import Literal

from pydantic import BaseModel


class Error(BaseModel, extra="allow"):
    msg: str
    type: str
    help: str = ""


class Response(BaseModel, extra="forbid"):
    """Message sent from compute app to validator in response to AuthenticateRequest"""

    status: Literal["error", "success"]
    errors: list[Error] = []

class RentedContainer(BaseModel):
    name: str
    pod_id: str

class RentedMachine(BaseModel):
    miner_hotkey: str
    executor_id: str
    executor_ip_address: str
    executor_ip_port: str
    containers: list[RentedContainer]
    owner_flag: bool = False


class RentedMachineResponse(BaseModel):
    machines: list[RentedMachine]
    banned_guids: list[str] = []


class ExecutorUptimeResponse(BaseModel):
    executor_ip_address: str
    executor_ip_port: str
    uptime_in_minutes: int | None = None


class RevenuePerGpuTypeResponse(BaseModel):
    revenues: dict[str, float]
