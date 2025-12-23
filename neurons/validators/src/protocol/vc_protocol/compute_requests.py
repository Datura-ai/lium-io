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

class RentedPod(BaseModel):
    """Pod data within an executor."""
    pod_id: str
    container_name: str
    rented_ports: list[int] = []


class RentedExecutor(BaseModel):
    """Executor with its rented pods."""
    miner_hotkey: str
    executor_ip_address: str
    executor_ip_port: str
    pods: list[RentedPod]
    owner_flag: bool = False
    rented_ports: list[int] = []


class RentedMachine(BaseModel):
    """Machine rental information for Redis storage."""
    miner_hotkey: str
    executor_id: str
    executor_ip_address: str
    executor_ip_port: str
    container_name: str


class RentedMachineResponse(BaseModel):
    machines: list[RentedMachine]
    banned_guids: list[str] = []


class RentedExecutorsResponse(BaseModel):
    """Response with executors dict and banned GUIDs."""
    executors: dict[str, RentedExecutor]  # key = executor_id
    banned_guids: list[str] = []


class ExecutorUptimeResponse(BaseModel):
    executor_ip_address: str
    executor_ip_port: str
    uptime_in_minutes: int | None = None


class RevenuePerGpuTypeResponse(BaseModel):
    revenues: dict[str, float]
