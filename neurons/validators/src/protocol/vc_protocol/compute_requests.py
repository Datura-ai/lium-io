from datetime import datetime
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

class RentedPod(BaseModel):
    """Pod data within an executor."""
    pod_id: str
    container_name: str
    rented_ports: list[int] = []
    created_at: datetime | None = None


class RentedExecutor(BaseModel):
    """Executor with its rented pods."""
    miner_hotkey: str
    executor_ip_address: str
    executor_ip_port: str
    pods: list[RentedPod]
    owner_flag: bool = False

    def get_rented_ports(self) -> list[int]:
        """Aggregate rented ports from all pods."""
        return sorted(port for pod in self.pods for port in pod.rented_ports)


class RentedMachine(BaseModel):
    """Machine rental information for Redis storage."""
    miner_hotkey: str
    executor_id: str
    executor_ip_address: str
    executor_ip_port: str
    containers: list[RentedContainer]
    owner_flag: bool = False


class RentedMachineResponse(BaseModel):
    machines: list[RentedMachine]
    banned_guids: list[str] = []


class NetworkEMA(BaseModel):
    """EMA-smoothed network speed measurements for an executor."""

    ema_download_speed: float | None = None
    ema_upload_speed: float | None = None
    ema_verifyx_download_speed: float | None = None
    ema_verifyx_upload_speed: float | None = None


class RentedExecutorsResponse(BaseModel):
    """Response with executors dict and banned GUIDs."""
    executors: dict[str, RentedExecutor]  # key = executor_id
    banned_guids: list[str] = []
    gpu_splitting_config: dict[str, int] = {}  # executor_id → min_gpu_count_for_rental
    network_ema: dict[str, NetworkEMA] = {}  # executor_id → EMA network speeds, all active executors


class ExecutorUptimeResponse(BaseModel):
    executor_ip_address: str
    executor_ip_port: str
    uptime_in_minutes: int | None = None


class RevenuePerGpuTypeResponse(BaseModel):
    revenues: dict[str, float]


class ExecutorHealthCheckResponse(BaseModel):
    """Response from executor health check endpoint."""
    success: bool
    error: str | None = None
    details: dict | None = None
