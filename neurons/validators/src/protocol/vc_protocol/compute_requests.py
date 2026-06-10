from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from services.const import FILLER_CONTAINER_PREFIX

GPU_RUNTIME_NVML_MISMATCH_REASON = "GPU_RUNTIME_NVML_MISMATCH"


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
    filler_containers_by_executor: dict[str, str] = {}  # executor_id -> filler_<FillerRun.id>
    banned_guids: list[str] = []
    gpu_splitting_config: dict[str, int] = {}  # executor_id → min_gpu_count_for_rental
    network_ema: dict[str, NetworkEMA] = {}  # executor_id → EMA network speeds, all active executors
    spot_executor_ids: list[str] = []  # executor_ids in spot tier (no incentive, no penalty)
    new_rentals_paused_executor_ids: list[str] = []  # executor_ids paused from unrented incentives
    provider_discord_connected_executor_ids: list[str] | None = None  # executor_ids whose provider has connected Discord

    @field_validator("filler_containers_by_executor")
    @classmethod
    def keep_only_filler_containers(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            executor_id: container_name
            for executor_id, container_name in value.items()
            if container_name.startswith(FILLER_CONTAINER_PREFIX)
        }

    def get_filler_container(self, executor_uuid: str) -> str | None:
        return self.filler_containers_by_executor.get(str(executor_uuid))


class PodRentalActiveResponse(BaseModel):
    active: bool
    rental_closed_at: datetime | None = None


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
    reason_code: str | None = None
