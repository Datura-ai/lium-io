from datetime import datetime
from typing import Literal

from pydantic import BaseModel, RootModel, field_validator

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
    default_job_opted_out_executor_ids: list[str] = []  # executor_ids opted out of the Lium default job (no unrented incentive)
    provider_discord_connected_executor_ids: list[str] | None = None  # executor_ids whose provider has connected Discord
    # executor_id → "miner" | "lium"; absent = no default job. Parsed leniently as str for
    # forward-compatibility (a future owner value must not break parsing of the whole response).
    default_job_owner_by_executor: dict[str, str] = {}

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

    def get_default_job_owner(self, executor_uuid: str) -> str | None:
        return self.default_job_owner_by_executor.get(str(executor_uuid))


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


class DefaultDockerImage(BaseModel, extra="allow"):
    """One recommended/default template image for a GPU + driver combo.

    Mirrors a single item from the backend `/executors/default-docker-image`
    response (the same endpoint the executor cache pre-pull consumes). Parsed
    leniently so new backend fields never break validation.
    """
    docker_image: str
    docker_image_tag: str
    docker_image_size: int | None = None
    # DAH-2265: bare manifest-list digest ("sha256:…") the backend says is current for this
    # image. None when the backend has no recorded digest. Compared against the executor's
    # local RepoDigest to flag stale content cached under an unchanged tag.
    docker_image_digest: str | None = None

    @property
    def image_ref(self) -> str:
        return f"{self.docker_image}:{self.docker_image_tag}"


class DefaultDockerImagesResponse(RootModel[list[DefaultDockerImage]]):
    """The backend returns a bare JSON list; wrap it so `BackendClient.get()`
    (which validates with a single model) can parse it."""
