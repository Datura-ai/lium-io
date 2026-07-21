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
    # GPUs this pod holds (DAH-2467). None = backend predates the field; the validator then
    # treats the whole executor as rented (no per-GPU split scoring).
    gpu_count: int | None = None


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


class ManualRentalInfo(BaseModel):
    """Specs a special manual (bare-metal) rental is force-passed against.

    A manually-rented node is never contacted, so these cannot be scraped — the backend sends
    what the node was last verified to have, and it is the sole basis for the forced score.
    Both fields are load-bearing: the incentive layer maps gpu_model to a total-count denominator
    and multiplies by gpu_count, so a missing model or a zero count means zero emission.
    """
    gpu_model: str
    gpu_count: int


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
    # executor_id → "miner" | "lium"; absent = no default job. Parsed leniently as str for
    # forward-compatibility (a future owner value must not break parsing of the whole response).
    default_job_owner_by_executor: dict[str, str] = {}
    # executor_id → specs to force-pass against. Present only for executors carrying a pod flagged
    # as a special manual (bare-metal) rental. Defaults to empty so an older backend that omits the
    # field force-passes nobody (fail-closed) rather than everybody.
    manual_rental_executors: dict[str, ManualRentalInfo] = {}

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

    def get_manual_rental_info(self, executor_uuid: str) -> "ManualRentalInfo | None":
        """Specs to force-pass this executor against, or None if it is not a manual rental."""
        return self.manual_rental_executors.get(str(executor_uuid))


class PodRentalActiveResponse(BaseModel):
    active: bool
    rental_closed_at: datetime | None = None


class FillerRunActiveResponse(BaseModel):
    active: bool
    status: str | None = None
    started_at: datetime | None = None


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
    # Bare manifest-list digest ("sha256:…"). Populated by the backend again as of
    # DAH-2461 (resolved live from Docker Hub) so executors can pull digest-pinned;
    # the validator still verifies against its OWN Hub snapshot (DAH-2380), not this field.
    docker_image_digest: str | None = None

    @property
    def image_ref(self) -> str:
        return f"{self.docker_image}:{self.docker_image_tag}"


class DefaultDockerImagesResponse(RootModel[list[DefaultDockerImage]]):
    """The backend returns a bare JSON list; wrap it so `BackendClient.get()`
    (which validates with a single model) can parse it."""
