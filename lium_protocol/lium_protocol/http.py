"""Bodies of the backend HTTP API the validator calls between cycles (`clients/backend_client.py`, and
`clients/compute_client.py` for the uptime poll), as the validator must be able to parse them. The three typeless replies the backend sends down the
WebSocket (`Response`, `RentedMachineResponse`, `RevenuePerGpuTypeResponse`) are in
`backend_to_validator.SOCKET_REPLIES`, not here.

Wire models only: the validator's copy (`vc_protocol/compute_requests.py`) adds filtering and lookup
helpers on top of these shapes; the backend's (`compute_app_requests.py`) is what it serialises.
Where the two differ (`RentedPod.created_at` required on one side) the field is optional here, so
either copy parses the other's output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pydantic


class RentedPod(pydantic.BaseModel):
    pod_id: str
    container_name: str
    rented_ports: list[int] = []
    created_at: datetime | None = None
    # DAH-2467: GPUs this pod holds; None = the backend predates the field, the validator then scores
    # the whole executor as rented
    gpu_count: int | None = None


class RentedExecutor(pydantic.BaseModel):
    miner_hotkey: str
    executor_ip_address: str
    executor_ip_port: str
    pods: list[RentedPod]
    owner_flag: bool = False


class NetworkEMA(pydantic.BaseModel):
    """EMA-smoothed network speed measurements for an executor."""

    ema_download_speed: float | None = None
    ema_upload_speed: float | None = None
    ema_verifyx_download_speed: float | None = None
    ema_verifyx_upload_speed: float | None = None


class ManualRentalInfo(pydantic.BaseModel):
    """Specs a special manual (bare-metal) rental is force-passed against; both fields drive the score."""

    gpu_model: str
    gpu_count: int


class RentedExecutorsResponse(pydantic.BaseModel):
    """`GET /internal/executors/rented`: every rented executor, the fillers to protect, the bans."""

    executors: dict[str, RentedExecutor]  # key = executor_id
    banned_guids: list[str] = []
    banned_hotkeys: list[str] = []
    banned_coldkeys: list[str] = []
    banned_provider_guids: list[str] = []
    # legacy single-filler map, kept for a peer that predates the list below
    filler_containers_by_executor: dict[str, str] = {}
    # executor_id → every active filler container name (DAH-2465)
    all_filler_containers_by_executor: dict[str, list[str]] = {}
    # executor_id → external ports its active fillers hold (DAH-2527)
    filler_ports_by_executor: dict[str, list[int]] = {}
    gpu_splitting_config: dict[str, int] = {}  # executor_id → min_gpu_count_for_rental
    network_ema: dict[str, NetworkEMA] = {}
    spot_executor_ids: list[str] = []
    new_rentals_paused_executor_ids: list[str] = []
    # DAH-2703: executor_ids whose filler container was destroyed during create
    filler_create_kill_executor_ids: list[str] = []
    provider_discord_connected_executor_ids: list[str] | None = None
    default_job_owner_by_executor: dict[str, str] = {}  # executor_id → "miner" | "lium"
    manual_rental_executors: dict[str, ManualRentalInfo] = {}


class PodRentalActiveResponse(pydantic.BaseModel):
    active: bool
    # DAH-2757: the pod's own state and the executor it belongs to
    status: str | None = None
    executor_id: str | None = None
    rental_closed_at: datetime | None = None
    # DAH-2545: where an encrypted rental volume is mounted in plaintext inside the container
    local_volume_path: str | None = None


class PodHostRebootRecoveredRequest(pydantic.BaseModel):
    """`POST …/host-reboot-recovered`: the container exit the recovery reacted to."""

    container_finished_at: datetime


class PodHostRebootRecoveredResponse(pydantic.BaseModel):
    recorded: bool


class FillerRunActiveResponse(pydantic.BaseModel):
    active: bool
    executor_id: str | None = None
    status: str | None = None
    started_at: datetime | None = None


# One item of `POST /executors` on the compute-app REST API, read by `compute_client.get_executors_uptime`
# every 20 minutes (`poll_executors_uptime`). A comment, not a docstring: a docstring lands in the schema snapshot.
class ExecutorUptimeResponse(pydantic.BaseModel):
    executor_ip_address: str
    executor_ip_port: str
    uptime_in_minutes: int | None = None


class ExecutorHealthCheckResponse(pydantic.BaseModel):
    success: bool
    error: str | None = None
    details: dict[str, Any] | None = None
    reason_code: str | None = None


class DefaultDockerImage(pydantic.BaseModel, extra="allow"):
    """One item of `GET /executors/default-docker-image`; parsed leniently so new backend fields never
    break validation."""

    docker_image: str
    docker_image_tag: str
    docker_image_size: int | None = None
    docker_image_digest: str | None = None


class DefaultDockerImagesResponse(pydantic.RootModel[list[DefaultDockerImage]]):
    """The backend returns a bare JSON list."""


class NvmlReportAckResponse(pydantic.BaseModel):
    """Ack for a reported unknown driver (DAH-2451); the body is intentionally near-empty."""

    status: str | None = None


# every HTTP body, by the name the schema snapshot files it under
HTTP_MODELS: dict[str, type[pydantic.BaseModel]] = {
    model.__name__: model
    for model in (
        RentedExecutorsResponse,
        PodRentalActiveResponse,
        PodHostRebootRecoveredRequest,
        PodHostRebootRecoveredResponse,
        FillerRunActiveResponse,
        ExecutorUptimeResponse,
        ExecutorHealthCheckResponse,
        DefaultDockerImagesResponse,
        NvmlReportAckResponse,
    )
}
