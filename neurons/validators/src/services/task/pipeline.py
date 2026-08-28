import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Tuple, runtime_checkable

import asyncssh
from pydantic import BaseModel, Field

from datura.requests.miner_requests import ExecutorSSHInfo

from core.utils import _m
from clients.backend_client import BackendClient
from services.ssh_service import SSHService
from services.redis_service import RedisService
from services.collateral_contract_service import CollateralContractService
from services.matrix_validation_service import ValidationService
from services.verifyx_validation_service import VerifyXValidationService
from services.executor_connectivity_service import ExecutorConnectivityService
from services.interactive_shell_service import InteractiveShellService
from services.inspector_validation_service import InspectorValidationService
from services.container_cleanup import ContainerCleanup
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from .models import ValidationEvent
from .runner import SSHCommandRunner

@runtime_checkable
class PodRecoverer(Protocol):
    # The slice of DockerService the rented-machine check needs. Declared here because
    # docker_service imports this package, so naming the class itself would be an import
    # cycle — and Context is a pydantic model, so a TYPE_CHECKING-only name would leave it
    # unbuildable and every validation cycle would raise instead of running.

    async def recover_pod_after_stale_vloopback_mount(
        self,
        *,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        private_key: str,
        container_name: str,
        pod_id: str,
        container_error: str | None,
        default_extra: dict[str, Any],
    ) -> bool: ...


@dataclass(frozen=True)
class ContextServices:
    ssh: SSHService
    redis: RedisService
    collateral: CollateralContractService
    validation: ValidationService
    verifyx: VerifyXValidationService
    inspector: InspectorValidationService
    connectivity: ExecutorConnectivityService
    shell: InteractiveShellService
    score_calculator: Callable[[str, bool, bool, str, bool, int], Tuple[float, float, str]]
    backend: BackendClient
    container_cleanup: ContainerCleanup
    pod_recovery: PodRecoverer


@dataclass(frozen=True)
class ContextConfig:
    executor_root: str
    compute_rest_app_url: str
    gpu_monitor_script_relative: str
    machine_scrape_filename: str
    machine_scrape_timeout: int
    obfuscation_keys: Any
    # DAH-2380: per-cycle snapshot of default cache-template image_ref -> bare digest,
    # fetched from Docker Hub at job-cycle start. Empty => digest check skips (fail-open).
    default_docker_image_digests: dict[str, str]
    validator_keypair: Optional[Any] = None
    max_gpu_count: Optional[int] = None
    gpu_model_rates: Optional[dict[str, Any]] = None
    nvml_digest_map: Optional[dict[str, str]] = None
    # Driver versions already confirmed as spoofs (DAH-2451). The nvml_digest check
    # rejects these without re-reporting them to the backend for verification.
    nvml_invalid_drivers: Optional[list[str]] = None
    enable_no_collateral: bool = False
    verifyx_enabled: bool = False
    inspector_enabled: bool = False
    port_private_key: Optional[str] = None
    port_public_key: Optional[str] = None
    job_batch_id: Optional[str] = None
    # DAH-2794: obfuscated scrape source, piped to the executor's interpreter over stdin.
    # None => the legacy path, where the scrape is a binary uploaded by UploadFilesCheck.
    machine_scrape_source: Optional[str] = None


@dataclass(frozen=True)
class ContextState:
    upload_local_dir: Optional[str] = None
    upload_remote_dir: Optional[str] = None
    remote_dir: Optional[str] = None
    # DAH-2794: set by UploadFilesCheck when the scrape source is available. True => nothing was
    # uploaded; MachineSpecScrapeCheck pipes the source in and uploads the binary only if it fails.
    scrape_over_stdin: bool = False
    specs: dict[str, Any] = field(default_factory=dict)
    gpu_model: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_details: list[dict] = field(default_factory=list)
    gpu_processes: list[dict] = field(default_factory=list)
    sysbox_runtime: bool = False
    supports_gpu_splitting: bool = False
    gpu_splitting_min_count: int | None = None
    gpu_model_count: Optional[str] = None
    gpu_uuids: Optional[str] = None
    verified_port_count: int = 0
    rented_data: RentedExecutorsResponse | None = None
    gpu_metrics: dict | None = None
    inspector_event: dict | None = None
    # DAH-2265 Plan 2: advisory result of the cached-template verification check.
    # True/False once measured; None = not measured this cycle (skipped/fail-open).
    # ResultHandler publishes it into executor.specs when not None.
    recommended_image_cached: bool | None = None
    # DAH-2265 digest: advisory digest-match for the recommended image. True = local
    # RepoDigest matches the backend's published manifest digest; False = differs (node
    # serves STALE content under an unchanged tag); None = not compared this cycle
    # (not cached / no backend digest / unreadable RepoDigest — strict fail-open).
    recommended_image_digest_match: bool | None = None


class CheckResult(BaseModel):
    passed: bool
    event: ValidationEvent
    updates: dict[str, Any] = {}
    halt: bool = False


class Context(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    pipeline_id: str
    executor: ExecutorSSHInfo
    miner_hotkey: str
    miner_coldkey: str | None = None
    miner_address: str
    miner_port: int
    ssh: asyncssh.SSHClientConnection
    runner: SSHCommandRunner
    verified: dict = {}
    settings: dict = {}
    encrypt_key: str | None = None
    # Already decrypted: pod recovery re-runs the rental start path, which opens its own
    # connection to the host rather than reusing `ssh`.
    executor_ssh_private_key: str | None = None
    default_extra: dict[str, Any] = {}
    services: ContextServices
    config: ContextConfig
    state: ContextState = Field(default_factory=ContextState)
    clear_verified_job_info: bool = False
    clear_verified_job_reason: str | None = None
    collateral_deposited: bool = False
    collateral_error_message: str | None = None
    contract_version: str | None = None
    is_rental_succeed: bool = False
    rented: bool = False
    renting_in_progress: bool = False
    ssh_pub_keys: list[str] | None = None
    port_count: int = 0
    score: float = 0.0
    job_score: float = 0.0
    score_warning: str | None = None
    log_status: str = "info"
    log_text: str | None = None
    success: bool = False
    is_provider_banned: bool = False
    tdx_attestation_passed: bool = False
    # False only once CpuTruthCheck sees a mismatch under enforcement; the score gate lives in
    # calculate_scores because the check is non-fatal and ScoreCheck runs after it.
    cpu_truth_passed: bool = True
    # False only once ProviderSideLoadCheck sees provider-side CPU/disk above the floors under
    # enforcement; the score gate lives in calculate_scores for the same reason as above.
    provider_side_load_passed: bool = True
    # G1 — NVIDIA CC GPU attestation outcome: True/False when verified, None when
    # not performed (non-CVM node, no evidence supplied, or NRAS undeterminable).
    gpu_attestation_passed: bool | None = None


class Check(Protocol):
    check_id: str
    fatal: bool

    async def run(self, ctx: Context) -> CheckResult: ...


class EventSink(Protocol):
    async def emit(self, event: ValidationEvent) -> None: ...


class LoggerSink:
    def __init__(self, logger_: logging.Logger):
        self.logger = logger_

    async def emit(self, event: ValidationEvent) -> None:
        level = {"info": "info", "warning": "warning", "error": "error"}[event.severity]
        getattr(self.logger, level)(_m(event.event, extra=event.model_dump(mode="json")))


class Pipeline:
    def __init__(self, checks: List[Check], sink: EventSink):
        self.checks = checks
        self.sink = sink

    async def run(self, ctx: Context) -> Tuple[bool, list[ValidationEvent], Context]:
        events: list[ValidationEvent] = []
        current_ctx = ctx
        pipeline_start_time = time.perf_counter()

        for chk in self.checks:
            check_start_time = time.perf_counter()
            res = await chk.run(current_ctx)
            check_end_time = time.perf_counter()

            execution_time_ms = int((check_end_time - check_start_time) * 1000)
            elapsed_time_ms = int((check_end_time - pipeline_start_time) * 1000)

            res.event.context["execution_time_ms"] = execution_time_ms
            res.event.context["elapsed_time_ms"] = elapsed_time_ms

            await self.sink.emit(res.event)
            events.append(res.event)

            if res.updates:
                current_ctx = current_ctx.model_copy(update=res.updates)

            if not res.passed and getattr(chk, "fatal", False):
                return False, events, current_ctx

            if res.halt:
                return True, events, current_ctx

        return True, events, current_ctx
