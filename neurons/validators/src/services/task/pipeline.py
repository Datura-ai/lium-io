import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import asyncssh
from clients.backend_client import BackendClient
from datura.requests.miner_requests import ExecutorSSHInfo
from lium_core.shared_config.client import SharedConfigClient
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from pydantic import BaseModel, Field
from services.collateral_contract_service import CollateralContractService
from services.container_cleanup import ContainerCleanup
from services.executor_connectivity_service import ExecutorConnectivityService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.redis_service import RedisService
from services.verifyx_validation_service import VerifyXValidationService

from core.utils import _m
from services.ssh_service import SSHService

from .models import ValidationEvent
from .runner import SSHCommandRunner


@dataclass(frozen=True)
class ContextServices:
    ssh: SSHService
    redis: RedisService
    collateral: CollateralContractService
    validation: ValidationService
    verifyx: VerifyXValidationService
    connectivity: ExecutorConnectivityService
    shell: InteractiveShellService
    score_calculator: Callable[[str, bool, bool, str, bool, int], tuple[float, float, str]]
    backend: BackendClient
    container_cleanup: ContainerCleanup
    shared_config_client: SharedConfigClient | None = None


@dataclass(frozen=True)
class ContextConfig:
    executor_root: str
    compute_rest_app_url: str
    gpu_monitor_script_relative: str
    machine_scrape_filename: str
    machine_scrape_timeout: int
    obfuscation_keys: Any
    validator_keypair: Any | None = None
    max_gpu_count: int | None = None
    gpu_model_rates: dict[str, Any] | None = None
    nvml_digest_map: dict[str, str] | None = None
    enable_no_collateral: bool = False
    verifyx_enabled: bool = False
    port_private_key: str | None = None
    port_public_key: str | None = None
    job_batch_id: str | None = None


@dataclass(frozen=True)
class ContextState:
    upload_local_dir: str | None = None
    upload_remote_dir: str | None = None
    remote_dir: str | None = None
    specs: dict[str, Any] = field(default_factory=dict)
    gpu_model: str | None = None
    gpu_count: int | None = None
    gpu_details: list[dict] = field(default_factory=list)
    gpu_processes: list[dict] = field(default_factory=list)
    sysbox_runtime: bool = False
    supports_gpu_splitting: bool = False
    gpu_splitting_min_count: int | None = None
    gpu_model_count: str | None = None
    gpu_uuids: str | None = None
    verified_port_count: int = 0
    rented_data: RentedExecutorsResponse | None = None


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
    miner_address: str
    miner_port: int
    ssh: asyncssh.SSHClientConnection
    runner: SSHCommandRunner
    verified: dict = {}
    settings: dict = {}
    encrypt_key: str | None = None
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
    tdx_attestation_passed: bool = False


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
    def __init__(self, checks: list[Check], sink: EventSink):
        self.checks = checks
        self.sink = sink

    async def run(self, ctx: Context) -> tuple[bool, list[ValidationEvent], Context]:
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
