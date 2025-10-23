import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple

import asyncssh
from pydantic import BaseModel, Field

from datura.requests.miner_requests import ExecutorSSHInfo

from core.utils import _m
from .models import ValidationEvent
@dataclass(frozen=True)
class ContextServices:
    ssh: Any
    redis: Any
    collateral: Any
    validation: Any
    verifyx: Any
    connectivity: Any
    validator_keypair: Optional[Any] = None


@dataclass(frozen=True)
class ContextConfig:
    executor_root: str
    compute_rest_app_url: str
    gpu_monitor_script_relative: str
    machine_scrape_filename: str
    machine_scrape_timeout: int
    obfuscation_keys: Any


@dataclass(frozen=True)
class ContextState:
    upload_local_dir: Optional[str] = None
    upload_remote_dir: Optional[str] = None


class CheckResult(BaseModel):
    passed: bool
    event: ValidationEvent
    updates: dict[str, Any] = {}
    halt: bool = False


class Context(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}
    executor: ExecutorSSHInfo
    miner_hotkey: str
    ssh: asyncssh.SSHClientConnection
    runner: Any
    specs: dict = {}
    verified: dict = {}
    settings: dict = {}
    encrypt_key: str | None = None
    remote_dir: str | None = None
    default_extra: dict[str, Any] = {}
    services: ContextServices
    config: ContextConfig
    state: ContextState = Field(default_factory=ContextState)
    gpu_model_count: str | None = None
    gpu_uuids: str | None = None
    clear_verified_job_info: bool = False
    clear_verified_job_reason: str | None = None
    gpu_model: str | None = None
    gpu_count: int = 0
    gpu_details: list[dict] = []
    gpu_processes: list[dict] = []
    sysbox_runtime: bool = False
    collateral_deposited: bool = False
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

        for chk in self.checks:
            res = await chk.run(current_ctx)

            await self.sink.emit(res.event)
            events.append(res.event)

            if res.updates:
                current_ctx = current_ctx.model_copy(update=res.updates)

            if not res.passed and getattr(chk, "fatal", False):
                return False, events, current_ctx

            if res.halt:
                return True, events, current_ctx

        return True, events, current_ctx
