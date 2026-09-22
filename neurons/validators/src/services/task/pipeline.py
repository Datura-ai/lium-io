import asyncio
import logging
import time
from dataclasses import dataclass, field, fields, replace
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
from services.executor_image_policy import ExecutorImageReport, ExpectedImageSnapshot
from services.local_verify_client import LocalVerifyOutcome
from services.interactive_shell_service import InteractiveShellService
from services.inspector_validation_service import InspectorValidationService
from services.container_cleanup import ContainerCleanup
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from .models import ValidationEvent
from .runner import SSHCommandRunner

@runtime_checkable
class PodRecoverer(Protocol):
    # The slice of DockerService the rented-machine check and the rental probe need. Declared here
    # because docker_service imports this package, so naming the class itself would be an import
    # cycle — and Context is a pydantic model, so a TYPE_CHECKING-only name would leave it
    # unbuildable and every validation cycle would raise instead of running.

    # DAH-3436: the rental probe rents the node through the same two entry points a renter's pod
    # takes (miner_service hands the backend's requests to these), so there is one start path and
    # one teardown path to keep correct. `private_key` is the Fernet-encrypted key as the backend
    # sends it; both decrypt it themselves.
    async def create_container(
        self,
        payload: Any,
        executor_info: ExecutorSSHInfo,
        keypair: Any,
        private_key: str,
    ) -> Any: ...

    async def delete_container(
        self,
        payload: Any,
        executor_info: ExecutorSSHInfo,
        keypair: Any,
        private_key: str,
    ) -> Any: ...

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
    executor_image_snapshot: ExpectedImageSnapshot | None = None
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
    # DAH-3011: True only for a never-validated executor's first, unscored verification AND with
    # FIRST_PASS_FAST_PATH_ENABLED on (resolved in PipelineFactory.build_context). The capability
    # matmul and VerifyX right-size their probes and the bandwidth gate is deferred to the first
    # scored cycle; every other check is the same.
    first_pass: bool = False


@dataclass(frozen=True)
class ContextState:
    upload_local_dir: Optional[str] = None
    upload_remote_dir: Optional[str] = None
    remote_dir: Optional[str] = None
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
    # DAH-2662: GPU UUIDs as the kernel reports them (/proc/driver/nvidia), read by BannedProviderCheck;
    # None = unreadable or not read; `kernel_gpu_uuids_read_attempted` tells the two apart so a failed read is
    # attempted once per cycle. Bans match against these too once KERNEL_GPU_BAN_ENFORCEMENT_ENABLED.
    kernel_gpu_uuids: list[str] | None = None
    kernel_gpu_uuids_read_attempted: bool = False
    # mounts that are not procfs at or under /proc/driver/nvidia/gpus ("<mount point> <fstype>");
    # non-empty = the kernel list above was withheld because it was read through them
    kernel_gpu_foreign_mounts: list[str] = field(default_factory=list)
    verified_port_count: int = 0
    # DAH-2991: orphaned rental containers the stale cleanup could not remove this cycle; they still
    # hold their published ports, so PortCountCheck names them in INSUFFICIENT_PORTS.
    orphaned_containers: list[str] = field(default_factory=list)
    # DAH-3436: the (internal, external) pairs PortConnectivityCheck proved reachable this cycle.
    # `specs["verified_ports"]` keeps only the external side for the backend; the rental probe
    # needs both to hand create_container the ports as the backend would.
    verified_port_pairs: list[tuple[int, int]] = field(default_factory=list)
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
    executor_image_report: ExecutorImageReport | None = None
    # liumd phase 1 (DAH-2834): what `POST /verify` answered this cycle, already judged. None =
    # not attempted or fell back entirely; the capability and VerifyX checks consume a judged
    # step when present and run over SSH otherwise.
    local_verify: LocalVerifyOutcome | None = None
    # Validation fast path: the collateral read CollateralPrefetchCheck started under the GPU
    # spec checks, for CollateralCheck to await instead of calling the contract itself. None =
    # no prefetch (the flag is off, or the scrape left no GPU to read for).
    collateral_prefetch: Any = None


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
    # DAH-3436: the same key as the backend sent it (Fernet-encrypted with the validator's hotkey).
    # create_container and delete_container decrypt it themselves, so the rental probe hands them
    # this one and never re-encrypts.
    executor_ssh_private_key_encrypted: str | None = None
    default_extra: dict[str, Any] = {}
    services: ContextServices
    config: ContextConfig
    state: ContextState = Field(default_factory=ContextState)
    clear_verified_job_info: bool = False
    clear_verified_job_reason: str | None = None
    # DAH-3386: what the check that cleared the verified job saw — reason_code, check_id and, for a pod found not
    # running, the container's death diagnostics. Sent to the backend with the reset so the penalty it raises
    # carries the evidence (lium-platform DAH-3385). The pipeline fills reason_code/check_id from the event when
    # the check did not set it itself.
    clear_verified_job_evidence: dict[str, Any] | None = None
    # DAH-3457: set by GpuFingerprintCheck under GPU_ANCHOR_HARD_ENABLED when the scrape shows a GPU outside the
    # anchored set. ResultHandler writes it into the executor's verified-job record, where it is sticky: the
    # node scores 0 on every later cycle under this executor id and is never re-anchored.
    gpu_anchor_broken: bool = False
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
    # False only once InspectorRentedCheck sees a provider-origin finding on a rented pod under
    # INSPECTOR_ENFORCE_ENABLED (DAH-3275); the score gate lives in calculate_scores.
    inspector_passed: bool = True
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


def updates_with_clear_verified_job_evidence(res: CheckResult, check_id: str) -> dict[str, Any]:
    """The check's updates, with ``clear_verified_job_evidence`` filled when the check clears the verified job.

    DAH-3386: every check that sets ``clear_verified_job_info`` names itself to the backend — its event's
    reason_code and its check_id — so an EXECUTOR_INACTIVE_MID_RENTAL raised from the reset says which check
    fired instead of the bare DEFAULT/POD_NOT_RUNNING enum. A check that already attached richer evidence (the
    rented-machine check adds the container's death diagnostics) keeps it; only the two names are filled in.
    """
    updates = dict(res.updates)
    if not updates.get("clear_verified_job_info"):
        return updates
    evidence = dict(updates.get("clear_verified_job_evidence") or {})
    evidence.setdefault("reason_code", res.event.reason_code)
    evidence.setdefault("check_id", res.event.check_id or check_id)
    updates["clear_verified_job_evidence"] = evidence
    return updates


def summarize_steps(
    steps: list[tuple[str, int]], elapsed_time_ms: int, failed_check_id: str | None = None
) -> dict[str, Any]:
    """What the run spent its time on, for the last event's `what_we_saw` (DAH-3012).

    The last event is the one the backend stores as the executor's `log_text` and the portal shows
    as `last_validation`, so this is how per-step durations reach the provider without a new
    field anywhere downstream.
    """
    # DAH-3019: every step the run executed, not only the slow ones. The provider's node page lists
    # this summary, and a list that silently drops the quick checks reads as "only 3 checks ran".
    summary: dict[str, Any] = {
        # a tenth of a second for the steps that carry the run's time, three decimals below a
        # second so a 4 ms check does not read as 0.0
        "steps": {
            check_id: round(ms / 1000, 1) if ms >= 1000 else round(ms / 1000, 3)
            for check_id, ms in steps
        },
        "steps_total_s": round(elapsed_time_ms / 1000, 1),
    }
    if failed_check_id:
        summary["steps_failed"] = failed_check_id
    return summary


class ProgressSink(Protocol):
    """Where the pipeline reports which check a run is on (validation fast path, support view)."""

    def step_started(self, ctx: Context, check_id: str) -> None: ...

    def step_finished(self, ctx: Context, check_id: str, event: ValidationEvent, passed: bool) -> None: ...

    def step_aborted(self, ctx: Context, check_id: str, error_class: str) -> None: ...


class ParallelStage:
    """Lanes of checks with no data dependency between them, run at once on one Context.

    Validation fast path: each lane runs its checks in order on its own copy of the context it was
    given, exactly as the serial pipeline would. The first lane to stop — a fatal failure, a halt
    or an exception — cancels the other lanes, whose in-flight check is interrupted and whose
    later checks never start, so a failing run does the same work as the serial one would have
    (a matmul that fails never lets the host lane rent the probe container). The pipeline then
    applies the completed results in lane order — events, step timings, context updates — so a
    run reads like the serial one: the stopping check's event ends it, a check's `updates` land
    the way they do today, and `state` changes are merged field by field (`specs` key by key)
    because each lane changed a disjoint part of it. Checks the sibling lane completed before the
    cancel have run but are not emitted and not counted in the step summary. Nothing here decides
    pass or fail; every check keeps its own verdict.
    """

    check_id = "pipeline.parallel"
    fatal = False

    def __init__(self, lanes: list[list[Check]]):
        self.lanes = [list(lane) for lane in lanes if lane]

    @property
    def checks(self) -> list[Check]:
        return [chk for lane in self.lanes for chk in lane]

    async def run(self, ctx: Context) -> CheckResult:
        # The pipeline runs the lanes itself (Pipeline._run_step); a stage is not a check of its own.
        raise NotImplementedError("ParallelStage is run by Pipeline, lane by lane")


@dataclass(frozen=True)
class _RanCheck:
    check: Check
    result: CheckResult
    before_state: ContextState
    started: float
    finished: float


def merge_state(current: ContextState, before: ContextState, after: ContextState) -> ContextState:
    """`current` with every field one lane changed (`before` → `after`) applied to it.

    `specs` is merged key by key so two lanes that each wrote their own keys (VerifyX: ram/network/
    hard_disk; ports: verified_ports) both land; a key both wrote takes the later lane's value, the
    way the later check wins in the serial pipeline.
    """
    changes: dict[str, Any] = {}
    for f in fields(ContextState):
        before_value = getattr(before, f.name)
        after_value = getattr(after, f.name)
        if after_value is before_value or after_value == before_value:
            continue
        if f.name == "specs" and isinstance(before_value, dict) and isinstance(after_value, dict):
            merged_specs = dict(getattr(current, "specs") or {})
            for key in set(before_value) - set(after_value):
                merged_specs.pop(key, None)
            for key, value in after_value.items():
                if key not in before_value or before_value[key] != value:
                    merged_specs[key] = value
            changes["specs"] = merged_specs
            continue
        changes[f.name] = after_value
    return replace(current, **changes) if changes else current


def _stops_run(chk: Check, res: CheckResult) -> bool:
    return (not res.passed and getattr(chk, "fatal", False)) or res.halt


def cancel_pending_collateral_prefetch(ctx: Context) -> bool:
    """Cancel a collateral read the fast path started that no check consumed (the run ended
    before CollateralCheck). Returns whether one was cancelled."""
    prefetch = getattr(ctx.state, "collateral_prefetch", None)
    task = getattr(prefetch, "task", None)
    if task is None or task.done():
        return False
    task.cancel()
    return True


class Pipeline:
    def __init__(self, checks: List[Check], sink: EventSink, progress: ProgressSink | None = None):
        self.checks = checks
        self.sink = sink
        self.progress = progress

    async def _run_check(self, chk: Check, ctx: Context) -> _RanCheck:
        if self.progress is not None:
            self.progress.step_started(ctx, chk.check_id)
        started = time.perf_counter()
        try:
            res = await chk.run(ctx)
        except BaseException as exc:
            # The exception class is what support sees; the text stays in the run's own log line.
            if self.progress is not None:
                self.progress.step_aborted(ctx, chk.check_id, type(exc).__name__)
            raise
        finished = time.perf_counter()
        return _RanCheck(check=chk, result=res, before_state=ctx.state, started=started, finished=finished)

    async def _run_lane(self, lane: list[Check], ctx: Context, ran: list[_RanCheck]) -> None:
        """One lane of a ParallelStage, serially, stopping where the serial pipeline would.

        Completed checks are appended to `ran` as they finish, so a lane cancelled by its sibling
        still hands over what it completed."""
        current = ctx
        for chk in lane:
            step = await self._run_check(chk, current)
            ran.append(step)
            res = step.result
            if _stops_run(chk, res):
                return
            if res.updates:
                current = current.model_copy(update=updates_with_clear_verified_job_evidence(res, chk.check_id))

    async def _run_stage(self, stage: ParallelStage, ctx: Context) -> list[_RanCheck]:
        """Run the lanes at once; the first lane to stop (fatal, halt or exception) cancels the rest."""
        lane_results: list[list[_RanCheck]] = [[] for _ in stage.lanes]
        tasks = [
            asyncio.ensure_future(self._run_lane(lane, ctx, lane_results[index]))
            for index, lane in enumerate(stage.lanes)
        ]
        first_error: BaseException | None = None
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                stop = False
                for task in done:
                    exc = asyncio.CancelledError() if task.cancelled() else task.exception()
                    if exc is not None:
                        first_error = first_error or exc
                        stop = True
                        continue
                    index = tasks.index(task)
                    if lane_results[index] and _stops_run(lane_results[index][-1].check, lane_results[index][-1].result):
                        stop = True
                if stop:
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if first_error is not None:
            raise first_error
        return [ran for lane in lane_results for ran in lane]

    async def _run_step(self, step: Check | ParallelStage, ctx: Context) -> list[_RanCheck]:
        if isinstance(step, ParallelStage):
            return await self._run_stage(step, ctx)
        return [await self._run_check(step, ctx)]

    def _apply(self, current_ctx: Context, ran: _RanCheck, parallel: bool) -> Context:
        res = ran.result
        if not res.updates:
            return current_ctx
        updates = updates_with_clear_verified_job_evidence(res, ran.check.check_id)
        if parallel and "state" in updates:
            updates["state"] = merge_state(current_ctx.state, ran.before_state, updates["state"])
        return current_ctx.model_copy(update=updates)

    async def run(self, ctx: Context) -> Tuple[bool, list[ValidationEvent], Context]:
        latest = [ctx]
        try:
            return await self._run(ctx, latest)
        except BaseException:
            # A run that raises leaves no check to consume the early collateral read.
            cancel_pending_collateral_prefetch(latest[0])
            raise

    async def _run(self, ctx: Context, latest: list[Context]) -> Tuple[bool, list[ValidationEvent], Context]:
        events: list[ValidationEvent] = []
        current_ctx = ctx
        pipeline_start_time = time.perf_counter()
        steps: list[tuple[str, int]] = []
        last_index = len(self.checks) - 1

        for index, step in enumerate(self.checks):
            parallel = isinstance(step, ParallelStage)
            ran_checks = await self._run_step(step, current_ctx)
            for ran in ran_checks:
                ran.result.event.context["execution_time_ms"] = int((ran.finished - ran.started) * 1000)
                ran.result.event.context["elapsed_time_ms"] = int((ran.finished - pipeline_start_time) * 1000)

            # A stage's wall time is its slowest lane's, whichever lane's event carries the summary.
            stage_elapsed_ms = max(ran.result.event.context["elapsed_time_ms"] for ran in ran_checks)
            for position, ran in enumerate(ran_checks):
                chk, res = ran.check, ran.result
                # Only emitted checks enter the summary: a sibling lane's checks completed before
                # the cancel ran, but the run does not report them.
                steps.append((chk.check_id, res.event.context["execution_time_ms"]))
                elapsed_time_ms = stage_elapsed_ms if parallel else res.event.context["elapsed_time_ms"]
                failed = not res.passed and getattr(chk, "fatal", False)
                last_of_run = index == last_index and position == len(ran_checks) - 1
                if failed or res.halt or last_of_run:
                    res.event.what_we_saw.update(
                        summarize_steps(
                            steps, elapsed_time_ms, failed_check_id=chk.check_id if failed else None
                        )
                    )

                await self.sink.emit(res.event)
                events.append(res.event)
                if self.progress is not None:
                    self.progress.step_finished(current_ctx, chk.check_id, res.event, res.passed)

                current_ctx = self._apply(current_ctx, ran, parallel)
                latest[0] = current_ctx

                if failed:
                    cancel_pending_collateral_prefetch(current_ctx)
                    return False, events, current_ctx

                if res.halt:
                    cancel_pending_collateral_prefetch(current_ctx)
                    return True, events, current_ctx

        return True, events, current_ctx
