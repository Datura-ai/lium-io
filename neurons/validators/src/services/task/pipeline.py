import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Tuple, runtime_checkable

import asyncssh
from pydantic import BaseModel, Field

from datura.requests.miner_requests import ExecutorSSHInfo

from core.utils import _m, get_extra_info
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

logger = logging.getLogger(__name__)

# The one `[local_verify] outcome` line every phase-2 probe reading writes (`step=rental_probe`);
# Loki counts by (outcome, reason). The checks import it from here.
RENTAL_PROBE_OUTCOME_EVENT = "[local_verify] outcome"


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
    # liumd phase 2: the caller's word alone — this is the executor's first, unscored verification
    # (the express lane, DAH-2958), whether or not FIRST_PASS_FAST_PATH_ENABLED sizes the probes.
    # The one-call `/verify` gate reads this, never `first_pass`: the saving is per cycle kind, not
    # per probe size, and the two flags stay independent.
    unscored: bool = False


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
    executor_image_report: ExecutorImageReport | None = None
    # liumd phase 1 (DAH-2834): what `POST /verify` answered this cycle, already judged. None =
    # not attempted or fell back entirely; the capability and VerifyX checks consume a judged
    # step when present and run over SSH otherwise.
    local_verify: LocalVerifyOutcome | None = None
    # liumd phase 2: the executor's read-only host facts from the early facts-only `POST /verify`
    # (services.local_verify_facts.LocalFacts), bounded and parsed. None = not asked or unusable;
    # the stale cleanup and the port selector read them when present and run their SSH listings
    # otherwise. Never a verdict: every fact only narrows what the SSH-proven steps go on to do.
    local_facts: Any | None = None


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
    # DAH-3386: what the check that cleared the verified job saw — reason_code, check_id and, for a pod found not
    # running, the container's death diagnostics. Sent to the backend with the reset so the penalty it raises
    # carries the evidence (lium-platform DAH-3385). The pipeline fills reason_code/check_id from the event when
    # the check did not set it itself.
    clear_verified_job_evidence: dict[str, Any] | None = None
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


class Pipeline:
    def __init__(self, checks: List[Check], sink: EventSink):
        self.checks = checks
        self.sink = sink

    async def run(self, ctx: Context) -> Tuple[bool, list[ValidationEvent], Context]:
        events: list[ValidationEvent] = []
        current_ctx = ctx
        pipeline_start_time = time.perf_counter()
        steps: list[tuple[str, int]] = []
        last_index = len(self.checks) - 1

        try:
            for index, chk in enumerate(self.checks):
                check_start_time = time.perf_counter()
                res = await chk.run(current_ctx)
                check_end_time = time.perf_counter()

                execution_time_ms = int((check_end_time - check_start_time) * 1000)
                elapsed_time_ms = int((check_end_time - pipeline_start_time) * 1000)

                res.event.context["execution_time_ms"] = execution_time_ms
                res.event.context["elapsed_time_ms"] = elapsed_time_ms
                steps.append((chk.check_id, execution_time_ms))

                failed = not res.passed and getattr(chk, "fatal", False)
                if failed or res.halt or index == last_index:
                    res.event.what_we_saw.update(
                        summarize_steps(
                            steps, elapsed_time_ms, failed_check_id=chk.check_id if failed else None
                        )
                    )

                await self.sink.emit(res.event)
                events.append(res.event)

                if res.updates:
                    current_ctx = current_ctx.model_copy(
                        update=updates_with_clear_verified_job_evidence(res, chk.check_id)
                    )

                if failed:
                    return False, events, current_ctx

                if res.halt:
                    return True, events, current_ctx

            return True, events, current_ctx
        finally:
            await _settle_background_work(current_ctx)


async def _settle_background_work(ctx: Context) -> None:
    """liumd phase 2/3: a check may leave work in flight for a later check — the rental probe task
    in `ctx.state.local_verify` (2b), the DinD container the executor started for the port check in
    `ctx.state.local_facts.dind` (2c) and the early GPU call in `ctx.state.local_verify.pending`
    (phase 3). A fatal check or a halt in between would leave it pending — a probe pod the backend
    already rented, a container holding a rental port, a GPU answer nobody judges — so whatever is
    still unconsumed is settled here, once, whatever ended the pipeline: cancelled or awaited
    (`BackgroundProbe.cancel_and_await`), the probe's `health_check_*` container force-removed
    (DAH-1991), the DinD container removed. The pipeline's own result is never replaced."""
    await _remove_unconsumed_dind(ctx)
    outcome = ctx.state.local_verify
    if outcome is None:
        return
    # Phase 3's early GPU call and phase 2b's rental probe: the same shape, settled the same way.
    for work, step in ((outcome.pending, "gpu_early"), (outcome.rental_probe, "rental_probe")):
        if work is None or work.consumed:
            continue
        work.consumed = True
        try:
            reason = await work.cancel_and_await()
        except Exception as exc:  # noqa: BLE001 — cleanup must not replace the pipeline's own result
            reason = f"settle_error: {type(exc).__name__}"
        extra = {
            **ctx.default_extra,
            "outcome": "fallback",
            "step": step,
            "reason": f"unconsumed_{reason}",
            "first_pass": ctx.config.first_pass,
        }
        if step == "rental_probe":
            # the backend spawned a health_check_* pod for the probe; only RentalVerificationCheck
            # removed it before (DAH-1991), so an unconsumed probe left its pod on the executor
            try:
                extra["health_checks_removed"] = await ctx.services.container_cleanup.force_remove_health_checks(
                    ctx.ssh, ctx.executor.uuid
                )
            except Exception as exc:  # noqa: BLE001 — same: the pipeline's own result stands
                extra["health_checks_removed"] = f"error: {type(exc).__name__}"
        logger.info(_m(RENTAL_PROBE_OUTCOME_EVENT, extra=get_extra_info(extra)))


DIND_SETTLE_TIMEOUT_SECONDS = 15


# Facts-call outcomes under which the executor provably ran nothing of ours: the intent was refused
# before any step (401 / 409), never understood (no capability), or the executor's own `dind` step
# answered `skipped` / `failed` (its by-label cleanup took its half-made container). Nothing to remove —
# and the name is derived from the miner hotkey and the port, the same for every validator probing
# that miner, so a container that IS there under it is another validator's probe, never ours.
DIND_NEVER_STARTED_REASONS = frozenset({"refused", "busy_or_replay", "not_supported", "skipped", "failed"})


async def _remove_unconsumed_dind(ctx: Context) -> None:
    """Remove the DinD container the validator asked the executor to start from the facts intent
    when no probe took it (the port check never ran, ran before the facts arrived, or the answer was
    lost after the executor may have started it: timeout, transport, http_error, a malformed answer
    or a schema/nonce/executor mismatch in it, a mismatched echo, a missing step). The name is the validator's own choice, so a `docker rm -f` of
    it is safe whether or not the container exists; best effort over the pipeline's SSH — the
    executor's TTL and the stale cleanup (`container_` prefix) are the backstops. An outcome under
    which the executor ran nothing of ours (`DIND_NEVER_STARTED_REASONS`) removes nothing: a
    same-named container then is another validator's."""
    facts = ctx.state.local_facts
    dind = facts.dind if facts is not None else None
    if dind is None or dind.consumed:
        return
    dind.consumed = True
    reason = "removed"
    if not dind.started and dind.reason in DIND_NEVER_STARTED_REASONS:
        reason = f"never_started_{dind.reason}"
    elif ctx.ssh is None:
        reason = "no_ssh"
    else:
        try:
            await asyncio.wait_for(
                ctx.ssh.run(f"/usr/bin/docker rm -fv {dind.name}"), timeout=DIND_SETTLE_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 — cleanup must not replace the pipeline's own result
            reason = f"rm_error: {type(exc).__name__}"
    logger.info(
        _m(
            RENTAL_PROBE_OUTCOME_EVENT,
            extra={
                **ctx.default_extra,
                "outcome": "fallback",
                "step": "dind",
                "reason": f"unconsumed_{reason}",
                "started": dind.started,
                "first_pass": ctx.config.first_pass,
            },
        )
    )
